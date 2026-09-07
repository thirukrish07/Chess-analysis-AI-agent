"""
================================================================================
 AI Chess Coach & Blunder Detection System
================================================================================

An automated Streamlit application that:
  1. Accepts a chess game/position via PGN text, PGN/TXT file upload,
     raw FEN string, or a screenshot image of a chessboard.
  2. Evaluates every position with a local Stockfish engine.
  3. Flags "blunders" -- moves whose evaluation drop exceeds a
     user-configurable centipawn threshold.
  4. Sends flagged positions to Google's Gemini (gemini-3.6-flash) to 
     generate encouraging, Grandmaster-style coaching explanations.

--------------------------------------------------------------------------------
SETUP INSTRUCTIONS
--------------------------------------------------------------------------------

1. Install Python dependencies:

    pip install streamlit python-chess google-genai pillow python-dotenv tenacity

2. Create a .env file in the root directory:

    GEMINI_API_KEY="your-api-key-here"
    STOCKFISH_PATH="stockfish.exe"

3. Install the Stockfish engine binary:
    Download from https://stockfishchess.org/download/
    and place the binary/exe path into your .env file or specify it in settings.

4. Run the app:

    streamlit run app.py

--------------------------------------------------------------------------------
"""

import io
import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine
import chess.pgn
import streamlit as st
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

# Load environment variables from .env file
load_dotenv()

# google-genai SDK (official Gemini SDK)
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ==============================================================================
# CONSTANTS & RETRY PREDICTORS
# ==============================================================================

GEMINI_MODEL = "gemini-3.6-flash"
MATE_SCORE_CP = 100_000  # centipawn stand-in used when the engine reports mate
DEFAULT_DEPTH = 12
DEFAULT_THRESHOLD = 150


def _is_transient_error(exception: Exception) -> bool:
    """Predicate to retry on transient 503 or quota/rate errors."""
    err_str = str(exception)
    return "503" in err_str or "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str or "429" in err_str


@dataclass
class BlunderRecord:
    """Container for a single detected blunder."""
    move_number: int
    side: str                 # "White" or "Black"
    move_san: str
    fen_before: str
    fen_after: str
    eval_before_cp: int
    eval_after_cp: int
    cp_drop: int
    best_move_san: str
    best_move_uci: str
    coaching_text: Optional[str] = None


# ==============================================================================
# SECTION 1: INPUT PARSING (PGN / FEN)
# ==============================================================================

def parse_pgn(pgn_text: str) -> chess.pgn.Game:
    """Parse a PGN string into a chess.pgn.Game object."""
    if not pgn_text or not pgn_text.strip():
        raise ValueError("PGN input is empty.")

    pgn_stream = io.StringIO(pgn_text)
    try:
        game = chess.pgn.read_game(pgn_stream)
    except Exception as exc:
        raise ValueError(f"Failed to parse PGN: {exc}") from exc

    if game is None:
        raise ValueError(
            "No valid game found in the PGN text. Check the formatting "
            "(headers, move list, and result tag)."
        )

    return game


def parse_fen(fen_text: str) -> chess.Board:
    """Parse a raw FEN string into a chess.Board object."""
    if not fen_text or not fen_text.strip():
        raise ValueError("FEN input is empty.")

    fen_text = fen_text.strip()
    try:
        board = chess.Board(fen_text)
    except Exception as exc:
        raise ValueError(f"Invalid FEN string: {exc}") from exc

    if not board.is_valid():
        raise ValueError(
            "FEN parsed but represents an illegal/invalid chess position "
            "(e.g. missing kings, too many pieces, etc.)."
        )

    return board


# ==============================================================================
# SECTION 2: IMAGE -> FEN VIA GEMINI VISION
# ==============================================================================

FEN_VISION_PROMPT = """You are a computer-vision chess-position extractor.

Look carefully at the chessboard image provided. Determine the exact
position of every piece on the 8x8 board.

Output ONLY a single valid FEN (Forsyth-Edwards Notation) string that
represents the board position shown in the image.

Rules:
- Output ONLY the FEN string. No explanation, no markdown, no code fences.
- Default to: side to move = white, full castling rights "KQkq" if both kings/rooks
  appear on their home squares (otherwise "-"), en-passant target "-", halfmove clock "0",
  and fullmove number "1" if uncertain.
"""

FEN_REGEX = re.compile(
    r"([pnbrqkPNBRQK1-8]+(?:/[pnbrqkPNBRQK1-8]+){7})\s+([wb])\s+"
    r"([KQkq-]+)\s+(-|[a-h][36])\s+(\d+)\s+(\d+)"
)


def _extract_fen_from_text(text: str) -> str:
    """Pull the first FEN-looking substring out of a model response."""
    text = text.strip().strip("`").strip()
    match = FEN_REGEX.search(text)
    if match:
        return match.group(0)
    return text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(min=2, max=10),
    retry=retry_if_exception(_is_transient_error),
    reraise=True
)
def image_to_fen(image_bytes: bytes, api_key: str, mime_type: str = "image/png") -> str:
    """Send a chessboard screenshot to Gemini Flash with tenacity retries."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-genai SDK is not installed. Run: pip install google-genai")

    try:
        client = genai.Client(api_key=api_key)
        image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[image_part, FEN_VISION_PROMPT],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=500,
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini vision request failed: {exc}") from exc

    raw_text = (response.text or "").strip()
    if not raw_text:
        raise ValueError("Gemini returned an empty response for the board image.")

    candidate_fen = _extract_fen_from_text(raw_text)

    try:
        board = chess.Board(candidate_fen)
    except Exception as exc:
        raise ValueError(
            f"Gemini's output could not be parsed as a valid FEN "
            f"(raw output: '{raw_text}'). Error: {exc}"
        ) from exc

    if not board.is_valid():
        raise ValueError(f"Gemini produced an invalid FEN position: '{candidate_fen}'.")

    return candidate_fen


# ==============================================================================
# SECTION 3: STOCKFISH ENGINE EVALUATION
# ==============================================================================

def find_stockfish_path(user_path: str = "") -> Optional[str]:
    """Resolve a usable Stockfish binary path."""
    if user_path and os.path.isfile(user_path) and os.access(user_path, os.X_OK):
        return user_path
        
    env_path = os.environ.get("STOCKFISH_PATH", "")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    auto = shutil.which("stockfish")
    if auto:
        return auto

    for name in ("stockfish.exe", "stockfish-ubuntu-x86-64", "stockfish_x64"):
        found = shutil.which(name)
        if found:
            return found
    return None


def score_to_cp(score: chess.engine.PovScore, pov: chess.Color) -> int:
    """Convert a python-chess PovScore into a plain centipawn integer."""
    pov_score = score.pov(pov)
    cp = pov_score.score(mate_score=MATE_SCORE_CP)
    return int(cp) if cp is not None else 0


def evaluate_with_stockfish(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    pov: Optional[chess.Color] = None,
) -> int:
    """Evaluate a board position with Stockfish."""
    if pov is None:
        pov = board.turn
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
    except Exception as exc:
        raise RuntimeError(f"Stockfish analysis failed: {exc}") from exc

    return score_to_cp(info["score"], pov)


def best_move_for_position(
    board: chess.Board, engine: chess.engine.SimpleEngine, depth: int
) -> chess.Move:
    """Ask Stockfish for its preferred move in a position."""
    try:
        result = engine.play(board, chess.engine.Limit(depth=depth))
    except Exception as exc:
        raise RuntimeError(f"Stockfish move search failed: {exc}") from exc
    if result.move is None:
        raise RuntimeError("Stockfish did not return a move.")
    return result.move


def analyze_game_for_blunders(
    game: chess.pgn.Game,
    engine: chess.engine.SimpleEngine,
    depth: int,
    threshold_cp: int,
    progress_callback=None,
) -> list:
    """Walk through every move of a parsed PGN game and detect blunders."""
    blunders = []
    board = game.board()
    mainline_moves = list(game.mainline_moves())
    total_moves = len(mainline_moves)

    if total_moves == 0:
        return blunders

    for idx, move in enumerate(mainline_moves):
        mover_color = board.turn
        mover_name = "White" if mover_color == chess.WHITE else "Black"
        fen_before = board.fen()
        move_number_display = board.fullmove_number

        eval_before = evaluate_with_stockfish(board, engine, depth, pov=mover_color)

        try:
            best_move = best_move_for_position(board, engine, depth)
            best_move_san = board.san(best_move)
            best_move_uci = best_move.uci()
        except Exception:
            best_move_san = "N/A"
            best_move_uci = "N/A"

        move_san = board.san(move)
        board.push(move)
        fen_after = board.fen()

        eval_after = evaluate_with_stockfish(board, engine, depth, pov=mover_color)
        cp_drop = eval_before - eval_after

        if cp_drop > threshold_cp:
            blunders.append(
                BlunderRecord(
                    move_number=move_number_display,
                    side=mover_name,
                    move_san=move_san,
                    fen_before=fen_before,
                    fen_after=fen_after,
                    eval_before_cp=eval_before,
                    eval_after_cp=eval_after,
                    cp_drop=cp_drop,
                    best_move_san=best_move_san,
                    best_move_uci=best_move_uci,
                )
            )

        if progress_callback:
            progress_callback((idx + 1) / total_moves)

    return blunders


def analyze_single_position(
    board: chess.Board, engine: chess.engine.SimpleEngine, depth: int
) -> dict:
    """Evaluate a single standalone position."""
    pov = board.turn
    eval_cp = evaluate_with_stockfish(board, engine, depth, pov=pov)
    best_move = best_move_for_position(board, engine, depth)
    best_move_san = board.san(best_move)
    return {
        "fen": board.fen(),
        "side_to_move": "White" if pov == chess.WHITE else "Black",
        "eval_cp": eval_cp,
        "best_move_san": best_move_san,
        "best_move_uci": best_move.uci(),
    }


# ==============================================================================
# SECTION 4: GEMINI COACHING INSIGHTS (WITH TENACITY RETRIES)
# ==============================================================================

COACH_SYSTEM_INSTRUCTION = """You are "Coach Caissa", a direct, highly analytical Grandmaster chess coach.

Rules:
1. Do NOT write introductory greetings, conversational fluff, or friendly meta-announcements.
2. Jump straight into the bullet points.
3. Complete every single bullet point and sentence cleanly without cutting off mid-thought.
"""


def _cp_to_readable(cp: int) -> str:
    if abs(cp) >= MATE_SCORE_CP - 1000:
        return "a forced mate"
    pawns = cp / 100.0
    sign = "+" if pawns >= 0 else ""
    return f"{sign}{pawns:.2f}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(min=2, max=10),
    retry=retry_if_exception(_is_transient_error),
    reraise=True
)
def get_gemini_coaching(
    api_key: str,
    fen_before: str,
    move_played_san: str,
    best_move_san: str,
    eval_before_cp: int,
    eval_after_cp: int,
    side: str,
) -> str:
    """Send one flagged blunder to Gemini with automatic tenacity retries."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-genai SDK is not installed.")

    board = chess.Board(fen_before)
    ascii_board = str(board)

    prompt = f"""Analyze this blunder made by {side}.

Board Snapshot (ASCII View):
{ascii_board}

Position BEFORE move (FEN): {fen_before}
Move played: {move_played_san}
Recommended move: {best_move_san}
Eval before: {_cp_to_readable(eval_before_cp)} pawns
Eval after: {_cp_to_readable(eval_after_cp)} pawns

Provide a concise breakdown starting immediately with bullet points:
* **Tactical Flaw**: What went wrong with {move_played_san}?
* **Why {best_move_san} is Better**: What specific advantage does this move secure?
* **Coach's Principle**: One short tactical takeaway to remember.
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=COACH_SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini coaching request failed: {exc}") from exc

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty coaching response.")
    return text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(min=2, max=10),
    retry=retry_if_exception(_is_transient_error),
    reraise=True
)
def get_gemini_position_commentary(
    api_key: str,
    fen: str,
    side_to_move: str,
    eval_cp: int,
    best_move_san: str,
) -> str:
    """Coaching commentary for a single standalone position with tenacity retries."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-genai SDK is not installed.")

    board = chess.Board(fen)
    ascii_board = str(board)

    prompt = f"""Analyze this chess position for {side_to_move}.

Board Snapshot (ASCII View):
{ascii_board}

FEN: {fen}
Side to Move: {side_to_move}
Stockfish Evaluation: {_cp_to_readable(eval_cp)} pawns
Stockfish Recommended Best Move: {best_move_san}

Provide a structured breakdown starting immediately with bullet points:
* **Position Snapshot**: Describe the central tension or key piece placement in 2 full sentences.
* **Why {best_move_san} is Best**: Explain what taking or moving here accomplishes in 2 full sentences.
* **Key Takeaway**: One actionable chess principle for this exact setup.
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=COACH_SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini coaching request failed: {exc}") from exc

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty coaching response.")
    return text


# ==============================================================================
# SECTION 5: STREAMLIT APP
# ==============================================================================

def get_api_key() -> str:
    """Resolve the Gemini API key from Streamlit secrets, .env file, or environment."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def render_eval_metric(label: str, cp_value: int):
    pawns = cp_value / 100.0
    st.metric(label, f"{pawns:+.2f}")


def render_eval_metric_col(col, label: str, cp_value: int):
    pawns = cp_value / 100.0
    col.metric(label, f"{pawns:+.2f}")


def _clear_blunder_state():
    """Wipes all analysis cache and past generated coaching text."""
    keys_to_delete = [
        "input_mode",
        "game_obj",
        "fen_board",
        "blunders",
        "single_position_result",
        "position_coaching",
    ]
    for key in list(st.session_state.keys()):
        if key in keys_to_delete or key.startswith("coach_text_"):
            st.session_state.pop(key, None)


# ------------------------------------------------------------------------------
# CALLBACK FUNCTIONS FOR CLEAR BUTTONS
# ------------------------------------------------------------------------------
def clear_pgn_callback():
    st.session_state["pgn_text_input"] = ""
    _clear_blunder_state()

def clear_fen_callback():
    st.session_state["fen_text_input"] = ""
    _clear_blunder_state()

def clear_image_callback():
    _clear_blunder_state()
    st.session_state.pop("image_file_uploader", None)


def render_blunder_card(idx: int, record: BlunderRecord, api_key: str):
    header = (
        f"Move {record.move_number} ({record.side}): {record.move_san}  "
        f"\u2014  dropped {record.cp_drop / 100:.2f} pawns"
    )
    with st.expander(header, expanded=(idx == 0)):
        col1, col2, col3 = st.columns(3)
        with col1:
            render_eval_metric("Eval before", record.eval_before_cp)
        with col2:
            render_eval_metric("Eval after", record.eval_after_cp)
        with col3:
            st.metric("CP dropped", f"-{record.cp_drop}")

        st.markdown(f"**Stockfish's suggested move instead:** `{record.best_move_san}`")

        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Position before the move")
                st.code(record.fen_before, language="text")
            with c2:
                st.caption("Position after the move")
                st.code(record.fen_after, language="text")

        st.divider()
        st.markdown("#### 🧠 Coach's Insight")

        button_key = f"coach_btn_{idx}"
        cache_key = f"coach_text_{idx}"

        if cache_key in st.session_state and st.session_state[cache_key]:
            st.markdown(st.session_state[cache_key])
            if st.button("🗑️ Clear / Regenerate Insight", key=f"clear_btn_{idx}"):
                st.session_state.pop(cache_key, None)
                st.rerun()
        else:
            if not api_key:
                st.warning(
                    "No Gemini API key configured -- set GEMINI_API_KEY in your .env "
                    "to unlock AI coaching explanations."
                )
            elif st.button("Get Coaching Explanation", key=button_key):
                with st.spinner("Coach is reviewing the position (retrying automatically if busy)..."):
                    try:
                        explanation = get_gemini_coaching(
                            api_key=api_key,
                            fen_before=record.fen_before,
                            move_played_san=record.move_san,
                            best_move_san=record.best_move_san,
                            eval_before_cp=record.eval_before_cp,
                            eval_after_cp=record.eval_after_cp,
                            side=record.side,
                        )
                        st.session_state[cache_key] = explanation
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Coaching request failed after retries: {exc}")


def main():
    st.set_page_config(
        page_title="AI Chess Coach & Blunder Detector",
        page_icon="\u265E",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("\u265E AI Chess Coach & Blunder Detection System")
    st.caption(
        "Upload a game or position, let Stockfish find the mistakes, and get "
        "Grandmaster-style coaching from Gemini on how to improve."
    )

    # ---------------------------------------------------------------
    # IN-PAGE CONFIGURATION (EXPANDABLE PANEL)
    # ---------------------------------------------------------------
    with st.expander("⚙️ Settings & Configuration", expanded=False):
        cfg_col1, cfg_col2 = st.columns(2)
        
        with cfg_col1:
            stockfish_path_input = st.text_input(
                "Stockfish binary path (optional)",
                value="",
                placeholder="Leave blank to use .env or auto-detect from PATH",
            )
            resolved_path = find_stockfish_path(stockfish_path_input)
            if resolved_path:
                st.success(f"Stockfish found: `{resolved_path}`")
            else:
                st.error("Stockfish binary not found. Set STOCKFISH_PATH in .env or above.")

            api_key = get_api_key()
            if api_key:
                st.success("Gemini API key detected.")
            else:
                st.warning("No GEMINI_API_KEY found in .env file or environment.")

        with cfg_col2:
            depth = st.slider("Stockfish Depth", min_value=8, max_value=18, value=DEFAULT_DEPTH)
            threshold_cp = st.slider(
                "Blunder Threshold (centipawns)",
                min_value=50,
                max_value=300,
                value=DEFAULT_THRESHOLD,
                step=10,
            )
            if st.button("Clear Past Output & Reset", use_container_width=True, type="secondary"):
                _clear_blunder_state()
                st.rerun()

    st.divider()

    # ---------------------------------------------------------------
    # MAIN INPUT TABS
    # ---------------------------------------------------------------
    tab_pgn, tab_fen, tab_image = st.tabs(
        ["\U0001F4C4 PGN Input", "\U0001F522 FEN Input", "\U0001F5BC\ufe0f Image Upload"]
    )

    with tab_pgn:
        st.subheader("Paste PGN or Upload a File")
        
        pgn_text_area = st.text_area(
            "Paste PGN text here",
            value=st.session_state.get("pgn_text_input", ""),
            height=200,
            placeholder='[Event "Casual Game"]\n1. e4 e5 2. Nf3 Nc6 3. Bb5 ...',
            key="pgn_text_input"
        )
        
        c_clear_pgn, _ = st.columns([1, 5])
        with c_clear_pgn:
            st.button("🧹 Clear PGN", key="clear_pgn_input_btn", on_click=clear_pgn_callback)

        pgn_file = st.file_uploader("...or upload a .pgn / .txt file", type=["pgn", "txt"], key="pgn_file_uploader")

        pgn_source_text = None
        if pgn_file is not None:
            try:
                pgn_source_text = pgn_file.read().decode("utf-8", errors="replace")
            except Exception as exc:
                st.error(f"Could not read uploaded file: {exc}")
        elif pgn_text_area.strip():
            pgn_source_text = pgn_text_area

        if st.button("Analyze PGN Game", type="primary", key="analyze_pgn_btn"):
            if not pgn_source_text:
                st.error("Please paste PGN text or upload a file first.")
            else:
                try:
                    game_obj = parse_pgn(pgn_source_text)
                    _clear_blunder_state()
                    st.session_state["input_mode"] = "pgn"
                    st.session_state["game_obj"] = game_obj
                    st.success("PGN parsed successfully.")
                except Exception as exc:
                    st.error(f"PGN error: {exc}")

    with tab_fen:
        st.subheader("Paste a FEN String")
        
        col_fen, col_btn = st.columns([4, 1])
        with col_fen:
            fen_text_input = st.text_input(
                "FEN",
                value=st.session_state.get("fen_text_input", ""),
                placeholder="r1bq1rk1/ppp2ppp/2n2n2/3pp3/3P4/2P1PN2/PP3PPP/RNBQ1RK1 w - - 4 8",
                key="fen_text_input"
            )
        with col_btn:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            st.button("🧹 Clear", key="clear_fen_input_btn", on_click=clear_fen_callback)

        if st.button("Analyze FEN Position", type="primary", key="analyze_fen_btn"):
            if not fen_text_input.strip():
                st.error("Please paste a FEN string first.")
            else:
                try:
                    fen_board = parse_fen(fen_text_input)
                    _clear_blunder_state()
                    st.session_state["input_mode"] = "fen"
                    st.session_state["fen_board"] = fen_board
                    st.success("FEN parsed successfully.")
                except Exception as exc:
                    st.error(f"FEN error: {exc}")

    with tab_image:
        st.subheader("Upload a Chessboard Screenshot")
        image_file = st.file_uploader("Upload .png / .jpg / .jpeg", type=["png", "jpg", "jpeg"], key="image_file_uploader")
        
        if image_file is not None:
            st.image(image_file, caption="Uploaded image preview", width=350)
            st.button("🧹 Remove Image", key="clear_image_input_btn", on_click=clear_image_callback)

        if st.button("Detect Position & Analyze", type="primary", key="analyze_image_btn"):
            if image_file is None:
                st.error("Please upload an image first.")
            elif not api_key:
                st.error("Gemini API key is required for image detection.")
            else:
                try:
                    image_bytes = image_file.getvalue()
                    mime_type = image_file.type or "image/png"
                    with st.spinner("Gemini is reading the board..."):
                        detected_fen = image_to_fen(image_bytes, api_key, mime_type)
                    
                    _clear_blunder_state()
                    st.success(f"Detected FEN: `{detected_fen}`")
                    fen_board = chess.Board(detected_fen)
                    st.session_state["input_mode"] = "fen"
                    st.session_state["fen_board"] = fen_board
                except Exception as exc:
                    st.error(f"Image-to-FEN detection failed: {exc}")

    st.divider()

    # ---------------------------------------------------------------
    # ANALYSIS EXECUTION
    # ---------------------------------------------------------------
    mode = st.session_state.get("input_mode")

    if mode is None:
        st.info("Choose an input method above to begin analysis.")
        return

    if not resolved_path:
        st.error("Cannot run analysis: Stockfish binary not found.")
        return

    try:
        engine = chess.engine.SimpleEngine.popen_uci(resolved_path)
    except Exception as exc:
        st.error(f"Failed to start Stockfish engine: {exc}")
        return

    try:
        if mode == "pgn":
            game_obj = st.session_state.get("game_obj")
            if "blunders" not in st.session_state:
                progress_bar = st.progress(0.0, text="Analyzing moves with Stockfish...")

                def _update_progress(frac):
                    progress_bar.progress(frac, text=f"Analyzing moves... {int(frac * 100)}%")

                with st.spinner("Running Stockfish analysis..."):
                    blunders = analyze_game_for_blunders(
                        game_obj, engine, depth, threshold_cp, progress_callback=_update_progress
                    )
                    st.session_state["blunders"] = blunders
                progress_bar.empty()

            blunders = st.session_state.get("blunders", [])

            st.header("\U0001F4CA Analysis Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Moves Analyzed", len(list(game_obj.mainline_moves())))
            c2.metric("Blunders Detected", len(blunders))
            c3.metric("Threshold Used", f"{threshold_cp} cp")

            if not blunders:
                st.success("\U0001F389 No blunders detected above threshold!")
            else:
                st.subheader("\u26A0\ufe0f Detected Blunders")
                for i, record in enumerate(blunders):
                    render_blunder_card(i, record, api_key)

        elif mode == "fen":
            fen_board = st.session_state.get("fen_board")
            if "single_position_result" not in st.session_state:
                with st.spinner("Evaluating position..."):
                    result = analyze_single_position(fen_board, engine, depth)
                    st.session_state["single_position_result"] = result

            result = st.session_state["single_position_result"]

            st.header("\U0001F4CA Position Analysis")
            c1, c2, c3 = st.columns(3)
            c1.metric("Side to Move", result["side_to_move"])
            render_eval_metric_col(c2, "Evaluation", result["eval_cp"])
            c3.metric("Best Move", result["best_move_san"])

            st.code(result["fen"], language="text")

            st.divider()
            st.markdown("#### 🧠 Coach's Insight")

            if "position_coaching" in st.session_state and st.session_state["position_coaching"]:
                st.markdown(st.session_state["position_coaching"])
                
                if st.button("🗑️ Clear Response & Re-run Coach", key="clear_single_pos_btn"):
                    st.session_state.pop("position_coaching", None)
                    st.rerun()
            elif not api_key:
                st.warning("No Gemini API key configured.")
            else:
                if st.button("Get Coaching Explanation", key="single_pos_coach_btn"):
                    with st.spinner("Coach is reviewing the position (retrying automatically if busy)..."):
                        try:
                            explanation = get_gemini_position_commentary(
                                api_key=api_key,
                                fen=result["fen"],
                                side_to_move=result["side_to_move"],
                                eval_cp=result["eval_cp"],
                                best_move_san=result["best_move_san"],
                            )
                            st.session_state["position_coaching"] = explanation
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Coaching request failed after retries: {exc}")

    finally:
        try:
            engine.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
# ♟️ AI Chess Coach & Blunder Detection System

An interactive AI-powered chess coaching application built with **Streamlit**, **Google Gemini**, and the **Stockfish** chess engine. This tool analyzes chess games via PGN text, FEN strings, or board images, identifying critical blunders and delivering grandmaster-level coaching feedback.

---

## ✨ Features

* **Multi-Input Game Loading**: Import chess positions or full games via PGN text paste, FEN string, or direct board image uploads.
* **Stockfish Blunder Detection**: Automated engine analysis that flags moves with centipawn drops exceeding customizable thresholds.
* **AI Coaching Insights**: Integrated with Google Gemini (`google-genai`) to generate clear, plain-English strategic recommendations and tactical explanations.
* **Interactive UI**: Custom Streamlit layout featuring smooth board visualization, move navigation, and configuration expanders.

---

## 🛠️ Tech Stack

* **Frontend/App Framework**: [Streamlit](https://streamlit.io/)
* **AI Engine**: [Google Gemini AI API](https://ai.google.dev/) (`google-genai`)
* **Chess Engine**: [Stockfish](https://stockfishchess.org/)
* **Chess Utilities**: `python-chess`
* **Language**: Python 3.10+

---

## 🚀 Getting Started Locally

### Prerequisites
* Python installed on your machine.
* Stockfish engine binary installed locally.
* A Google Gemini API Key.

### Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/thirukrish07/Chess-analysis-AI-agent.git](https://github.com/thirukrish07/Chess-analysis-AI-agent.git)
   cd Chess-analysis-AI-agent

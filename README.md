# Local RAG Chatbot with Llama 3

## Overview
A local Retrieval-Augmented Generation (RAG) application that lets you chat with your own documents using Llama 3 running entirely on your machine. No API keys, no cloud, no data leaving your computer.

---

## How It Works

1. **Load Documents** — reads your PDF, TXT, MD, and DOCX files from the `data/` folder
2. **Create Vector Index** — converts documents into searchable vector embeddings using a local HuggingFace model
3. **Query** — when you ask a question, the app finds the most relevant chunks from your documents and passes them to Llama 3 to generate an answer

---

## Project Structure
```
project/
│
├── data/                  # Put your documents here (.pdf, .txt, .md, .docx)
├── app.py                 # Terminal-based chat interface
├── webchat.py             # Streamlit web chat interface
└── requirements.txt
```

---

## Prerequisites
- [Ollama](https://ollama.com) installed and running
- Llama 3 pulled locally:
```bash
ollama pull llama3
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create and activate virtual environment
```bash
conda create --name ragenv python=3.12
conda activate ragenv
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your documents
Place your files inside the `data/` folder. Supported formats: `.pdf`, `.txt`, `.md`, `.docx`

---

## Usage

### Terminal version
```bash
python app.py
```
Type your question and press Enter. Type `exit` to quit.

### Web UI version
```bash
streamlit run webchat.py
```
Opens a chat interface in your browser.

---

## Tech Stack
- **LLM** — Llama 3 via Ollama (runs locally)
- **RAG Framework** — LlamaIndex
- **Embeddings** — BAAI/bge-small-en-v1.5 (HuggingFace, runs locally)
- **Web UI** — Streamlit

---

## Notes
- First run takes longer — it downloads the embedding model and indexes your documents
- Everything runs locally — no internet connection needed after setup
- The web UI caches the index so it only builds once per session


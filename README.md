# AskMyDocs 📄🔍

A private document Q&A app using Retrieval-Augmented Generation (RAG) and LLMs.

## Requirements
- Python 3.11+


## 🔧 Features
- Upload and index PDF documents
- Ask questions and get context-aware answers
- Powered by LangChain, Chroma, and Mistral (via Ollama)

## 🚀 Stack
- Python
- LangChain
- ChromaDB
- SentenceTransformers
- Mistral via Ollama or OpenAI
- Streamlit or FastAPI

## 📦 Setup

```bash
git clone https://github.com/yourusername/askmydocs
cd askmydocs
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
cp .env.example .env


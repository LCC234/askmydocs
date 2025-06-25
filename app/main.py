# app/main.py

import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import DATA_DIR, VECTOR_DIR
from app.document_loader import extract_text_from_pdf
from app.embedder import get_vectorstore
from app.retriever import get_relevant_docs
from app.llm_chain import generate_prompt, ask_llm


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks


def index_document(vectorstore, chunks, metadata):
    # Add texts to vectorstore with metadata for traceability
    metadatas = [{"source": metadata} for _ in chunks]
    vectorstore.add_texts(chunks, metadatas=metadatas)


def process_question(pdf_filename, question, llm_provider="openai"):
    file_path = os.path.join(DATA_DIR, pdf_filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("🔍 Extracting text from PDF...")
    text = extract_text_from_pdf(file_path)

    logger.info("✂️ Splitting text into chunks...")
    chunks = chunk_text(text)

    logger.info("📚 Initializing vector store...")
    vectorstore = get_vectorstore(VECTOR_DIR)

    logger.info("📥 Indexing document chunks in vector store...")
    index_document(vectorstore, chunks, metadata=pdf_filename)

    logger.info("📡 Retrieving relevant documents...")
    context_docs = get_relevant_docs(vectorstore, question)

    logger.info("📝 Building prompt...")
    prompt = generate_prompt(context_docs, question)

    logger.info(f"🤖 Querying LLM provider: {llm_provider}...")
    answer = ask_llm(prompt, provider=llm_provider)

    return answer


if __name__ == "__main__":
    print("=== AskMyDocs ===")
    pdf = input("Enter PDF file name (in data/uploaded_docs): ").strip()
    question = input("What would you like to know? ").strip()

    try:
        answer = process_question(pdf, question)
        print("\n🧠 Answer:")
        print(answer)
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        print(f"Error: {e}")
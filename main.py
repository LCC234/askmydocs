# app/main.py

import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.config import BASE_DIR, DATA_DIR, VECTOR_DIR
from app.document_loader import extract_text_from_pdf
from app.embedder import get_vectorstore
from app.retriever import get_relevant_docs
from app.llm_chain import generate_prompt, ask_llm

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')






DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# check sqlite3 version
import sqlite3
logger.info(f"SQLite version: {sqlite3.sqlite_version}")


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks


def index_document(vectorstore :  Chroma, chunks, metadata):
    # Add texts to vectorstore with metadata for traceability
    metadatas = [{"source": metadata} for _ in chunks]
    vectorstore.add_texts(texts=chunks, metadatas=metadatas)


def process_question(pdf_filename, question, llm_provider="openai"):
    file_path = os.path.join(DATA_DIR, pdf_filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("🔍 Extracting text from PDF...")
    text = extract_text_from_pdf(file_path)
    logger.info(f"Extracted {len(text)} characters from {pdf_filename}")
    logger.info("Content preview:")
    logger.info(text[:500] + "...")  # Preview first 500 characters

    logger.info("✂️ Splitting text into chunks...")
    chunks = chunk_text(text)
    logger.info(f"Created {len(chunks)} chunks from the document.")
    logger.info(f"First chunk preview start: {chunks[0][:100]}...")  # Preview first 100 characters of the first chunk
    logger.info(f"First chunk preview end: {chunks[0][-100:]}...")  # Preview last 100 characters of the first chunk
    logger.info(f"Last chunk preview start: {chunks[-1][:100]}...")  # Preview first 100 characters of the last chunk
    logger.info(f"Last chunk preview end: {chunks[-1][-100:]}...")  # Preview last 100 characters of the last chunk

    logger.info("📚 Initializing vector store...")
    vectorstore = get_vectorstore(logger=logger)
    logger.info(f"Vector store initialized at {VECTOR_DIR} with {vectorstore} existing documents.")
    

    logger.info("📥 Indexing document chunks in vector store...")
    index_document(vectorstore, chunks, metadata=pdf_filename)
    logger.info(f"Vector store metadata: {vectorstore.get()['metadatas']}")
    
    logger.info("📡 Retrieving relevant documents...")
    context_docs = get_relevant_docs(vectorstore, question)
    logger.info(f"Retrieved {len(context_docs)} relevant documents for the question.")
    logger.info("Context documents preview:")
    for i, doc in enumerate(context_docs):
        logger.info(f"Document {i+1} preview: {doc.page_content[:50]}...")
    

    logger.info("📝 Building prompt...")
    prompt = generate_prompt(context_docs, question)
    logger.info("Prompt ready for LLM query:")
    logger.info(prompt)
    

    # logger.info(f"🤖 Querying LLM provider: {llm_provider}...")
    # answer = ask_llm(prompt, provider=llm_provider)

    return 


if __name__ == "__main__":
    print("=== AskMyDocs ===")
    pdf = input("Enter PDF file name (in data/uploaded_docs): ").strip()
    question = input("What would you like to know? ").strip()


    if not pdf.endswith('.pdf'):
        pdf += '.pdf'

    try:
        answer = process_question(pdf, question)
        print("\n🧠 Answer:")
        print(answer)
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        print(f"Error: {e}")
# app/main.py

import os
import logging


from app.config import DATA_DIR
from app.embedder import embed_file, get_vectorstore
from app.retriever import get_relevant_docs
from app.llm_chain import generate_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def process_question(pdf_filenames: list[str], question :str):
    
    if pdf_filenames:    
        vectorstore = get_vectorstore(logger=logger)
        embed_file(pdf_filenames, vectorstore=vectorstore)
        context_docs = get_relevant_docs(vectorstore, question)
        
    

    prompt = generate_prompt(context_docs, question)
    logger.info("Prompt ready for LLM query:")
    logger.info(prompt) 
    

    # logger.info(f"🤖 Querying LLM provider: {llm_provider}...")
    # answer = ask_llm(prompt, provider=llm_provider)

    return 


if __name__ == "__main__":
    print("=== AskMyDocs ===")
    question = input("What would you like to know? ").strip()

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    logger.info(f"PDF files found: {pdf_files}")

    try:
        answer = process_question(pdf_files, question)
        print("\n🧠 Answer:")
        print(answer)
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        print(f"Error: {e}")
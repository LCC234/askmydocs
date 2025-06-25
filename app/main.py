from app.document_loader import extract_text_from_pdf
from app.embedder import get_vectorstore
from app.retriever import get_relevant_docs
from app.llm_chain import generate_prompt, ask_llm

def process_question(pdf_filename, question):
    # Step 1: Extract text
    text = extract_text_from_pdf(f"data/uploaded_docs/{pdf_filename}")
    
    # Step 2: Embed & store
    vectorstore = get_vectorstore()
    vectorstore.add_texts([text])  # You might want to chunk before adding
    
    # Step 3: Retrieve context
    docs = get_relevant_docs(vectorstore, question)
    
    # Step 4: Build prompt
    prompt = generate_prompt(docs, question)
    
    # Step 5: Ask the LLM
    answer = ask_llm(prompt)
    return answer
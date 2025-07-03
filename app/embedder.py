import logging
import os
from langchain_chroma import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import BASE_DIR, DATA_DIR, EMBEDDING_MODEL, VECTOR_STORE_DIR
from langchain.document_loaders import PyPDFLoader


def embed_file(pdf_filename: list[str], vectorstore : Chroma):
    logger = logging.getLogger(__name__)
    
    existing_docs = vectorstore.get()['metadatas']

    for filename in pdf_filename:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if existing_docs:
            sources_in_vectorstore = [doc.get("source") for doc in existing_docs]
            if filename in sources_in_vectorstore:
                logger.info(f"Document '{filename}' already indexed. Skipping re-indexing.")
                continue
            
        logger.info(f"Processing file: {file_path}")
        pdf_loader = PyPDFLoader(file_path)
        pdf_docs = pdf_loader.load()
        
        if not pdf_docs:
            logger.warning(f"No content found in the PDF file: {file_path}")
            continue
        
        for doc in pdf_docs:
            doc.metadata['source'] = filename
        
        vectorstore.add_documents(pdf_docs)
        logger.info(f"Document '{filename}' indexed successfully.")


def get_vectorstore(logger=None):
    persist_dir =  os.path.join(BASE_DIR, VECTOR_STORE_DIR)
    if logger:
        logger.info(f"Using embedding model: {EMBEDDING_MODEL} with device {'cuda' if os.environ.get('USE_CUDA', 'false').lower() == 'true' else 'cpu'}")
        
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda" if os.environ.get("USE_CUDA", "false").lower() == "true" else "cpu"}
    ) 
    
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model, collection_name="documents")
    if logger:
        logger.info(f"Vector store initialized with {vectorstore} documents.")
    return vectorstore

    
def get_relevant_docs(vectorstore, question, k=5):
    """
    Retrieve relevant documents from the vector store based on the question.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Retrieving top {k} relevant documents for the question: {question}")
    
    # Use the vectorstore's similarity search to find relevant documents
    results = vectorstore.similarity_search(question, k=k)
    
    if not results:
        logger.warning("No relevant documents found.")
        return []
    
    logger.info(f"Found {len(results)} relevant documents.")
    return results
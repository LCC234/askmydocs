import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import BASE_DIR, EMBEDDING_MODEL, VECTOR_STORE_DIR


def get_vectorstore(logger=None):
    persist_dir =  os.path.join(BASE_DIR, VECTOR_STORE_DIR)
    if logger:
        logger.info(f"Initializing vector store at {persist_dir}...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model, collection_name="documents")
    if logger:
        logger.info(f"Vector store initialized with {vectorstore} documents.")
    return vectorstore

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import BASE_DIR, EMBEDDING_MODEL, VECTOR_STORE_DIR

# embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
# vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

def get_vectorstore(logger=None):
    persist_dir =  os.path.join(BASE_DIR, VECTOR_STORE_DIR)
    if logger:
        logger.info(f"Initializing vector store at {persist_dir}...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # embeddings = OpenAIEmbeddings(
    #     model=EMBEDDING_MODEL,
    #     api_key="NA",
    # )

    
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model, collection_name="documents")
    if logger:
        logger.info(f"Vector store initialized with {vectorstore} documents.")
    return vectorstore

# def index_document(vectorstore, chunks, metadata=None):
#     # Optionally add metadata per chunk to identify source
#     texts_with_metadata = [{"text": c, "metadata": metadata} for c in chunks]
#     # You may want to check if texts already indexed before adding
#     vectorstore.add_texts([t['text'] for t in texts_with_metadata], metadatas=[t['metadata'] for t in texts_with_metadata])
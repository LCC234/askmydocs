from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

from app.config import EMBEDDING_MODEL, VECTOR_STORE_DIR

embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

def get_vectorstore(persist_dir=VECTOR_STORE_DIR):
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    return vectorstore

def index_document(vectorstore, chunks, metadata=None):
    # Optionally add metadata per chunk to identify source
    texts_with_metadata = [{"text": c, "metadata": metadata} for c in chunks]
    # You may want to check if texts already indexed before adding
    vectorstore.add_texts([t['text'] for t in texts_with_metadata], metadatas=[t['metadata'] for t in texts_with_metadata])
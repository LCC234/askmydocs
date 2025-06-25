from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="data/vector_store", embedding_function=embedding_model)
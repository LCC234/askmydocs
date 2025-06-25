import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_DIR = "data/vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
from langchain_community.vectorstores import Chroma


def get_relevant_docs(vectorstore : Chroma, query, k=4):
    return vectorstore.similarity_search(query, k=k)
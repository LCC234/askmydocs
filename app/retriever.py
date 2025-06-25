def get_relevant_docs(vectorstore, query, k=4):
    return vectorstore.similarity_search(query, k=k)
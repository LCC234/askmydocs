from langchain.llms import OpenAI 


def build_prompt(question, context_docs):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    return f"""You are an assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""


def ask_llm(prompt, temperature=0):
    llm = OpenAI(temperature=temperature)
    return llm(prompt)
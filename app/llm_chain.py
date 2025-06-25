from langchain.llms import OpenAI
from langchain_community.llms import Ollama

from langchain.prompts import PromptTemplate

template = """
You are a helpful assistant that answers questions based on the provided document context.
Use the context to answer the question as accurately as possible.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

def generate_prompt(context_docs, question):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    return prompt.format(context=context, question=question)


def build_prompt(question, context_docs):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    return f"""You are an assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""


def get_llm(provider="openai", temperature=0):
    if provider == "openai":
        return OpenAI(temperature=temperature)
    elif provider == "ollama":
        return Ollama(model="mistral", temperature=temperature)
    else:
        raise ValueError(f"Unknown LLM provider {provider}")

def ask_llm(prompt, provider="openai"):
    llm = get_llm(provider)
    return llm(prompt)
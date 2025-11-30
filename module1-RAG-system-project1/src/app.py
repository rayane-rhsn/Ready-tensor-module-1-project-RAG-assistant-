import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from vectordb import VectorDB
from langchain_openai import ChatOpenAI 
from langchain_groq import ChatGroq 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_community.document_loaders import TextLoader 
from langchain_core.document import Document

load_dotenv("file.env")

# Documents directory (can be overridden with env var)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to the data folder in the repo
documents_dir = os.path.join(base_dir, 'data', 'example_docs')


def load_documents(documents_dir: str) -> List[str]:
    """Load all .txt files under `documents_dir` and return their text contents."""
    results: List[str] = []
    if not os.path.exists(documents_dir):
        print("Documents directory not found. Skipping loading.")
        return results

    for filename in os.listdir(documents_dir):
        if not filename.lower().endswith('.txt'):
            continue
        path = os.path.join(documents_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                results.append(f.read())
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"Total documents loaded: {len(results)}")
    return results


class RAGAssistant:
    """A simple RAG-based AI assistant using a vector DB and multiple LLM providers."""

    def __init__(self):
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError("No valid API key found. Set OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in file.env")

        self.vector_db = VectorDB()

        self.prompt_template = ChatPromptTemplate.from_template(
            """
            You are a RAG AI assistant.
            Use ONLY the following retrieved context to answer the question.

            Context:{context}
            Question:{question}
            """
        )

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _initialize_llm(self):
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI
            except Exception as e:
                raise ImportError("Provider module 'langchain_openai' not available: " + str(e))
            return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.0)

        if os.getenv("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq
            except Exception as e:
                raise ImportError("Provider module 'langchain_groq' not available: " + str(e))
            return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), temperature=0.0)

        if os.getenv("GOOGLE_API_KEY"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except Exception as e:
                raise ImportError("Provider module 'langchain_google_genai' not available: " + str(e))
            return ChatGoogleGenerativeAI(google_api_key=os.getenv("GOOGLE_API_KEY"), model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"), temperature=0.0)

        return None

    def add_documents(self, documents: List[str]) -> None:
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        res = self.vector_db.search(input, n_results=n_results)
        context_str = ""
        if isinstance(res, dict):
            docs = res.get('documents') or []
            if docs:
                first = docs[0]
                if isinstance(first, (list, tuple)):
                    context_str = "\n\n".join(first)
                elif isinstance(first, str):
                    context_str = first
                else:
                    context_str = "\n\n".join([str(x) for x in first])

        answer = self.chain.invoke({"context": context_str, "question": input})
        return answer


def main():
    print("Initializing RAG Assistant...")
    assistant = RAGAssistant()

    print("\nLoading documents...")
    docs = load_documents(DOCUMENTS_DIR)
    assistant.add_documents(docs)

    while True:
        q = input("\nEnter a question or 'quit': ")
        if q.lower().strip() == 'quit':
            break
        print("\n[AI]:", assistant.invoke(q))


if __name__ == '__main__':
    main()

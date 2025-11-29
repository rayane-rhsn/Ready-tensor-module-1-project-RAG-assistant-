# Ready-tensor-module-1-project-RAG-assistant-
RAG-based AI assistant with Python and LangChain
# RAG Assistant

A **RAG-based AI assistant** using ChromaDB for vector storage and supporting multiple LLM providers: OpenAI, Groq, and Google Gemini.  
This assistant retrieves context from documents and generates answers with step-by-step reasoning.

---

## Features

- **Multi-LLM support:** OpenAI GPT, Groq LLM, Google Gemini  
- **Vector database integration:** ChromaDB + HuggingFace embeddings  
- **Custom document ingestion:** Automatically chunks and embeds `.txt` files  
- **Interactive CLI:** Ask questions in real time  
- **Step-by-step reasoning:** Generates answers with structured reasoning  

---

## Project Structure

rag-ai-assistant/
│
├── README.md # This file
├── requirements.txt # Python dependencies
├── .gitignore # Files/folders to ignore in Git
│
├── src/
│ ├── app.py # Main RAG assistant code
│ ├── vectordb.py # Vector database wrapper
│
├── data/
│ └── example_docs/ # Sample documents
│
└── examples/
└── demo_run.ipynb # Notebook demonstrating usage


---

## Installation & Setup

These instructions are for **running the project locally** on your computer.

1. **Clone the repository** (download it locally):

```bash
git clone https://github.com/rayane-rhsn/Ready-tensor-module-1-project-RAG-assistant.git
cd Ready-tensor-module-1-project-RAG-assistant



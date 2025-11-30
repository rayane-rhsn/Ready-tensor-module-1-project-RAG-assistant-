-Ready-tensor-module-1-project-RAG-assistant-

RAG-based AI assistant with Python and LangChain

Overview

This project implements a RAG (Retrieval-Augmented Generation) AI assistant that retrieves information from documents and generates answers with structured reasoning. The assistant supports multiple LLM providers and uses a vector database for embeddings.

Key Features

Multi-LLM support: OpenAI GPT, Groq LLM, Google Gemini

Vector database: ChromaDB + HuggingFace embeddings

Document ingestion: Automatically chunks and embeds .txt files

Interactive CLI: Ask questions in real time

Step-by-step reasoning: Structured explanations for answers

Project Structure
rag-ai-assistant/
│
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── .gitignore               # Files/folders to ignore in Git
├── sensitive.env            # Your API keys (ignored in Git)
│
├── src/
│   ├── app.py               # Main RAG assistant code
│   ├── vectordb.py          # Vector database wrapper
│
├── data/
│   └── example_docs/        # Sample documents for ingestion
│
└── examples/
    └── demo_run.ipynb       # Notebook demonstrating usage

Installation & Setup

Follow these steps to run the project locally:

1. Clone the repository
git clone https://github.com/rayane-rhsn/Ready-tensor-module-1-project-RAG-assistant-.git
cd Ready-tensor-module-1-project-RAG-assistant-

2. Create a Python virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Configure API keys

Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key


These keys are required for the LLMs to function. The .env file is ignored in Git, so your keys remain private.

5. Run the assistant
python -m src.app


The assistant will automatically load documents from data/example_docs/, embed them into ChromaDB, and allow interactive queries.

6. Optional: Run the demo notebook

Open the notebook to see an interactive demonstration:

jupyter notebook examples/demo_run.ipynb

Notes

Ensure the .env file is present; otherwise, the assistant will fail to connect to the APIs.

You can add your own documents in the data/ folder to extend the knowledge base.

This setup works on any machine, as long as Python 3.11+ is installed and the .env file contains valid API keys.

Rayane’s RAG AI Assistant – Ready for Ready Tensor publication. ✅

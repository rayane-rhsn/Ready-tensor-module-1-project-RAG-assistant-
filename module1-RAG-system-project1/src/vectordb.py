import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import chromadb

class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    # Load the environment variables from file.env
    def __init__(self, collection_name: str = None, embedding_model: str = None):
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        Split text into smaller chunks for embedding.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return chunks

    @staticmethod
    def load_documents(DOCUMENTS_DIR):
        """
        Load all .txt files under DOCUMENTS_DIR and return their text contents.

        This implementation avoids depending on langchain's TextLoader so the
        project can still run if langchain document loaders are not available.
        """
        results = []

        if not os.path.exists(DOCUMENTS_DIR):
            print("Documents directory not found. Skipping loading.")
            return results

        for file in os.listdir(DOCUMENTS_DIR):
            if file.endswith(".txt"):
                file_path = os.path.join(DOCUMENTS_DIR, file)
                try:
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    results.extend(docs)
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")

        print(f"Total documents loaded: {len(results)}")
        return results

    def add_documents(self, documents) -> None:
        """
        Split, embed, and store documents in ChromaDB.
        """
        print(f"Processing {len(documents)} documents...")

        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk_text(doc.page_content)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc{doc_idx}_chunk{chunk_idx}"
                embedding = self.embedding_model.encode(chunk)
                self.collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding.tolist()],
                    documents=[chunk],
                    metadatas=[doc.metadata or {}]
                )
        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar chunks to a query.
        """
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results

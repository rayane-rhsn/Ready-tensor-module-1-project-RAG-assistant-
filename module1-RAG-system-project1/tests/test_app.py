import unittest
from src import app, vectordb

class TestRAGAssistant(unittest.TestCase):

    def test_app_initialization(self):
        """Test that the app can initialize without errors."""
        try:
            assistant = app.RAGAssistant()
        except Exception as e:
            self.fail(f"Initialization failed with exception: {e}")

    def test_vectordb_initialization(self):
        """Test that the vector database can initialize."""
        try:
            db = vectordb.VectorDB()
        except Exception as e:
            self.fail(f"VectorDB initialization failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()

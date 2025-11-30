import unittest
from src.app import RAGAssistant  # Adjust class/function names if different
from src.vectordb import VectorDB

class TestRAGAssistant(unittest.TestCase):

    def setUp(self):
        # Initialize your objects here
        self.db = VectorDB()
        self.rag = RAGAssistant(self.db)

    def test_basic_functionality(self):
        # Example test: check if RAGAssistant returns expected type
        query = "Hello"
        result = self.rag.query(query)  # Replace with your actual method
        self.assertIsInstance(result, str)

    def test_vectordb_empty(self):
        # Check that an empty VectorDB returns expected output
        result = self.db.search("Nothing")
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()

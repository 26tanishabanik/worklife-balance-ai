import unittest
from unittest.mock import MagicMock
from src.vectorstore import RAGVectorStore


class TestRAGVectorStore(unittest.TestCase):
    def setUp(self):
        self.mock_bedrock_client = MagicMock()
        self.vector_store = RAGVectorStore(self.mock_bedrock_client, "ap-south-1")

    def test_build_vector_store(self):
        fake_posts = {"id1": "This is a test document."}
        retriever = self.vector_store.build_vector_store(fake_posts)
        self.assertIsNotNone(retriever)


if __name__ == "__main__":
    unittest.main()

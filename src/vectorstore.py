import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.embeddings import BedrockEmbeddings
from chromadb import EmbeddingFunction, Embeddings

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyEmbeddingFunction(EmbeddingFunction):
    """Custom Embedding Function for ChromaDB using AWS Bedrock."""

    def __init__(self, client, region_name: str, model_id: str):
        self.embedder = BedrockEmbeddings(
            client=client, region_name=region_name, model_id=model_id
        )

    def embed_query(self, query: str) -> Embeddings:
        return self.embedder.embed_query(query)

    def embed_documents(self, documents: list[str]) -> Embeddings:
        return self.embedder.embed_documents(documents)


class RAGVectorStore:
    """Handles vector storage and retrieval for RAG pipeline."""

    def __init__(self, bedrock_client, aws_region, embedding_model_id):
        self.bedrock_client = bedrock_client
        self.embedding_function = MyEmbeddingFunction(
            client=bedrock_client, region_name=aws_region, model_id=embedding_model_id
        )

    def build_vector_store(self, posts):
        """Builds vector store from Reddit posts."""
        docs_list = [
            Document(page_content=content, metadata={"id": post_id})
            for post_id, content in posts.items()
        ]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0.2
        )
        doc_splits = text_splitter.split_documents(docs_list)

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=self.embedding_function,
            collection_name="rag-chroma",
        )

        retriever = vectorstore.as_retriever()
        logger.info("Vector store built successfully.")
        return retriever

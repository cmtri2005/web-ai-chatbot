import logging
import threading
from typing import List, Dict, Any
from ..llm import BedrockLLM
from ..schemas import ChatResponse, Reference
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.llm_service = None
        self.vector_store = None

    def _get_llm_service(self):
        if self.llm_service is None:
            self.llm_service = BedrockLLM()
        return self.llm_service

    def _get_vector_store(self):
        if self.vector_store is None:
            self.vector_store = VectorStore()
        return self.vector_store

    def process_query(self, query: str, conversation_history=None) -> ChatResponse:
        """Process a query through the RAG pipeline and return standardized ChatResponse"""
        try:
            logger.info(f"Processing query: {query[:100]}...")

            # Search in vector store using existing Qdrant payloads
            vector_store = self._get_vector_store()
            vector_results = vector_store.search_by_text(query, top_k=5)

            formatted_docs = []

            # Process all vector results for LLM context
            for doc in vector_results:
                score = float(doc.get("score", 0.0) or 0.0)
                formatted_docs.append(
                    {
                        "source": doc.get("source", ""),
                        "title": doc.get("title", ""),
                        "url": doc.get("url", ""),
                        "content": doc.get("content", ""),
                        "score": score,
                    }
                )
            llm_service = self._get_llm_service()
            llm_response = llm_service.generate_with_context(
                user_query=query,
                documents=formatted_docs,
                conversation_history=conversation_history,
            )

            return ChatResponse(
                answer=llm_response.answer,
                references=llm_response.references,
            )

        except Exception as e:
            logger.error(f"RAG pipeline failed: {str(e)}")
            return ChatResponse(
                answer=f"Xin lỗi, có lỗi xảy ra: {str(e)}",
                references=[],
            )

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            vector_store = self._get_vector_store()
            return vector_store.search_by_text(query, top_k)

        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            return []

    def load_financial_data(self, parquet_path: str) -> bool:
        """Load financial news data from Parquet file into the vector store"""
        try:
            vector_store = self._get_vector_store()
            return vector_store.load_from_parquet(parquet_path)
        except Exception as e:
            logger.error(f"Failed to load financial data: {str(e)}")
            return False

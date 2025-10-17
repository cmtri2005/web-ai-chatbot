import sys
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.services.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        logger.info("Starting data ingestion...")

        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        parquet_path = "../data/financial_news_embedded.parquet"
        success = rag_pipeline.load_financial_data(parquet_path)

        if success:
            logger.info("Data ingestion completed successfully!")

            # Get stats
            stats = rag_pipeline.vector_store.get_collection_stats()
            logger.info(f"Vector store stats: {stats}")

            # Test search
            test_results = rag_pipeline.search_documents("VN-Index", top_k=3)
            logger.info(f"Test search returned {len(test_results)} results")

        else:
            logger.error("Data ingestion failed!")

    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

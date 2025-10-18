import logging
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import numpy as np
from ..config import config
from ..llm import BedrockEmbedding

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = "financial_news",
        vector_size: int = None,
        hybrid_weight: float = 0.7,
    ):
        host = host or config.QDRANT_HOST
        port = port or config.QDRANT_PORT
        self.host = host
        self.port = port
        self.client = None
        self.collection_name = collection_name
        self.vector_size = vector_size or config.EMBEDDING_DIM
        self.hybrid_weight = hybrid_weight
        self._bedrock_embedding = None

    def _get_client(self):
        """Lazy initialization of Qdrant client"""
        if self.client is None:
            self.client = QdrantClient(host=self.host, port=self.port)
            self._ensure_collection_exists()
        return self.client

    def _ensure_collection_exists(self):
        """Create collection if it does not exist or has wrong vector size"""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if collection_exists:
                try:
                    collection_info = self.client.get_collection(self.collection_name)
                    existing_vector_size = collection_info.config.params.vectors.size
                    if existing_vector_size != self.vector_size:
                        logger.info(
                            f"Deleting existing collection with wrong vector size ({existing_vector_size})"
                        )
                        self.client.delete_collection(self.collection_name)
                        collection_exists = False
                except Exception as e:
                    logger.warning(f"Could not check existing collection: {str(e)}")
                    collection_exists = False

            if not collection_exists:
                logger.info(
                    f"Creating new Qdrant collection: {self.collection_name} with vector size: {self.vector_size}"
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {str(e)}")

    def _get_bedrock_embedding(self) -> BedrockEmbedding:
        if self._bedrock_embedding is None:
            self._bedrock_embedding = BedrockEmbedding()
        return self._bedrock_embedding

    def load_from_parquet(self, parquet_path: str) -> bool:
        """Load embeddings and documents from a Parquet file"""
        try:
            client = self._get_client()
            df = pd.read_parquet(parquet_path)
            if "embedding" not in df.columns:
                raise ValueError("Missing 'embedding' column in parquet file.")

            df["embedding"] = df["embedding"].apply(
                lambda x: np.array(x, dtype=np.float32)
            )
            payloads = df.to_dict(orient="records")

            # Upload to Qdrant using upsert
            points = []
            for i, payload in enumerate(payloads):
                points.append(
                    models.PointStruct(
                        id=i,
                        vector=payload["embedding"].tolist(),
                        payload={
                            "source": payload.get("source", ""),
                            "title": payload.get("title", ""),
                            "url": payload.get("url", ""),
                            "content": payload.get("content", ""),
                        },
                    )
                )

            # Upload in batches
            batch_size = 256
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                client.upsert(collection_name=self.collection_name, points=batch)
                logger.info(
                    f"Uploaded batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1}"
                )

            logger.info(
                f"Uploaded {len(df)} documents to Qdrant collection '{self.collection_name}'"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load parquet file into Qdrant: {str(e)}")
            return False

    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            client = self._get_client()
            embedding_client = self._get_bedrock_embedding()
            query_vector = np.array(
                embedding_client.get_query_embedding(query), dtype=np.float32
            )

            results = client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                with_payload=True,
            )

            hits = []
            for point in results:
                hit = point.payload
                hit["score"] = point.score
                hits.append(hit)

            return hits

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            client = self._get_client()
            collection_info = client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "vector_size": collection_info.config.params.vectors.size,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "points_count": 0,
                "status": "error",
                "error": str(e),
            }

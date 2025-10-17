import os
import json
import threading
from typing import Any, Dict
from dotenv import dotenv_values


class ConfigSingleton:
    __instance = None
    __lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            with cls.__lock:
                if cls.__instance is None:
                    cls.__instance = super().__new__(cls)
                    cls.__instance.__initialize()
        return cls.__instance

    def __initialize(self) -> None:
        env = {**dotenv_values(".env"), **os.environ}


        self.AWS_ACCESS_KEY_ID = env.get("AWS_ACCESS_KEY_ID", "")
        self.AWS_SECRET_ACCESS_KEY = env.get("AWS_SECRET_ACCESS_KEY", "")
        self.AWS_REGION = env.get("AWS_REGION", "us-east-1")
        self.BEDROCK_MODEL_REGION = env.get("BEDROCK_MODEL_REGION", "us-east-1")
        self.BEDROCK_EMBEDDING_MODEL = env.get(
            "BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"
        )
        self.BEDROCK_LLM_MODEL = env.get(
            "BEDROCK_LLM_MODEL", "meta.llama3-70b-instruct-v1:0"
        )
        # Embedding dims and local fallback model
        self.EMBEDDING_DIM = self._to_int(env.get("EMBEDDING_DIM", "1024"), 1024)
        self.EMBEDDING_MODEL = env.get(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.MODEL_KWARGS = self._parse_json(env.get("MODEL_KWARGS", "{}"))
        # Vector Store
        self.QDRANT_URL = env.get("QDRANT_URL", "http://localhost:6333")
        self.QDRANT_HOST = env.get("QDRANT_HOST", "localhost")
        self.QDRANT_PORT = self._to_int(env.get("QDRANT_PORT", "6333"), 6333)
        self.MAX_CONTEXT_LENGTH = self._to_int(env.get("MAX_CONTEXT_LENGTH", "4000"), 4000)
        # Optional thresholds for showing references (0-1); set high to suppress weak refs
        self.REFERENCE_SCORE_THRESHOLD = self._to_float(
            env.get("REFERENCE_SCORE_THRESHOLD", "0.7"), 0.7
        )

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _to_int(raw: Any, default: int) -> int:
        try:
            return int(raw)
        except Exception:
            return default

    @staticmethod
    def _to_float(raw: Any, default: float) -> float:
        try:
            return float(raw)
        except Exception:
            return default


config = ConfigSingleton()

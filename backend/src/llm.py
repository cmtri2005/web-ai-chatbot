import boto3
import json
import logging
import re
import threading
from typing import List, Optional, Dict, Any

from langchain_aws import ChatBedrock
from .config import config
from .schemas import ChatResponse, Reference
from .models import ChatMessage, RoleEnum

# logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BaseBedrock:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        try:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=config.AWS_REGION,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            )
            logger.info(f"Initialized AWS Bedrock client in region {config.AWS_REGION}")
        except Exception as e:
            logger.exception("Failed to initialize Bedrock client")
            raise e

class BedrockEmbedding:
    """Generate embeddings using AWS Bedrock models."""

    def __init__(self):
        self.client = BaseBedrock().get_client()
        self.model_id = config.BEDROCK_EMBEDDING_MODEL

    def get_text_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({"inputText": text}),
                contentType="application/json",
            )
            response_body = json.loads(response["body"].read())
            return response_body.get("embedding", [])
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

    def get_query_embedding(self, query: str) -> List[float]:
        return self.get_text_embedding(query)


class BedrockLLM:
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
        self.bedrock_client = BaseBedrock().get_client()
        self.model_id = config.BEDROCK_LLM_MODEL
        self.chat = ChatBedrock(model_id=self.model_id, region_name=config.BEDROCK_MODEL_REGION)
        logger.info(f"Initialized Bedrock LLM model: {self.model_id}")

    def generate_with_context(
        self,
        user_query: str,
        documents: List[Dict[str, Any]],
        conversation_history: Optional[List[ChatMessage]] = None,
        system_prompt: Optional[str] = None,
    ) -> ChatResponse:
        try:
            context_parts = []
            for i, doc in enumerate(documents, 1):
                url = f"\nURL: {doc.get('url', '')}" if doc.get("url") else ""
                context_parts.append(
                    f"Tài liệu {i}:\nNguồn: {doc.get('source', 'N/A')}\nTiêu đề: {doc.get('title', 'N/A')}{url}\nNội dung: {(doc.get('content', '') or '')[:500]}..."
                )
            context = "\n\n".join(context_parts)

            messages = list(conversation_history[-5:]) if conversation_history else []
            messages.append(ChatMessage(role=RoleEnum.USER, content=user_query))
            return self.generate_response(messages, context=context, system_prompt=system_prompt)

        except Exception:
            logger.exception("generate_with_context failed")
            return ChatResponse(answer="Xin lỗi, có lỗi xảy ra khi xử lý ngữ cảnh.", references=[])

    def generate_response(
        self,
        messages: List[ChatMessage],
        context: Optional[str] = "",
        system_prompt: Optional[str] = None,
    ) -> ChatResponse:
        try:
            langchain_messages = []
            if system_prompt:
                langchain_messages.append({"role": "system", "content": system_prompt})

            for msg in messages:
                langchain_messages.append({"role": msg.role.value, "content": msg.content})

            if context:
                langchain_messages.append({"role": "system", "content": f"Context: {context}"})

            response = self.chat.invoke(langchain_messages)
            cleaned = self._clean_response(str(response))
            return self._parse_json_response(cleaned)

        except Exception:
            logger.exception("LLM generation failed")
            return ChatResponse(answer="Xin lỗi, có lỗi xảy ra trong quá trình tạo phản hồi.", references=[])

    def _clean_response(self, response: str) -> str:
        response = re.sub(r"```(?:json|JSON)?", "", response)
        response = re.sub(r"<\|response\|>.*?<\|end\|>", "", response, flags=re.DOTALL)
        response = re.sub(r"^.*?(\{[\s\S]*\}).*$", r"\1", response.strip())
        return response.strip()

    def _parse_json_response(self, response: str) -> ChatResponse:
        json_blocks = re.findall(r"```json\s*(.*?)\s*```", response, flags=re.DOTALL)
        json_candidates = json_blocks or re.findall(r"\{[\s\S]*?\}", response, flags=re.DOTALL)

        for cand in reversed(json_candidates):
            try:
                data = json.loads(cand)
                refs_raw = data.get("references") or data.get("sources") or []
                references = [
                    Reference(
                        title=ref.get("title", ""),
                        url=ref.get("url", ""),
                        source=ref.get("source", ""),
                        content=ref.get("content", ""),
                    )
                    for ref in refs_raw if isinstance(ref, dict)
                ]
                return ChatResponse(answer=data.get("answer", ""), references=references)
            except json.JSONDecodeError:
                continue

        return ChatResponse(answer=response, references=[])

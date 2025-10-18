from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import logging
from .config import config
from .models import ChatRequest
from .schemas import ChatResponse
from .services.rag_pipeline import RAGPipeline
from .services.vector_store import VectorStore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial News Chatbot API",
    description="RAG-based chatbot for Vietnamese financial news",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_service():
    return RAGPipeline()


def get_vector_store():
    return VectorStore()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Financial News Chatbot API is running"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        return {
            "status": "healthy",
            "vector_store": stats,
            "models": {
                "llm": config.BEDROCK_LLM_MODEL,
                "embedding": config.BEDROCK_EMBEDDING_MODEL,
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint"""
    try:
        logger.info(f"Received chat request: {request.message[:100]}...")

        # Process query through RAG pipeline
        rag_service = get_rag_service()
        response = rag_service.process_query(request.message)
        print(f"Response: {response}")
        print("*************************************************")
        logger.info(f"Generated response with {len(response.references)} references")
        return response

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/search")
async def search_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search documents without generating answer"""
    try:
        rag_service = get_rag_service()
        documents = rag_service.search_documents(query, top_k)
        return {"query": query, "documents": documents, "total_found": len(documents)}
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="localhost", port=8000, reload=True)

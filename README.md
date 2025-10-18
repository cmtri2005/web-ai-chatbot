# Web AI Financial Chatbot
```
web-ai-chatbot/
├── backend/                    # Backend API (FastAPI)
│   ├── src/
│   │   ├── app.py             # FastAPI application
│   │   ├── config.py          # Configuration management
│   │   ├── llm.py             # AWS Bedrock LLM & Embedding
│   │   ├── models.py          # Pydantic models
│   │   ├── schemas.py         # Response schemas
│   │   └── services/
│   │       ├── rag_pipeline.py # RAG pipeline logic
│   │       └── vector_store.py # Qdrant vector store
│   ├── ingest_data.py         # Data ingestion script
│   ├── requirements.txt       # Python dependencies
│   ├── docker-compose.yml     # Qdrant container setup
│   └── venv/                  # Virtual environment
├── chatbot-ui/                # Frontend UI (Streamlit)
│   ├── chat_interface.py      # Streamlit chat interface
│   ├── requirements.txt       # UI dependencies
│   └── venv/                  # Virtual environment
├── data_preprocessing/        # Data processing & embedding
│   ├── embedding.ipynb        # Jupyter notebook for embeddings
│   ├── financial_news.csv     # Raw financial news data
│   └── financial_news_embedded.parquet # Processed embeddings
├── .env_example              # Environment variables template
└── README.md                 # This file
```
### Clone repository

```bash
git clone <repository-url>
cd web-ai-chatbot
```

### Qdrant
```bash
cd backend
docker-compose up -d
```
```bash
curl http://localhost:6333/health
```

### Backend
```bash
cd backend

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```
### Ingest data
```bash
python ingest_data.py
```
### Backend
```bash
#
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
### Frontend
```bash
cd chatbot-ui

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

streamlit run chat_interface.py --server.port 8501
```

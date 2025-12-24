import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = BASE_DIR / "cache"
    CHROMA_DIR = CACHE_DIR / "chroma_db"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Ensure directories exist
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Embedding model (FREE - Hugging Face)
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    # Reranker model (FREE - Hugging Face)
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # LLM API (Groq - Free tier)
    LLM_PROVIDER = "groq"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.1-8b-instant"
    
    # Retrieval settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_RETRIEVAL = 10
    TOP_K_RERANK = 3
    
    # Cache settings
    CACHE_SIMILARITY_THRESHOLD = 0.90
    CACHE_COLLECTION_NAME = "query_cache"
    
    # Batch processing
    BATCH_SIZE = 10
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
    
    # Web UI settings
    WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
    WEB_PORT = int(os.getenv("WEB_PORT", 8000))
    
    # Load testing & production
    MAX_QUEUE_SIZE = 1000
    REQUEST_TIMEOUT = 30
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 60))
    ENABLE_METRICS = True
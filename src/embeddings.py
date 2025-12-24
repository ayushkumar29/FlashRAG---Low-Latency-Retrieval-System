from sentence_transformers import SentenceTransformer
from typing import List
from src.config import Config
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

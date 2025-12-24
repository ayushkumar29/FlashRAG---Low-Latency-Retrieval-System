from sentence_transformers import CrossEncoder
from typing import List, Dict
from src.config import Config
import logging

logger = logging.getLogger(__name__)

class DocumentReranker:
    def __init__(self, model_name: str = Config.RERANKER_MODEL):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = Config.TOP_K_RERANK) -> List[Dict]:
        """Rerank documents using cross-encoder"""
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, doc['text']] for doc in documents]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score and return top-k
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]

import chromadb
from chromadb.config import Settings
from typing import Optional, List, Dict
from src.config import Config
from src.embeddings import EmbeddingGenerator
import logging

logger = logging.getLogger(__name__)

class SemanticCache:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(Config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_gen = EmbeddingGenerator()
        
        self.collection = self.client.get_or_create_collection(
            name=Config.CACHE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    def check_cache(self, query: str, threshold: float = Config.CACHE_SIMILARITY_THRESHOLD) -> Optional[Dict]:
        try:
            query_embedding = self.embedding_gen.embed_query(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1
            )
            
            if results['ids'][0]:
                similarity = 1 - results['distances'][0][0]
                
                if similarity >= threshold:
                    logger.info(f"Cache hit! Similarity: {similarity:.3f}")
                    return {
                        "cached_response": results['metadatas'][0][0]['response'],
                        "similarity": similarity,
                        "cached_query": results['documents'][0][0]
                    }
        except Exception as e:
            logger.error(f"Cache check error: {e}")
        
        return None
    
    def add_to_cache(self, query: str, response: str, retrieved_docs: List[str]):
        try:
            query_embedding = self.embedding_gen.embed_query(query)
            
            self.collection.add(
                embeddings=[query_embedding],
                documents=[query],
                metadatas=[{
                    "response": response,
                    "doc_count": len(retrieved_docs)
                }],
                ids=[f"cache_{self.collection.count()}"]
            )
            logger.info("Added to cache")
        except Exception as e:
            logger.error(f"Cache add error: {e}")


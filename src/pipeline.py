import time
from typing import Dict, Iterator
from src.semantic_cache import SemanticCache
from src.retriever import DocumentRetriever
from src.reranker import DocumentReranker
from src.llm_client import LLMClient
import threading
import logging

logger = logging.getLogger(__name__)

class FlashRAGPipeline:
    def __init__(self):
        self.cache = SemanticCache()
        self.retriever = DocumentRetriever()
        self.reranker = DocumentReranker()
        self.llm = LLMClient()
        self._lock = threading.Lock()
    
    def query(self, question: str, use_cache: bool = True) -> Dict:
        """Main query pipeline"""
        start_time = time.time()
        metrics = {
            "cache_hit": False,
            "num_retrieved": 0,
            "num_reranked": 0,
            "latency_ms": 0
        }
        
        if use_cache:
            cached_result = self.cache.check_cache(question)
            if cached_result:
                metrics["cache_hit"] = True
                metrics["latency_ms"] = (time.time() - start_time) * 1000
                return {
                    "answer": cached_result["cached_response"],
                    "source": "cache",
                    "metrics": metrics
                }
        
        retrieved_docs = self.retriever.retrieve(question)
        metrics["num_retrieved"] = len(retrieved_docs)
        
        reranked_docs = self.reranker.rerank(question, retrieved_docs)
        metrics["num_reranked"] = len(reranked_docs)
        
        answer = self.llm.generate_response(question, reranked_docs)
        
        if use_cache:
            with self._lock:
                self.cache.add_to_cache(question, answer, [doc['text'] for doc in reranked_docs])
        
        metrics["latency_ms"] = (time.time() - start_time) * 1000
        
        return {
            "answer": answer,
            "source": "llm",
            "retrieved_docs": reranked_docs,
            "metrics": metrics
        }
    
    def query_stream(self, question: str, use_cache: bool = True) -> Iterator[Dict]:
        """Streaming query pipeline"""
        start_time = time.time()
        
        if use_cache:
            cached_result = self.cache.check_cache(question)
            if cached_result:
                yield {
                    "type": "cache_hit",
                    "data": cached_result["cached_response"],
                    "done": True,
                    "metrics": {
                        "cache_hit": True,
                        "latency_ms": (time.time() - start_time) * 1000
                    }
                }
                return
        
        retrieved_docs = self.retriever.retrieve(question)
        reranked_docs = self.reranker.rerank(question, retrieved_docs)
        
        yield {
            "type": "retrieval_complete",
            "data": f"Retrieved {len(reranked_docs)} relevant documents",
            "done": False
        }
        
        full_response = ""
        for chunk in self.llm.generate_response_stream(question, reranked_docs):
            full_response += chunk
            yield {
                "type": "content",
                "data": chunk,
                "done": False
            }
        
        if use_cache:
            with self._lock:
                self.cache.add_to_cache(question, full_response, [doc['text'] for doc in reranked_docs])
        
        yield {
            "type": "complete",
            "data": "",
            "done": True,
            "metrics": {
                "cache_hit": False,
                "num_retrieved": len(retrieved_docs),
                "num_reranked": len(reranked_docs),
                "latency_ms": (time.time() - start_time) * 1000
            }
        }

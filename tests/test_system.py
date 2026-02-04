import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import FlashRAGPipeline
from src.batch_processor import BatchProcessor
from src.semantic_cache import SemanticCache
from src.retriever import DocumentRetriever


@pytest.fixture
def pipeline():
    return FlashRAGPipeline()


@pytest.fixture
def cache():
    return SemanticCache()


class TestCache:
    
    def test_cache_miss(self, pipeline):
        result = pipeline.query("What is quantum computing?", use_cache=False)
        assert result['answer'] is not None
        assert result['metrics']['cache_hit'] == False
        assert result['metrics']['latency_ms'] > 0
    
    def test_cache_hit(self, pipeline):
        query = "What is machine learning?"
        
        result1 = pipeline.query(query, use_cache=True)
        assert result1['metrics']['cache_hit'] == False
        latency1 = result1['metrics']['latency_ms']
        
        result2 = pipeline.query(query, use_cache=True)
        assert result2['metrics']['cache_hit'] == True
        latency2 = result2['metrics']['latency_ms']
        
        assert latency2 < latency1 / 10
    
    def test_similar_query_cache_hit(self, pipeline):
        result1 = pipeline.query("What is machine learning?", use_cache=True)
        
        result2 = pipeline.query("What is ML?", use_cache=True)
        
        assert result2['metrics'] is not None


class TestRetrieval:
    
    def test_retrieval_returns_docs(self, pipeline):
        result = pipeline.query("What is deep learning?", use_cache=False)
        assert result['metrics']['num_retrieved'] > 0
        assert result['metrics']['num_reranked'] > 0
    
    def test_reranking_reduces_docs(self, pipeline):
        result = pipeline.query("Explain neural networks", use_cache=False)
        assert result['metrics']['num_reranked'] <= result['metrics']['num_retrieved']


class TestStreaming:
    
    def test_streaming_query(self, pipeline):
        chunks = list(pipeline.query_stream("What is machine learning?"))
        
        assert len(chunks) > 0
        
        assert chunks[-1]['done'] == True
        
        content_chunks = [c for c in chunks if c['type'] == 'content']
        assert len(content_chunks) > 0
    
    def test_streaming_cache_hit(self, pipeline):
        pipeline.query("What is AI?", use_cache=True)
        
        chunks = list(pipeline.query_stream("What is AI?", use_cache=True))
        
        cache_hit_chunks = [c for c in chunks if c['type'] == 'cache_hit']
        assert len(cache_hit_chunks) > 0


class TestBatchProcessing:
    
    def test_batch_processing(self):
        processor = BatchProcessor(max_workers=2)
        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "What is deep learning?"
        ]
        
        results = processor.process_batch(queries, use_cache=False)
        
        assert len(results) == len(queries)
        
        assert all('answer' in r or 'error' in r for r in results)
        
        assert all('query' in r for r in results)


class TestErrorHandling:
    
    def test_empty_query(self, pipeline):
        with pytest.raises(Exception):
            pipeline.query("")
    
    def test_very_long_query(self, pipeline):
        long_query = "What is machine learning? " * 100
        result = pipeline.query(long_query, use_cache=False)
        assert result is not None
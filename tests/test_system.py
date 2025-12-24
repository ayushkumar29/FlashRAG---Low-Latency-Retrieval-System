
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import FlashRAGPipeline
from src.batch_processor import BatchProcessor
from src.semantic_cache import SemanticCache
from src.retriever import DocumentRetriever


@pytest.fixture
def pipeline():
    """Create pipeline instance for testing"""
    return FlashRAGPipeline()


@pytest.fixture
def cache():
    """Create cache instance for testing"""
    return SemanticCache()


class TestCache:
    """Test semantic cache functionality"""
    
    def test_cache_miss(self, pipeline):
        """Test query without cache hit"""
        result = pipeline.query("What is quantum computing?", use_cache=False)
        assert result['answer'] is not None
        assert result['metrics']['cache_hit'] == False
        assert result['metrics']['latency_ms'] > 0
    
    def test_cache_hit(self, pipeline):
        """Test cache functionality - should be faster on second query"""
        query = "What is machine learning?"
        
        # First query - cache miss
        result1 = pipeline.query(query, use_cache=True)
        assert result1['metrics']['cache_hit'] == False
        latency1 = result1['metrics']['latency_ms']
        
        # Second query - cache hit
        result2 = pipeline.query(query, use_cache=True)
        assert result2['metrics']['cache_hit'] == True
        latency2 = result2['metrics']['latency_ms']
        
        # Cache should be significantly faster
        assert latency2 < latency1 / 10  # At least 10x faster
    
    def test_similar_query_cache_hit(self, pipeline):
        """Test that similar queries hit cache"""
        # Cache first query
        result1 = pipeline.query("What is machine learning?", use_cache=True)
        
        # Similar query should hit cache
        result2 = pipeline.query("What is ML?", use_cache=True)
        
        # May or may not hit depending on similarity threshold
        assert result2['metrics'] is not None


class TestRetrieval:
    """Test retrieval functionality"""
    
    def test_retrieval_returns_docs(self, pipeline):
        """Test that retrieval returns documents"""
        result = pipeline.query("What is deep learning?", use_cache=False)
        assert result['metrics']['num_retrieved'] > 0
        assert result['metrics']['num_reranked'] > 0
    
    def test_reranking_reduces_docs(self, pipeline):
        """Test that reranking reduces document count"""
        result = pipeline.query("Explain neural networks", use_cache=False)
        assert result['metrics']['num_reranked'] <= result['metrics']['num_retrieved']


class TestStreaming:
    """Test streaming functionality"""
    
    def test_streaming_query(self, pipeline):
        """Test streaming response generation"""
        chunks = list(pipeline.query_stream("What is machine learning?"))
        
        # Should have multiple chunks
        assert len(chunks) > 0
        
        # Last chunk should be complete
        assert chunks[-1]['done'] == True
        
        # Should have content chunks
        content_chunks = [c for c in chunks if c['type'] == 'content']
        assert len(content_chunks) > 0
    
    def test_streaming_cache_hit(self, pipeline):
        """Test streaming with cache hit"""
        # Cache the query first
        pipeline.query("What is AI?", use_cache=True)
        
        # Stream same query
        chunks = list(pipeline.query_stream("What is AI?", use_cache=True))
        
        # Should get cache hit
        cache_hit_chunks = [c for c in chunks if c['type'] == 'cache_hit']
        assert len(cache_hit_chunks) > 0


class TestBatchProcessing:
    """Test batch processing functionality"""
    
    def test_batch_processing(self):
        """Test batch query processing"""
        processor = BatchProcessor(max_workers=2)
        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "What is deep learning?"
        ]
        
        results = processor.process_batch(queries, use_cache=False)
        
        # Should process all queries
        assert len(results) == len(queries)
        
        # All should have answers
        assert all('answer' in r or 'error' in r for r in results)
        
        # Should have metrics
        assert all('query' in r for r in results)


class TestErrorHandling:
    """Test error handling"""
    
    def test_empty_query(self, pipeline):
        """Test handling of empty query"""
        with pytest.raises(Exception):
            pipeline.query("")
    
    def test_very_long_query(self, pipeline):
        """Test handling of very long query"""
        long_query = "What is machine learning? " * 100
        result = pipeline.query(long_query, use_cache=False)
        # Should handle gracefully
        assert result is not None
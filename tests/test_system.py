import pytest
from src.pipeline import FlashRAGPipeline
from src.batch_processor import BatchProcessor

@pytest.fixture
def pipeline():
    return FlashRAGPipeline()

def test_query_without_cache(pipeline):
    """Test basic query without cache"""
    result = pipeline.query("What is machine learning?", use_cache=False)
    assert result['answer'] is not None
    assert result['metrics']['cache_hit'] == False
    assert result['metrics']['latency_ms'] > 0

def test_query_with_cache(pipeline):
    """Test cache functionality"""
    query = "What is machine learning?"
    
    # First query - cache miss
    result1 = pipeline.query(query, use_cache=True)
    assert result1['metrics']['cache_hit'] == False
    latency1 = result1['metrics']['latency_ms']
    
    # Second query - cache hit
    result2 = pipeline.query(query, use_cache=True)
    assert result2['metrics']['cache_hit'] == True
    latency2 = result2['metrics']['latency_ms']
    
    # Cache should be faster
    assert latency2 < latency1

def test_streaming_query(pipeline):
    """Test streaming response"""
    chunks = list(pipeline.query_stream("What is machine learning?"))
    assert len(chunks) > 0
    assert chunks[-1]['done'] == True

def test_batch_processing():
    """Test batch processor"""
    processor = BatchProcessor(max_workers=2)
    queries = [
        "What is machine learning?",
        "Explain neural networks"
    ]
    results = processor.process_batch(queries)
    assert len(results) == len(queries)

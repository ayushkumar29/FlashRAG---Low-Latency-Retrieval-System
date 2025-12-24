"""
Locust load testing file for FlashRAG

Usage:
    locust -f benchmarks/locustfile.py --host=http://localhost:8000
    
Then open http://localhost:8089 and configure:
    - Number of users: 50-100
    - Spawn rate: 10 users/sec
    - Host: http://localhost:8000
"""

from locust import HttpUser, task, between
import random


class FlashRAGUser(HttpUser):
    """Simulates a user interacting with FlashRAG API"""
    
    # Wait 1-3 seconds between requests
    wait_time = between(1, 3)
    
    # Test queries
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
        "How does supervised learning work?",
        "What are neural networks?",
        "Explain reinforcement learning",
        "What is NLP?",
        "How does deep learning differ from machine learning?",
        "What are the applications of AI?",
        "Explain convolutional neural networks"
    ]
    
    @task(5)
    def query_with_cache(self):
        """Most common: Regular query with cache enabled"""
        self.client.post("/api/query", json={
            "query": random.choice(self.queries),
            "use_cache": True,
            "stream": False
        }, name="/api/query [cached]")
    
    @task(2)
    def query_without_cache(self):
        """Query without cache to test full pipeline"""
        self.client.post("/api/query", json={
            "query": random.choice(self.queries),
            "use_cache": False,
            "stream": False
        }, name="/api/query [no cache]")
    
    @task(2)
    def streaming_query(self):
        """Streaming query"""
        self.client.post("/api/query", json={
            "query": random.choice(self.queries),
            "use_cache": True,
            "stream": True
        }, name="/api/query [streaming]")
    
    @task(1)
    def batch_query(self):
        """Batch query"""
        batch_queries = random.sample(self.queries, 3)
        self.client.post("/api/batch", json={
            "queries": batch_queries,
            "use_cache": True
        }, name="/api/batch")
    
    @task(1)
    def check_metrics(self):
        """Check system metrics"""
        self.client.get("/api/metrics", name="/api/metrics")
    
    @task(1)
    def health_check(self):
        """Health check"""
        self.client.get("/api/health", name="/api/health")
    
    def on_start(self):
        """Called when a user starts"""
        # Could add login or initialization here
        pass


class StressTestUser(HttpUser):
    """Aggressive user for stress testing"""
    
    wait_time = between(0.1, 0.5)  # Very short wait time
    
    queries = FlashRAGUser.queries
    
    @task
    def rapid_fire_queries(self):
        """Rapid successive queries"""
        self.client.post("/api/query", json={
            "query": random.choice(self.queries),
            "use_cache": True,
            "stream": False
        })

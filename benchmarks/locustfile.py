from locust import HttpUser, task, between
import random

class FlashRAGUser(HttpUser):
    wait_time = between(1, 3)
    
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
        "How does NLP work?",
        "What is reinforcement learning?"
    ]
    
    @task(3)
    def query_with_cache(self):
        """Regular query with cache enabled"""
        self.client.post("/api/query", json={
            "query": random.choice(self.queries),
            "use_cache": True,
            "stream": False
        })
    
    @task(1)
    def query_without_cache(self):
        """Query without cache"""
        self.client.post("/api/query", json={
            "query": random.choice(self.queries),
            "use_cache": False,
            "stream": False
        })
    
    @task(1)
    def streaming_query(self):
        """Streaming query"""
        self.client.post("/api/query", json={
            "query": random.choice(self.queries),
            "use_cache": True,
            "stream": True
        })
    
    @task(1)
    def check_health(self):
        """Health check"""
        self.client.get("/api/health")

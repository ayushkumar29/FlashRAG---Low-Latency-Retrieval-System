from locust import HttpUser, task, between
import random


class FlashRAGUser(HttpUser):
    
    wait_time = between(1, 3)
    
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
        self.client.post("/api/query", json={
            "query": random.choice(self.queries),
            "use_cache": True,
            "stream": False
        }, name="/api/query [cached]")
    
    @task(2)
    def query_without_cache(self):
        self.client.post("/api/query", json={
            "query": random.choice(self.queries),
            "use_cache": False,
            "stream": False
        }, name="/api/query [no cache]")
    
    @task(2)
    def streaming_query(self):
        self.client.post("/api/query", json={
            "query": random.choice(self.queries),
            "use_cache": True,
            "stream": True
        }, name="/api/query [streaming]")
    
    @task(1)
    def batch_query(self):
        batch_queries = random.sample(self.queries, 3)
        self.client.post("/api/batch", json={
            "queries": batch_queries,
            "use_cache": True
        }, name="/api/batch")
    
    @task(1)
    def check_metrics(self):
        self.client.get("/api/metrics", name="/api/metrics")
    
    @task(1)
    def health_check(self):
        self.client.get("/api/health", name="/api/health")
    
    def on_start(self):
        pass


class StressTestUser(HttpUser):
    
    wait_time = between(0.1, 0.5)
    
    queries = FlashRAGUser.queries
    
    @task
    def rapid_fire_queries(self):
        self.client.post("/api/query", json={
            "query": random.choice(self.queries),
            "use_cache": True,
            "stream": False
        })

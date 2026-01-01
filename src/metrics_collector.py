from threading import Lock
import time

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_latency": 0,
            "avg_latency": 0,
            "requests_per_second": 0,
            "start_time": time.time()
        }
        self.lock = Lock()
    
    def record_request(self, request_metrics: dict):
        with self.lock:
            self.metrics["total_requests"] += 1
            
            if request_metrics.get("cache_hit"):
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1
            
            self.metrics["total_latency"] += request_metrics.get("latency_ms", 0)
            self.metrics["avg_latency"] = (
                self.metrics["total_latency"] / self.metrics["total_requests"]
            )
            
            elapsed = time.time() - self.metrics["start_time"]
            self.metrics["requests_per_second"] = self.metrics["total_requests"] / elapsed
    
    def get_summary(self) -> dict:
        with self.lock:
            cache_hit_rate = (
                self.metrics["cache_hits"] / self.metrics["total_requests"] * 100
                if self.metrics["total_requests"] > 0 else 0
            )
            
            return {
                **self.metrics,
                "cache_hit_rate": f"{cache_hit_rate:.2f}%"
            }

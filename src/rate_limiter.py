import time
from collections import defaultdict
from threading import Lock

class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = Lock()
    
    def allow_request(self, client_id: str) -> bool:
        with self.lock:
            current_time = time.time()
            
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < self.window_seconds
            ]
            
            if len(self.requests[client_id]) >= self.max_requests:
                return False
            
            self.requests[client_id].append(current_time)
            return True

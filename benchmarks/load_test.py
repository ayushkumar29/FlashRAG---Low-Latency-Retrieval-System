import asyncio
import aiohttp
import time
from typing import List
import statistics

async def send_request(session: aiohttp.ClientSession, url: str, query: str) -> dict:
    """Send single async request"""
    start = time.time()
    try:
        async with session.post(
            url,
            json={"query": query, "use_cache": True},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            data = await response.json()
            latency = (time.time() - start) * 1000
            return {
                "status": response.status,
                "latency": latency,
                "cache_hit": data.get("metrics", {}).get("cache_hit", False)
            }
    except Exception as e:
        return {
            "status": 500,
            "latency": (time.time() - start) * 1000,
            "error": str(e)
        }

async def load_test(
    url: str,
    queries: List[str],
    concurrent_users: int = 10,
    requests_per_user: int = 10
):
    """Run load test"""
    print(f"🔥 Starting load test:")
    print(f"  - Concurrent users: {concurrent_users}")
    print(f"  - Requests per user: {requests_per_user}")
    print(f"  - Total requests: {concurrent_users * requests_per_user}\n")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.time()
        
        # Create tasks
        for user in range(concurrent_users):
            for req in range(requests_per_user):
                query = queries[req % len(queries)]
                tasks.append(send_request(session, url, query))
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Analyze results
        latencies = [r["latency"] for r in results if "latency" in r]
        cache_hits = sum(1 for r in results if r.get("cache_hit"))
        errors = sum(1 for r in results if r["status"] != 200)
        
        print(f"\n📊 Load Test Results:")
        print(f"  - Total Time: {total_time:.2f}s")
        print(f"  - Requests/sec: {len(results)/total_time:.2f}")
        print(f"  - Success Rate: {(len(results)-errors)/len(results)*100:.1f}%")
        print(f"  - Cache Hit Rate: {cache_hits/len(results)*100:.1f}%")
        print(f"\n⚡ Latency Stats:")
        print(f"  - Min: {min(latencies):.2f}ms")
        print(f"  - Max: {max(latencies):.2f}ms")
        print(f"  - Mean: {statistics.mean(latencies):.2f}ms")
        print(f"  - Median: {statistics.median(latencies):.2f}ms")
        
        if len(latencies) >= 20:
            print(f"  - P95: {statistics.quantiles(latencies, n=20)[18]:.2f}ms")
        if len(latencies) >= 100:
            print(f"  - P99: {statistics.quantiles(latencies, n=100)[98]:.2f}ms")

if __name__ == "__main__":
    # Test queries
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
        "How does NLP work?",
        "What is reinforcement learning?"
    ]
    
    # Run load test
    asyncio.run(load_test(
        url="http://localhost:8000/api/query",
        queries=queries,
        concurrent_users=50,
        requests_per_user=20
    ))

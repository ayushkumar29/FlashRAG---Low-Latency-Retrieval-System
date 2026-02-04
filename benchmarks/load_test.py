import asyncio
import aiohttp
import time
from typing import List, Dict
import statistics
import json


async def send_request(
    session: aiohttp.ClientSession, 
    url: str, 
    query: str,
    request_id: int
) -> Dict:
    start = time.time()
    try:
        async with session.post(
            url,
            json={"query": query, "use_cache": True, "stream": False},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            data = await response.json()
            latency = (time.time() - start) * 1000
            
            return {
                "request_id": request_id,
                "status": response.status,
                "latency": latency,
                "cache_hit": data.get("metrics", {}).get("cache_hit", False),
                "query": query
            }
    except asyncio.TimeoutError:
        return {
            "request_id": request_id,
            "status": 408,
            "latency": (time.time() - start) * 1000,
            "error": "Timeout",
            "query": query
        }
    except Exception as e:
        return {
            "request_id": request_id,
            "status": 500,
            "latency": (time.time() - start) * 1000,
            "error": str(e),
            "query": query
        }


async def load_test(
    url: str,
    queries: List[str],
    concurrent_users: int = 10,
    requests_per_user: int = 10
):
    total_requests = concurrent_users * requests_per_user
    
    print("=" * 60)
    print("FlashRAG Load Test")
    print("=" * 60)
    print(f"Configuration:")
    print(f"   Concurrent Users: {concurrent_users}")
    print(f"   Requests per User: {requests_per_user}")
    print(f"   Total Requests: {total_requests}")
    print(f"   Target URL: {url}")
    print(f"   Query Pool: {len(queries)} unique queries")
    print()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.time()
        
        request_id = 0
        for user in range(concurrent_users):
            for req in range(requests_per_user):
                query = queries[req % len(queries)]
                tasks.append(send_request(session, url, query, request_id))
                request_id += 1
        
        print("Starting load test...")
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("Results Analysis")
        print("=" * 60)
        
        successful = [r for r in results if r["status"] == 200]
        failed = [r for r in results if r["status"] != 200]
        cache_hits = sum(1 for r in results if r.get("cache_hit", False))
        
        print(f"\nSuccess Metrics:")
        print(f"   Total Requests: {len(results)}")
        print(f"   Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"   Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        print(f"   Cache Hits: {cache_hits} ({cache_hits/len(results)*100:.1f}%)")
        
        print(f"\nThroughput:")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Requests/sec: {len(results)/total_time:.2f}")
        print(f"   Avg Time per Request: {total_time/len(results)*1000:.2f}ms")
        
        if successful:
            latencies = [r["latency"] for r in successful]
            cache_hit_latencies = [r["latency"] for r in successful if r.get("cache_hit")]
            cache_miss_latencies = [r["latency"] for r in successful if not r.get("cache_hit")]
            
            print(f"\nLatency Statistics (All Requests):")
            print(f"   Min: {min(latencies):.2f}ms")
            print(f"   Max: {max(latencies):.2f}ms")
            print(f"   Mean: {statistics.mean(latencies):.2f}ms")
            print(f"   Median: {statistics.median(latencies):.2f}ms")
            print(f"   Std Dev: {statistics.stdev(latencies) if len(latencies) > 1 else 0:.2f}ms")
            
            if len(latencies) >= 20:
                percentiles = statistics.quantiles(latencies, n=20)
                print(f"   P50: {percentiles[9]:.2f}ms")
                print(f"   P90: {percentiles[17]:.2f}ms")
                print(f"   P95: {percentiles[18]:.2f}ms")
            
            if len(latencies) >= 100:
                percentiles_99 = statistics.quantiles(latencies, n=100)
                print(f"   P99: {percentiles_99[98]:.2f}ms")
            
            if cache_hit_latencies and cache_miss_latencies:
                print(f"\nCache Performance:")
                print(f"   Cache Hit Latency: {statistics.mean(cache_hit_latencies):.2f}ms")
                print(f"   Cache Miss Latency: {statistics.mean(cache_miss_latencies):.2f}ms")
                speedup = statistics.mean(cache_miss_latencies) / statistics.mean(cache_hit_latencies)
                print(f"   Speedup: {speedup:.1f}x faster")
        
        if failed:
            print(f"\nError Analysis:")
            error_types = {}
            for r in failed:
                error = r.get("error", f"Status {r['status']}")
                error_types[error] = error_types.get(error, 0) + 1
            
            for error, count in error_types.items():
                print(f"   {error}: {count} ({count/len(failed)*100:.1f}%)")
        
        query_stats = {}
        for r in results:
            q = r["query"]
            if q not in query_stats:
                query_stats[q] = {"count": 0, "cache_hits": 0, "latencies": []}
            query_stats[q]["count"] += 1
            if r.get("cache_hit"):
                query_stats[q]["cache_hits"] += 1
            if r["status"] == 200:
                query_stats[q]["latencies"].append(r["latency"])
        
        print(f"\nQuery Distribution:")
        for query, stats in list(query_stats.items())[:5]:
            avg_lat = statistics.mean(stats["latencies"]) if stats["latencies"] else 0
            cache_rate = stats["cache_hits"] / stats["count"] * 100
            print(f"   '{query[:40]}...'")
            print(f"     Requests: {stats['count']} | Cache: {cache_rate:.0f}% | Avg: {avg_lat:.0f}ms")
        
        print("\n" + "=" * 60)
        print("Load test complete!")
        print("=" * 60)
        
        output_file = "load_test_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                "config": {
                    "concurrent_users": concurrent_users,
                    "requests_per_user": requests_per_user,
                    "total_requests": total_requests
                },
                "summary": {
                    "total_time": total_time,
                    "requests_per_sec": len(results)/total_time,
                    "success_rate": len(successful)/len(results)*100,
                    "cache_hit_rate": cache_hits/len(results)*100
                },
                "results": results
            }, f, indent=2)
        
        print(f"Detailed results saved to: {output_file}\n")


async def main():
    queries = [
        "What is machine learning?",
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
        "How does supervised learning work?",
        "What are the applications of machine learning?",
        "Explain neural networks",
        "What is reinforcement learning?",
        "What is machine learning?",
        "How do neural networks work?"
    ]
    
    await load_test(
        url="http://localhost:8000/api/query",
        queries=queries,
        concurrent_users=50,
        requests_per_user=20
    )


if __name__ == "__main__":
    print("\nStarting FlashRAG Load Test\n")
    print("Make sure the server is running:")
    print("   python main.py serve\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nLoad test interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")

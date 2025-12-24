from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.pipeline import FlashRAGPipeline
from src.config import Config
import logging
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, max_workers: int = Config.MAX_WORKERS):
        self.pipeline = FlashRAGPipeline()
        self.max_workers = max_workers
    
    def process_batch(self, queries: List[str], use_cache: bool = True) -> List[Dict]:
        """Process multiple queries in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(self.pipeline.query, query, use_cache): query 
                for query in queries
            }
            
            # Collect results with progress bar
            with tqdm(total=len(queries), desc="Processing queries") as pbar:
                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        result = future.result()
                        result['query'] = query
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing query '{query}': {e}")
                        results.append({
                            "query": query,
                            "error": str(e),
                            "answer": None
                        })
                    pbar.update(1)
        
        return results
    
    def process_file(self, input_file: str, output_file: str, use_cache: bool = True):
        """Process queries from file and save results"""
        # Read queries
        with open(input_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(queries)} queries from {input_file}")
        
        # Process in batches
        results = self.process_batch(queries, use_cache)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        cache_hits = sum(1 for r in results if r.get('metrics', {}).get('cache_hit', False))
        avg_latency = sum(r.get('metrics', {}).get('latency_ms', 0) for r in results) / len(results)
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"  - Total Queries: {len(results)}")
        print(f"  - Cache Hits: {cache_hits} ({cache_hits/len(results)*100:.1f}%)")
        print(f"  - Average Latency: {avg_latency:.2f}ms")

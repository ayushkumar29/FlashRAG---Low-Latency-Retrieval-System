from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.pipeline import FlashRAGPipeline
from src.config import Config
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, max_workers: int = Config.MAX_WORKERS):
        self.pipeline = FlashRAGPipeline()
        self.max_workers = max_workers
    
    def process_batch(self, queries: List[str], use_cache: bool = True) -> List[Dict]:
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_query = {
                executor.submit(self.pipeline.query, query, use_cache): query 
                for query in queries
            }
            
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

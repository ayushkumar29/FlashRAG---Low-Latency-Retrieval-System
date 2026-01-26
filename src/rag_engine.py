
import os
import time
import json
import shutil
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from tqdm import tqdm

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = BASE_DIR / "cache"
    CHROMA_DIR = CACHE_DIR / "chroma_db"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Ensure directories exist
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    (DATA_DIR / "documents").mkdir(exist_ok=True, parents=True)
    
    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    # Reranker model
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # LLM API
    LLM_PROVIDER = "groq"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.1-8b-instant"
    
    # Retrieval settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_RETRIEVAL = 10
    TOP_K_RERANK = 3
    
    # Cache settings
    CACHE_SIMILARITY_THRESHOLD = 0.90
    CACHE_COLLECTION_NAME = "query_cache"
    
    # Batch processing
    BATCH_SIZE = 10
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 2))
    
    # Web UI settings
    WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
    WEB_PORT = int(os.getenv("PORT", os.getenv("WEB_PORT", 10000)))
    
    # Production settings
    REQUEST_TIMEOUT = 30
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 60))


# ==========================================
# EMBEDDINGS
# ==========================================
class EmbeddingGenerator:
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])[0]
        return embedding.tolist()


# ==========================================
# RERANKER
# ==========================================
class DocumentReranker:
    def __init__(self, model_name: str = Config.RERANKER_MODEL):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = Config.TOP_K_RERANK) -> List[Dict]:
        if not documents:
            return []
        
        pairs = [[query, doc['text']] for doc in documents]
        scores = self.model.predict(pairs)
        
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]


# ==========================================
# SEMANTIC CACHE
# ==========================================
class SemanticCache:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(Config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_gen = EmbeddingGenerator()
        
        self.collection = self.client.get_or_create_collection(
            name=Config.CACHE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    def check_cache(self, query: str, threshold: float = Config.CACHE_SIMILARITY_THRESHOLD) -> Optional[Dict]:
        try:
            query_embedding = self.embedding_gen.embed_query(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1
            )
            
            if results['ids'][0]:
                similarity = 1 - results['distances'][0][0]
                
                if similarity >= threshold:
                    logger.info(f"Cache hit! Similarity: {similarity:.3f}")
                    return {
                        "cached_response": results['metadatas'][0][0]['response'],
                        "similarity": similarity,
                        "cached_query": results['documents'][0][0]
                    }
        except Exception as e:
            logger.error(f"Cache check error: {e}")
        
        return None
    
    def add_to_cache(self, query: str, response: str, retrieved_docs: List[str]):
        try:
            query_embedding = self.embedding_gen.embed_query(query)
            
            self.collection.add(
                embeddings=[query_embedding],
                documents=[query],
                metadatas=[{
                    "response": response,
                    "doc_count": len(retrieved_docs)
                }],
                ids=[f"cache_{self.collection.count()}"]
            )
            logger.info("Added to cache")
        except Exception as e:
            logger.error(f"Cache add error: {e}")


# ==========================================
# DOCUMENT PROCESSOR
# ==========================================
class DocumentProcessor:
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, 
                 chunk_overlap: int = Config.CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_documents(self, directory: str) -> List:
        docs = []
        doc_path = Path(directory)
        
        if not doc_path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        for txt_file in doc_path.glob("*.txt"):
            encodings = ['utf-8', 'utf-16', 'latin-1']
            success = False
            for encoding in encodings:
                try:
                    loader = TextLoader(str(txt_file), encoding=encoding)
                    docs.extend(loader.load())
                    success = True
                    break
                except Exception:
                    continue
            
            if not success:
                print(f"Error loading {txt_file}: Could not decode with any supported encoding")
        
        for pdf_file in doc_path.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")
        
        if not docs:
            raise ValueError(f"No documents found in {directory}")
        
        return docs
    
    def chunk_documents(self, documents: List) -> List[Dict]:
        chunks = self.text_splitter.split_documents(documents)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "id": f"chunk_{i}",
                "text": chunk.page_content,
                "metadata": chunk.metadata
            })
        
        return processed_chunks


# ==========================================
# RETRIEVER
# ==========================================
class DocumentRetriever:
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(
            path=str(Config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_gen = EmbeddingGenerator()
        self.collection_name = collection_name
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def index_documents(self, chunks: List[Dict]):
        texts = [chunk['text'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        print("Generating embeddings...")
        embeddings = self.embedding_gen.embed_documents(texts)
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        logger.info(f"Indexed {len(chunks)} document chunks")
    
    def retrieve(self, query: str, top_k: int = Config.TOP_K_RETRIEVAL) -> List[Dict]:
        query_embedding = self.embedding_gen.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "score": 1 - results['distances'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        
        return retrieved_docs


# ==========================================
# LLM CLIENT
# ==========================================
class LLMClient:
    def __init__(self):
        if Config.LLM_PROVIDER == "groq":
            from groq import Groq
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            self.model = Config.GROQ_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: {Config.LLM_PROVIDER}")
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using LLM with retrieved context"""
        prompt = self._build_prompt(query, context_docs)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Answer questions concisely and accurately based on the provided context. Do not repeat the context documents - only provide a clear, direct answer to the question."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=512,
                stream=False
            )
            
            answer = response.choices[0].message.content
            
            # Clean up any control characters
            answer = ''.join(char for char in answer if char.isprintable() or char in '\n\r\t')
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    def generate_response_stream(self, query: str, context_docs: List[Dict]) -> Iterator[str]:
        """Generate streaming response"""
        prompt = self._build_prompt(query, context_docs)
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Answer questions concisely and accurately based on the provided context. Do not repeat the context documents - only provide a clear, direct answer."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=512,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # Clean content
                    content = ''.join(char for char in content if char.isprintable() or char in '\n\r\t')
                    yield content
                    
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise
    
    def _build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Build improved prompt from query and context"""
        
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            # Clean the document text
            doc_text = doc['text'].strip()
            context_parts.append(f"[Source {i}]\n{doc_text}")
        
        context = "\n\n".join(context_parts)
        
        # IMPROVED PROMPT
        prompt = f"""You are answering a question based on the provided sources.

SOURCES:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Read the sources carefully
2. Answer the question directly and concisely
3. Use information from the sources to support your answer
4. Do NOT repeat or copy the source text
5. Do NOT list the sources in your answer
6. If the answer is not in the sources, say "I don't have enough information to answer that question."

YOUR ANSWER:"""
        
        return prompt


# ==========================================
# MAIN PIPELINE
# ==========================================
class FlashRAGPipeline:
    def __init__(self):
        self.cache = SemanticCache()
        self.retriever = DocumentRetriever()
        self.reranker = DocumentReranker()
        self.llm = LLMClient()
        self._lock = threading.Lock()
    
    def query(self, question: str, use_cache: bool = True) -> Dict:
        start_time = time.time()
        metrics = {
            "cache_hit": False,
            "num_retrieved": 0,
            "num_reranked": 0,
            "latency_ms": 0
        }
        
        if use_cache:
            cached_result = self.cache.check_cache(question)
            if cached_result:
                metrics["cache_hit"] = True
                metrics["latency_ms"] = (time.time() - start_time) * 1000
                return {
                    "answer": cached_result["cached_response"],
                    "source": "cache",
                    "metrics": metrics
                }
        
        retrieved_docs = self.retriever.retrieve(question)
        metrics["num_retrieved"] = len(retrieved_docs)
        
        reranked_docs = self.reranker.rerank(question, retrieved_docs)
        metrics["num_reranked"] = len(reranked_docs)
        
        answer = self.llm.generate_response(question, reranked_docs)
        
        if use_cache:
            with self._lock:
                self.cache.add_to_cache(question, answer, [doc['text'] for doc in reranked_docs])
        
        metrics["latency_ms"] = (time.time() - start_time) * 1000
        
        return {
            "answer": answer,
            "source": "llm",
            "retrieved_docs": reranked_docs,
            "metrics": metrics
        }
    
    def query_stream(self, question: str, use_cache: bool = True) -> Iterator[Dict]:
        start_time = time.time()
        
        if use_cache:
            cached_result = self.cache.check_cache(question)
            if cached_result:
                yield {
                    "type": "cache_hit",
                    "data": cached_result["cached_response"],
                    "done": True,
                    "metrics": {
                        "cache_hit": True,
                        "latency_ms": (time.time() - start_time) * 1000
                    }
                }
                return
        
        retrieved_docs = self.retriever.retrieve(question)
        reranked_docs = self.reranker.rerank(question, retrieved_docs)
        
        yield {
            "type": "retrieval_complete",
            "data": f"Retrieved {len(reranked_docs)} relevant documents",
            "done": False
        }
        
        full_response = ""
        for chunk in self.llm.generate_response_stream(question, reranked_docs):
            full_response += chunk
            yield {
                "type": "content",
                "data": chunk,
                "done": False
            }
        
        if use_cache:
            with self._lock:
                self.cache.add_to_cache(question, full_response, [doc['text'] for doc in reranked_docs])
        
        yield {
            "type": "complete",
            "data": "",
            "done": True,
            "metrics": {
                "cache_hit": False,
                "num_retrieved": len(retrieved_docs),
                "num_reranked": len(reranked_docs),
                "latency_ms": (time.time() - start_time) * 1000
            }
        }


# ==========================================
# BATCH PROCESSOR
# ==========================================
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

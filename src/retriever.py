import chromadb
from chromadb.config import Settings
from typing import List, Dict
from src.config import Config
from src.embeddings import EmbeddingGenerator
import logging

logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(
            path=str(Config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_gen = EmbeddingGenerator()
        self.collection_name = collection_name
        
        # Get or create document collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def index_documents(self, chunks: List[Dict]):
        """Index document chunks into ChromaDB"""
        texts = [chunk['text'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_gen.embed_documents(texts)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        logger.info(f"✓ Indexed {len(chunks)} document chunks")
    
    def retrieve(self, query: str, top_k: int = Config.TOP_K_RETRIEVAL) -> List[Dict]:
        """Retrieve top-k relevant documents"""
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
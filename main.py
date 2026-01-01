import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import after path is set
from src.web_server import app
from src.config import Config

# Auto-index on startup if documents exist
def startup_index():
    """Index documents on startup"""
    try:
        docs_dir = Config.DATA_DIR / "documents"
        if docs_dir.exists() and any(docs_dir.glob("*.txt")) or any(docs_dir.glob("*.pdf")):
            print("📚 Indexing documents on startup...")
            from src.document_processor import DocumentProcessor
            from src.retriever import DocumentRetriever
            
            processor = DocumentProcessor()
            docs = processor.load_documents(str(docs_dir))
            chunks = processor.chunk_documents(docs)
            
            retriever = DocumentRetriever()
            retriever.index_documents(chunks)
            print(f"✅ Indexed {len(chunks)} chunks from {len(docs)} documents")
    except Exception as e:
        print(f"⚠️  Startup indexing error: {e}")

# Run indexing on startup
startup_index()

# For local development
if __name__ == "__main__":
    import uvicorn
    print(f"🌐 Starting FlashRAG on port {Config.WEB_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=Config.WEB_PORT)

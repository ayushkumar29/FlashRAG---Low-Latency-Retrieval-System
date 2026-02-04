import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.server import app
from src.rag_engine import Config

import threading


def startup_index():
    try:
        docs_dir = Config.DATA_DIR / "documents"
        if docs_dir.exists() and any(docs_dir.glob("*.txt")) or any(docs_dir.glob("*.pdf")):
            print("Indexing documents on startup...", flush=True)
            from src.rag_engine import DocumentProcessor, DocumentRetriever
            
            processor = DocumentProcessor()
            docs = processor.load_documents(str(docs_dir))
            chunks = processor.chunk_documents(docs)
            
            retriever = DocumentRetriever()
            retriever.index_documents(chunks)
            print(f"Indexed {len(chunks)} chunks from {len(docs)} documents", flush=True)
    except Exception as e:
        print(f"Startup indexing error: {e}", flush=True)


threading.Thread(target=startup_index, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    print(f"Starting FlashRAG on port {Config.WEB_PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=Config.WEB_PORT)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from src.config import Config
from src.document_processor import DocumentProcessor
from src.retriever import DocumentRetriever
from src.pipeline import FlashRAGPipeline
from src.batch_processor import BatchProcessor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def index_documents(data_dir: str):
    """Index documents"""
    print("📄 Loading documents...")
    processor = DocumentProcessor()
    docs = processor.load_documents(data_dir)
    print(f"✓ Loaded {len(docs)} documents")
    
    print("✂️  Chunking documents...")
    chunks = processor.chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")
    
    print("🔍 Indexing chunks...")
    retriever = DocumentRetriever()
    retriever.index_documents(chunks)
    print("✅ Indexing complete!")

def query_system(question: str, stream: bool = False):
    """Query system"""
    pipeline = FlashRAGPipeline()
    print(f"\n❓ Question: {question}\n")
    
    if stream:
        print("💡 Answer: ", end='', flush=True)
        for chunk in pipeline.query_stream(question):
            if chunk['type'] == 'content':
                print(chunk['data'], end='', flush=True)
            elif chunk['type'] == 'complete':
                print(f"\n\n📊 Metrics:")
                print(f"  - Latency: {chunk['metrics']['latency_ms']:.2f}ms")
    else:
        result = pipeline.query(question)
        print(f"💡 Answer: {result['answer']}\n")
        print(f"📊 Metrics:")
        print(f"  - Cache Hit: {result['metrics']['cache_hit']}")
        print(f"  - Latency: {result['metrics']['latency_ms']:.2f}ms")

def start_server():
    """Start server"""
    import uvicorn
    from src.web_server import app
    
    print(f"🌐 Starting server at http://127.0.0.1:{Config.WEB_PORT}")
    uvicorn.run(app, host="127.0.0.1", port=Config.WEB_PORT)

def main():
    parser = argparse.ArgumentParser(description="FlashRAG CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--data-dir", default=str(Config.DATA_DIR / "documents"))
    
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("question")
    query_parser.add_argument("--stream", action="store_true")
    
    serve_parser = subparsers.add_parser("serve")
    
    args = parser.parse_args()
    
    if args.command == "index":
        index_documents(args.data_dir)
    elif args.command == "query":
        query_system(args.question, args.stream)
    elif args.command == "serve":
        start_server()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

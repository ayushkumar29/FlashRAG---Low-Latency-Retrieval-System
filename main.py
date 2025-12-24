import argparse
import sys
from pathlib import Path
from src.config import Config
from src.document_processor import DocumentProcessor
from src.retriever import DocumentRetriever
from src.pipeline import FlashRAGPipeline
from src.batch_processor import BatchProcessor
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / 'flashrag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def index_documents(data_dir: str):
    """Index documents from directory"""
    print("📄 Loading documents...")
    processor = DocumentProcessor()
    
    try:
        docs = processor.load_documents(data_dir)
        print(f"✓ Loaded {len(docs)} documents")
        
        print("✂️  Chunking documents...")
        chunks = processor.chunk_documents(docs)
        print(f"✓ Created {len(chunks)} chunks")
        
        print("🔍 Indexing chunks...")
        retriever = DocumentRetriever()
        retriever.index_documents(chunks)
        
        print("✅ Indexing complete!")
        print(f"\n💡 Now you can query the system:")
        print(f"   python main.py query \"What is machine learning?\"")
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


def query_system(question: str, stream: bool = False):
    """Query the RAG system"""
    pipeline = FlashRAGPipeline()
    
    print(f"\n❓ Question: {question}\n")
    
    try:
        if stream:
            print("💡 Answer: ", end='', flush=True)
            for chunk in pipeline.query_stream(question):
                if chunk['type'] == 'content':
                    print(chunk['data'], end='', flush=True)
                elif chunk['type'] == 'complete':
                    print(f"\n\n📊 Metrics:")
                    metrics = chunk['metrics']
                    print(f"  - Cache Hit: {metrics['cache_hit']}")
                    print(f"  - Latency: {metrics['latency_ms']:.2f}ms")
                    if not metrics['cache_hit']:
                        print(f"  - Retrieved: {metrics.get('num_retrieved', 'N/A')}")
                        print(f"  - Reranked: {metrics.get('num_reranked', 'N/A')}")
                elif chunk['type'] == 'cache_hit':
                    print(chunk['data'])
                    print(f"\n\n📊 Metrics:")
                    metrics = chunk['metrics']
                    print(f"  - Cache Hit: {metrics['cache_hit']}")
                    print(f"  - Latency: {metrics['latency_ms']:.2f}ms")
        else:
            result = pipeline.query(question)
            print(f"💡 Answer: {result['answer']}\n")
            print(f"📊 Metrics:")
            print(f"  - Source: {result['source']}")
            print(f"  - Cache Hit: {result['metrics']['cache_hit']}")
            print(f"  - Latency: {result['metrics']['latency_ms']:.2f}ms")
            if not result['metrics']['cache_hit']:
                print(f"  - Retrieved: {result['metrics']['num_retrieved']}")
                print(f"  - Reranked: {result['metrics']['num_reranked']}")
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


def batch_process(input_file: str, output_file: str):
    """Process batch queries"""
    print(f"📦 Starting batch processing...")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}\n")
    
    try:
        processor = BatchProcessor()
        processor.process_file(input_file, output_file)
        print(f"\n✅ Batch processing complete!")
        print(f"   Results saved to: {output_file}")
    
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


def start_server():
    """Start web server"""
    import uvicorn
    from src.web_server import app
    
    print(f"🌐 Starting FlashRAG Web Server...")
    print(f"📍 URL: http://{Config.WEB_HOST}:{Config.WEB_PORT}")
    print(f"📚 API Docs: http://{Config.WEB_HOST}:{Config.WEB_PORT}/docs")
    print(f"💡 Press Ctrl+C to stop\n")
    
    try:
        uvicorn.run(
            app,
            host=Config.WEB_HOST,
            port=Config.WEB_PORT,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="FlashRAG - Lightning-fast RAG system with semantic caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index documents
  python main.py index
  python main.py index --data-dir /path/to/docs
  
  # Query (normal response)
  python main.py query "What is machine learning?"
  
  # Query (streaming response)
  python main.py query "What is machine learning?" --stream
  
  # Batch processing
  echo "What is ML?" > queries.txt
  python main.py batch queries.txt results.json
  
  # Start web server
  python main.py serve

For more information, visit: https://github.com/yourusername/flashrag
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index command
    index_parser = subparsers.add_parser(
        "index", 
        help="Index documents from directory"
    )
    index_parser.add_argument(
        "--data-dir", 
        default=str(Config.DATA_DIR / "documents"),
        help="Directory containing documents to index"
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        "query", 
        help="Query the RAG system"
    )
    query_parser.add_argument(
        "question", 
        help="Question to ask the system"
    )
    query_parser.add_argument(
        "--stream", 
        action="store_true",
        help="Use streaming response (real-time generation)"
    )
    
    # Batch command
    batch_parser = subparsers.add_parser(
        "batch", 
        help="Process multiple queries from file"
    )
    batch_parser.add_argument(
        "input", 
        help="Input text file with one query per line"
    )
    batch_parser.add_argument(
        "output", 
        help="Output JSON file for results"
    )
    
    # Serve command
    serve_parser = subparsers.add_parser(
        "serve", 
        help="Start web server and API"
    )
    
    args = parser.parse_args()
    
    # Route to appropriate function
    if args.command == "index":
        index_documents(args.data_dir)
    elif args.command == "query":
        query_system(args.question, args.stream)
    elif args.command == "batch":
        batch_process(args.input, args.output)
    elif args.command == "serve":
        start_server()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

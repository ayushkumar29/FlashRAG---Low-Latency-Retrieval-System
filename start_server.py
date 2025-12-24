# start_server.py - Clean Windows Version
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("🌐 Starting FlashRAG Web Server...")
    print("📍 URL: http://127.0.0.1:8000")
    print("📚 API Docs: http://127.0.0.1:8000/docs")
    print("💡 Press Ctrl+C to stop\n")
    
    try:
        import uvicorn
        
        # Clean uvicorn config - no extra parameters
        uvicorn.run(
            "src.web_server:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
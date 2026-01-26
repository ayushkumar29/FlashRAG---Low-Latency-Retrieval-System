
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("DEBUG: Starting imports...")
from src.document_processor import DocumentProcessor
print("DEBUG: Imports done.")

def test_document_loading():
    print("Testing compliance with fix...")
    processor = DocumentProcessor()
    docs_dir = "data/documents"
    
    try:
        docs = processor.load_documents(docs_dir)
        print(f"Loaded {len(docs)} documents.")
        
        for doc in docs:
            content = doc.page_content
            # Check for common garbage characters from reading UTF-16 as UTF-8
            if "\x00" in content:
                print("FAILURE: Null bytes found in document content!")
                sys.exit(1)
            
            # Check for a known string from the file if possible, or just print a snippet
            print(f"Snippet: {content[:100]}...")
            
        print("SUCCESS: Documents loaded without null bytes.")
        
    except Exception as e:
        print(f"FAILURE: Exception during loading: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_document_loading()

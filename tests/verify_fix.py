import sys
from pathlib import Path

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
            if "\x00" in content:
                print("FAILURE: Null bytes found in document content!")
                sys.exit(1)
            
            print(f"Snippet: {content[:100]}...")
            
        print("SUCCESS: Documents loaded without null bytes.")
        
    except Exception as e:
        print(f"FAILURE: Exception during loading: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_document_loading()

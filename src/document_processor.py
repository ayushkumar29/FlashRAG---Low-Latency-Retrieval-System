from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from src.config import Config

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
            try:
                loader = TextLoader(str(txt_file))
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
        
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

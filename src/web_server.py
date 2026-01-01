from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import asyncio
import shutil
from pathlib import Path
from src.pipeline import FlashRAGPipeline
from src.config import Config
from src.rate_limiter import RateLimiter
from src.metrics_collector import MetricsCollector
from src.document_processor import DocumentProcessor
from src.retriever import DocumentRetriever
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="FlashRAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = FlashRAGPipeline()
rate_limiter = RateLimiter(max_requests=Config.RATE_LIMIT_PER_MINUTE)
metrics = MetricsCollector()

class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True
    stream: bool = False

@app.post("/api/query")
async def query(request: QueryRequest, req: Request):
    client_ip = req.client.host
    
    if not rate_limiter.allow_request(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        if request.stream:
            return StreamingResponse(
                stream_response(request.query, request.use_cache),
                media_type="text/event-stream"
            )
        else:
            result = await asyncio.to_thread(
                pipeline.query, 
                request.query, 
                request.use_cache
            )
            metrics.record_request(result['metrics'])
            return result
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        allowed_extensions = {'.txt', '.pdf', '.docx'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        file_path = Config.DATA_DIR / "documents" / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Reindex
        await asyncio.to_thread(reindex_documents)
        
        return {"status": "success", "message": f"File '{file.filename}' uploaded and indexed"}
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def reindex_documents():
    try:
        processor = DocumentProcessor()
        docs_dir = str(Config.DATA_DIR / "documents")
        
        docs = processor.load_documents(docs_dir)
        chunks = processor.chunk_documents(docs)
        
        retriever = DocumentRetriever()
        retriever.index_documents(chunks)
        
        logger.info(f"Reindexed {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        raise

@app.get("/api/metrics")
async def get_metrics():
    return metrics.get_summary()

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

async def stream_response(query: str, use_cache: bool):
    for chunk in pipeline.query_stream(query, use_cache):
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>FlashRAG</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .header h1 { font-size: 3rem; margin-bottom: 10px; }
        .card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            min-height: 100px;
            font-family: inherit;
        }
        textarea:focus { outline: none; border-color: #667eea; }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 15px;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .response {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            min-height: 150px;
            line-height: 1.8;
        }
        .metrics {
            display: flex;
            gap: 15px;
            margin: 15px 0;
            flex-wrap: wrap;
        }
        .metric {
            background: white;
            padding: 8px 15px;
            border-radius: 6px;
            font-size: 14px;
        }
        .cache-badge {
            background: #10b981;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
        }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ FlashRAG</h1>
            <p>Lightning-fast document Q&A with semantic caching</p>
        </div>
        <div class="card">
            <label>Ask a Question</label>
            <textarea id="query" placeholder="e.g., What is machine learning?"></textarea>
            <button class="btn" onclick="submitQuery()">
                <span id="btnText">Ask Question</span>
            </button>
            <div id="response" class="response hidden"></div>
            <div id="metrics" class="metrics hidden"></div>
        </div>
    </div>
    <script>
        async function submitQuery() {
            const query = document.getElementById('query').value.trim();
            if (!query) { alert('Please enter a question'); return; }
            
            document.getElementById('btnText').textContent = 'Loading...';
            document.getElementById('response').classList.remove('hidden');
            document.getElementById('response').textContent = 'Processing...';
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, use_cache: true })
                });
                
                const data = await response.json();
                document.getElementById('response').textContent = data.answer;
                
                const metricsDiv = document.getElementById('metrics');
                metricsDiv.classList.remove('hidden');
                metricsDiv.innerHTML = `
                    ${data.metrics.cache_hit ? '<span class="cache-badge">⚡ CACHE HIT</span>' : ''}
                    <div class="metric"><strong>Latency:</strong> ${data.metrics.latency_ms.toFixed(0)}ms</div>
                    ${data.metrics.num_retrieved ? `<div class="metric"><strong>Retrieved:</strong> ${data.metrics.num_retrieved} docs</div>` : ''}
                `;
            } catch (error) {
                document.getElementById('response').textContent = 'Error: ' + error.message;
            } finally {
                document.getElementById('btnText').textContent = 'Ask Question';
            }
        }
    </script>
</body>
</html>
    """

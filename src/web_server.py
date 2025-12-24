from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import asyncio
from src.pipeline import FlashRAGPipeline
from src.config import Config
from src.rate_limiter import RateLimiter
from src.metrics_collector import MetricsCollector
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

class BatchQueryRequest(BaseModel):
    queries: List[str]
    use_cache: bool = True

@app.post("/api/query")
async def query(request: QueryRequest, req: Request):
    """Single query endpoint"""
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

@app.post("/api/batch")
async def batch_query(request: BatchQueryRequest):
    """Batch query endpoint"""
    from src.batch_processor import BatchProcessor
    
    try:
        processor = BatchProcessor()
        results = await asyncio.to_thread(
            processor.process_batch,
            request.queries,
            request.use_cache
        )
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Batch query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics"""
    return metrics.get_summary()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

async def stream_response(query: str, use_cache: bool):
    """Generator for streaming responses"""
    for chunk in pipeline.query_stream(query, use_cache):
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve web UI"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlashRAG - Smart Document Q&A</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .header p { font-size: 1.1rem; opacity: 0.9; }
        .card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }
        .input-group { margin-bottom: 20px; }
        .input-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 100px;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .options {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .checkbox-group input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .response-area {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            min-height: 200px;
            margin-top: 20px;
        }
        .response-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }
        .metrics {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }
        .metric {
            background: white;
            padding: 8px 15px;
            border-radius: 6px;
            font-size: 14px;
            color: #666;
        }
        .metric strong { color: #333; }
        .cache-badge {
            background: #10b981;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .answer-text {
            line-height: 1.8;
            color: #333;
            white-space: pre-wrap;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .hidden { display: none; }
        .error { color: #dc2626; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ FlashRAG</h1>
            <p>Lightning-fast document Q&A with semantic caching</p>
        </div>

        <div class="card">
            <div class="input-group">
                <label for="query">Ask a Question</label>
                <textarea id="query" placeholder="e.g., What is machine learning?"></textarea>
            </div>

            <div class="options">
                <div class="checkbox-group">
                    <input type="checkbox" id="useCache" checked>
                    <label for="useCache">Use Cache</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="useStream">
                    <label for="useStream">Stream Response</label>
                </div>
            </div>

            <button class="btn" id="submitBtn" onclick="submitQuery()">
                <span id="btnText">Ask Question</span>
                <span id="btnLoader" class="loading hidden"></span>
            </button>
        </div>

        <div class="card" id="responseCard" style="display: none;">
            <div class="response-header">
                <h3>📝 Response</h3>
                <div id="cacheIndicator"></div>
            </div>
            <div class="metrics" id="metricsContainer"></div>
            <div class="response-area">
                <div class="answer-text" id="answerText"></div>
            </div>
        </div>
    </div>

    <script>
        async function submitQuery() {
            const query = document.getElementById('query').value.trim();
            if (!query) {
                alert('Please enter a question');
                return;
            }

            const useCache = document.getElementById('useCache').checked;
            const useStream = document.getElementById('useStream').checked;

            document.getElementById('submitBtn').disabled = true;
            document.getElementById('btnText').classList.add('hidden');
            document.getElementById('btnLoader').classList.remove('hidden');
            document.getElementById('responseCard').style.display = 'block';
            document.getElementById('answerText').textContent = '';

            try {
                if (useStream) {
                    await handleStreamResponse(query, useCache);
                } else {
                    await handleNormalResponse(query, useCache);
                }
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('answerText').innerHTML = 
                    '<span class="error">Error: ' + error.message + '</span>';
            } finally {
                document.getElementById('submitBtn').disabled = false;
                document.getElementById('btnText').classList.remove('hidden');
                document.getElementById('btnLoader').classList.add('hidden');
            }
        }

        async function handleNormalResponse(query, useCache) {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, use_cache: useCache, stream: false })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            console.log('Response data:', data); // Debug log
            
            // Safely access answer
            const answer = data.answer || 'No answer received';
            document.getElementById('answerText').textContent = answer;
            
            // Safely access metrics
            if (data.metrics) {
                updateMetrics(data.metrics);
            } else {
                console.warn('No metrics in response');
            }
        }

        async function handleStreamResponse(query, useCache) {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, use_cache: useCache, stream: true })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            console.log('Stream data:', data); // Debug log
                            
                            if (data.type === 'content') {
                                document.getElementById('answerText').textContent += data.data;
                            } else if (data.type === 'complete') {
                                if (data.metrics) {
                                    updateMetrics(data.metrics);
                                }
                            } else if (data.type === 'cache_hit') {
                                document.getElementById('answerText').textContent = data.data;
                                if (data.metrics) {
                                    updateMetrics(data.metrics);
                                }
                            }
                        } catch (e) {
                            console.error('Error parsing stream data:', e);
                        }
                    }
                }
            }
        }

        function updateMetrics(metrics) {
            console.log('Updating metrics:', metrics); // Debug log
            
            // Safely access cache_hit with default value
            const cacheHit = metrics && metrics.cache_hit === true;
            const latency = metrics && metrics.latency_ms ? metrics.latency_ms : 0;
            const numRetrieved = metrics && metrics.num_retrieved ? metrics.num_retrieved : 0;
            const numReranked = metrics && metrics.num_reranked ? metrics.num_reranked : 0;
            
            const cacheIndicator = document.getElementById('cacheIndicator');
            if (cacheHit) {
                cacheIndicator.innerHTML = '<span class="cache-badge">⚡ CACHE HIT</span>';
            } else {
                cacheIndicator.innerHTML = '';
            }

            const metricsContainer = document.getElementById('metricsContainer');
            let metricsHtml = `<div class="metric"><strong>Latency:</strong> ${latency.toFixed(2)}ms</div>`;
            
            if (numRetrieved > 0) {
                metricsHtml += `<div class="metric"><strong>Retrieved:</strong> ${numRetrieved} docs</div>`;
            }
            if (numReranked > 0) {
                metricsHtml += `<div class="metric"><strong>Reranked:</strong> ${numReranked} docs</div>`;
            }
            
            metricsContainer.innerHTML = metricsHtml;
        }

        document.getElementById('query').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                submitQuery();
            }
        });
    </script>
</body>
</html>
    """
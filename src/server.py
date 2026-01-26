
import asyncio
import logging
import json
import time
from collections import defaultdict
from threading import Lock
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import combined engine
from src.rag_engine import (
    Config, 
    FlashRAGPipeline, 
    DocumentProcessor, 
    DocumentRetriever
)

logger = logging.getLogger(__name__)

# ==========================================
# RATE LIMITER
# ==========================================
class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = Lock()
    
    def allow_request(self, client_id: str) -> bool:
        with self.lock:
            current_time = time.time()
            
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < self.window_seconds
            ]
            
            if len(self.requests[client_id]) >= self.max_requests:
                return False
            
            self.requests[client_id].append(current_time)
            return True


# ==========================================
# METRICS COLLECTOR
# ==========================================
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_latency": 0,
            "avg_latency": 0,
            "requests_per_second": 0,
            "start_time": time.time()
        }
        self.lock = Lock()
    
    def record_request(self, request_metrics: dict):
        with self.lock:
            self.metrics["total_requests"] += 1
            
            if request_metrics.get("cache_hit"):
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1
            
            self.metrics["total_latency"] += request_metrics.get("latency_ms", 0)
            self.metrics["avg_latency"] = (
                self.metrics["total_latency"] / self.metrics["total_requests"]
            )
            
            elapsed = time.time() - self.metrics["start_time"]
            self.metrics["requests_per_second"] = self.metrics["total_requests"] / elapsed
    
    def get_summary(self) -> dict:
        with self.lock:
            cache_hit_rate = (
                self.metrics["cache_hits"] / self.metrics["total_requests"] * 100
                if self.metrics["total_requests"] > 0 else 0
            )
            
            return {
                **self.metrics,
                "cache_hit_rate": f"{cache_hit_rate:.2f}%"
            }


# ==========================================
# WEB SERVER
# ==========================================
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
                media_type="text/event-stream; charset=utf-8"
            )
        else:
            result = await asyncio.to_thread(
                pipeline.query, 
                request.query, 
                request.use_cache
            )
            
            if isinstance(result.get('answer'), str):
                result['answer'] = result['answer'].encode('utf-8', errors='ignore').decode('utf-8')
            
            metrics.record_request(result['metrics'])
            
            return JSONResponse(
                content=result,
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear-cache")
async def clear_cache():
    """Clear the semantic cache"""
    try:
        from src.rag_engine import SemanticCache
        cache = SemanticCache()
        cache.clear_cache()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        allowed_extensions = {'.txt', '.pdf', '.docx'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        file_path = Config.DATA_DIR / "documents" / file.filename
        
        if file_extension == '.txt':
            content = await file.read()
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(content.decode('utf-8', errors='ignore'))
        else:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        
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
    try:
        for chunk in pipeline.query_stream(query, use_cache):
            if 'data' in chunk and isinstance(chunk['data'], str):
                chunk['data'] = chunk['data'].encode('utf-8', errors='ignore').decode('utf-8')
            
            json_str = json.dumps(chunk, ensure_ascii=False)
            yield f"data: {json_str}\n\n".encode('utf-8')
            await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {"type": "error", "data": str(e), "done": True}
        yield f"data: {json.dumps(error_chunk)}\n\n".encode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlashRAG // Neo</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-dark: #09090b;
            --bg-card: rgba(24, 24, 27, 0.6);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.4);
            --accent: #06b6d4;
            --text-main: #f4f4f5;
            --text-muted: #a1a1aa;
            --border: rgba(255, 255, 255, 0.1);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.08), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(6, 182, 212, 0.08), transparent 25%);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            height: 100vh;
            display: flex;
            overflow: hidden;
        }

        /* Glassmorphism Utilities */
        .glass {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border);
        }

        /* Sidebar */
        .sidebar {
            width: 280px;
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--border);
            padding: 24px;
            gap: 24px;
            z-index: 10;
        }

        .brand {
            font-size: 1.5rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
            letter-spacing: -0.5px;
        }

        .brand-icon {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 8px;
            display: grid;
            place-items: center;
            font-size: 18px;
        }

        .history-list {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .history-item {
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9rem;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .history-item:hover, .history-item.active {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
        }

        /* Main Chat Area */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }

        .header {
            height: 70px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 40px;
            border-bottom: 1px solid var(--border);
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            color: #10b981;
            background: rgba(16, 185, 129, 0.1);
            padding: 6px 12px;
            border-radius: 20px;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 40px;
            padding-bottom: 140px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            scroll-behavior: smooth;
        }

        /* Messages */
        .message {
            display: flex;
            gap: 16px;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
            animation: fadeIn 0.3s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            flex-shrink: 0;
            display: grid;
            place-items: center;
            font-weight: 600;
            font-size: 0.9rem;
        }

        .avatar.ai {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            box-shadow: 0 0 15px var(--primary-glow);
        }

        .avatar.user {
            background: #27272a;
            color: var(--text-muted);
        }

        .message-content {
            flex: 1;
            line-height: 1.6;
            font-size: 1rem;
        }

        .message-content p { margin-bottom: 1rem; }
        .message-content p:last-child { margin-bottom: 0; }
        
        .message.user .message-content {
            color: var(--text-muted);
        }

        /* Metrics Card */
        .metrics-card {
            margin-top: 12px;
            display: inline-flex;
            gap: 16px;
            padding: 10px 16px;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border);
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
        }

        .metric-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .metric-value {
            color: var(--accent);
            font-weight: 700;
        }

        /* Input Area */
        .input-area {
            position: absolute;
            bottom: 40px;
            left: 50%;
            transform: translateX(-50%);
            width: 90%;
            max-width: 800px;
        }

        .input-box {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 8px;
            display: flex;
            gap: 12px;
            align-items: flex-end;
            backdrop-filter: blur(20px);
            box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);
            transition: border-color 0.2s;
        }

        .input-box:focus-within {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px var(--primary-glow);
        }

        textarea {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-main);
            padding: 12px 16px;
            font-family: inherit;
            font-size: 1rem;
            resize: none;
            max-height: 200px;
            min-height: 24px;
        }

        textarea:focus { outline: none; }

        .send-btn {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            border: none;
            background: var(--primary);
            color: white;
            cursor: pointer;
            display: grid;
            place-items: center;
            transition: all 0.2s;
        }

        .send-btn:hover {
            background: var(--accent);
            transform: scale(1.05);
        }

        .send-btn:disabled {
            background: #3f3f46;
            cursor: not-allowed;
            transform: none;
        }

        .options-bar {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 12px;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        
        .switch {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
        }
        
        .switch input { opacity: 0; width: 0; }
        .slider {
            width: 32px;
            height: 18px;
            background: #3f3f46;
            border-radius: 20px;
            position: relative;
            transition: .3s;
        }
        .slider:before {
            content: "";
            position: absolute;
            height: 14px;
            width: 14px;
            left: 2px;
            bottom: 2px;
            background: white;
            border-radius: 50%;
            transition: .3s;
        }
        input:checked + .slider { background: var(--primary); }
        input:checked + .slider:before { transform: translateX(14px); }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="sidebar glass">
        <div class="brand">
            <div class="brand-icon">⚡</div>
            FlashRAG
        </div>
        <div class="history-list">
            <div class="history-item active">New Chat</div>
        </div>
    </div>

    <div class="main">
        <div class="header glass">
            <div style="font-weight: 500;">New Session</div>
            <div class="status-badge">● System Online</div>
        </div>

        <div class="chat-container" id="chatContainer">
            <!-- Messages will appear here -->
            <div class="message">
                <div class="avatar ai">AI</div>
                <div class="message-content">
                    <p>Hello! I'm FlashRAG, your low-latency knowledge assistant. I can answer questions using your provided documents with millisecond-level speeds. How can I help?</p>
                </div>
            </div>
        </div>

        <div class="input-area">
            <div class="options-bar">
                <label class="switch">
                    <input type="checkbox" id="useCache" checked>
                    <span class="slider"></span>
                    Semantic Cache
                </label>
                <label class="switch">
                    <input type="checkbox" id="useStream" checked>
                    <span class="slider"></span>
                    Stream Response
                </label>
            </div>
            <div class="input-box">
                <textarea id="query" rows="1" placeholder="Ask anything..." oninput="this.style.height = ''; this.style.height = this.scrollHeight + 'px'"></textarea>
                <button class="send-btn" id="submitBtn" onclick="submitQuery()">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                </button>
            </div>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chatContainer');
        const queryInput = document.getElementById('query');
        
        // Auto resize input
        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitQuery();
            }
        });

        function appendMessage(role, content) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}`;
            msgDiv.innerHTML = `
                <div class="avatar ${role}">${role === 'ai' ? 'AI' : 'U'}</div>
                <div class="message-content">${role === 'ai' ? marked.parse(content) : `<p>${content}</p>`}</div>
            `;
            chatContainer.appendChild(msgDiv);
            scrollToBottom();
            return msgDiv;
        }

        function appendMetrics(msgDiv, metrics) {
            const contentDiv = msgDiv.querySelector('.message-content');
            const metricsDiv = document.createElement('div');
            metricsDiv.className = 'metrics-card';
            
            let html = `
                <div class="metric-item">
                    <span>⚡</span>
                    <span class="metric-value">${Math.round(metrics.latency_ms)}ms</span>
                </div>
            `;
            
            if (metrics.cache_hit) {
                html += `<div class="metric-item"><span style="color:#10b981">● Cache Hit</span></div>`;
            }
            if (metrics.num_retrieved) {
                html += `<div class="metric-item"><span>📚</span><span class="metric-value">${metrics.num_retrieved}</span> src</div>`;
            }
            
            metricsDiv.innerHTML = html;
            contentDiv.appendChild(metricsDiv);
            scrollToBottom();
        }

        function scrollToBottom() {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function submitQuery() {
            const query = queryInput.value.trim();
            if (!query) return;

            // Reset input
            queryInput.value = '';
            queryInput.style.height = 'auto';

            // Add user message
            appendMessage('user', query);

            const useCache = document.getElementById('useCache').checked;
            const useStream = document.getElementById('useStream').checked;
            const btn = document.getElementById('submitBtn');
            btn.disabled = true;

            try {
                // Create AI message placeholder
                let aiMsgDiv = appendMessage('ai', '<span class="typing">Thinking...</span>');
                let aiContentDiv = aiMsgDiv.querySelector('.message-content');
                let currentText = '';

                if (useStream) {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ query, use_cache: useCache, stream: true })
                    });

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value, {stream: true});
                        const lines = chunk.split('\\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    if (data.type === 'content') {
                                        currentText += data.data;
                                        aiContentDiv.innerHTML = marked.parse(currentText);
                                    } else if (data.type === 'complete' || data.type === 'cache_hit') {
                                        if (data.type === 'cache_hit') {
                                            aiContentDiv.innerHTML = marked.parse(data.data);
                                        }
                                        if (data.metrics) {
                                            appendMetrics(aiMsgDiv, data.metrics);
                                        }
                                    }
                                    scrollToBottom();
                                } catch (e) {}
                            }
                        }
                    }
                } else {
                    // Normal Request
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ query, use_cache: useCache, stream: false })
                    });
                    const data = await response.json();
                    
                    aiContentDiv.innerHTML = marked.parse(data.answer);
                    if (data.metrics) {
                        appendMetrics(aiMsgDiv, data.metrics);
                    }
                }

            } catch (error) {
                appendMessage('ai', `⚠️ Error: ${error.message}`);
            } finally {
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
    """

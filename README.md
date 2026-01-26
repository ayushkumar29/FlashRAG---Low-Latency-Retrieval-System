# FlashRAG - Low-Latency RAG System

A production-ready Retrieval-Augmented Generation system that delivers millisecond-level responses through intelligent semantic caching and efficient vector retrieval.

## Features

- **Semantic Caching**: Reduces response time from 2500ms to under 45ms for cached queries.
- **Accurate Retrieval**: Uses Cross-Encoder reranking (MS-MARCO) for high precision.
- **Real-time Streaming**: Instant token generation for better user experience.
- **Document Support**: Upload and index PDF, TXT, and DOCX files.
- **Production Ready**: Includes rate limiting, metrics monitoring, and robust error handling.

## Quick Start (Local)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd flashrag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   Set your Groq API key (required for LLM):
   - Windows (PowerShell): `$env:GROQ_API_KEY="your_key_here"`
   - Linux/Mac: `export GROQ_API_KEY=your_key_here`

4. **Run the server**
   ```bash
   python main.py
   ```
   Access the application at http://127.0.0.1:10000

## How It Works

1. **User asks a question**.
2. **Semantic Cache Check**: If a similar question was asked before, return the cached answer immediately (45ms).
3. **Retrieval (if cache miss)**: Search the vector database for relevant document chunks.
4. **Reranking**: Use a high-precision model to rank the retrieved chunks.
5. **Generation**: Send the best chunks to the LLM (Llama 3.1) to generate an answer.
6. **Cache Update**: Store the new question-answer pair for future use.

## Deployment (Render)

This project is configured for one-click deployment on Render.

1. Push your code to a GitHub repository.
2. Link the repository to a new Render Web Service.
3. Set the environment variable `GROQ_API_KEY`.
4. Render will automatically build and deploy using the included configuration.

## API Usage

- **Query**: `POST /api/query` - Send JSON with `{"query": "your question"}`.
- **Upload**: `POST /api/upload` - Upload a document file.
- **Metrics**: `GET /api/metrics` - View system performance stats.

## License

MIT License
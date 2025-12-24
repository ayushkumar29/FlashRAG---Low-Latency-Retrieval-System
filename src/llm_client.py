from typing import List, Dict, Iterator
from src.config import Config
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        if Config.LLM_PROVIDER == "groq":
            from groq import Groq
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            self.model = Config.GROQ_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: {Config.LLM_PROVIDER}")
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using LLM"""
        prompt = self._build_prompt(query, context_docs)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    def generate_response_stream(self, query: str, context_docs: List[Dict]) -> Iterator[str]:
        """Generate streaming response"""
        prompt = self._build_prompt(query, context_docs)
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise
    
    def _build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Build prompt from query and context"""
        context = "\n\n".join([
            f"Document {i+1}:\n{doc['text']}" 
            for i, doc in enumerate(context_docs)
        ])
        
        return f"""Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

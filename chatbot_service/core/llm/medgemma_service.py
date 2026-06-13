import logging
import httpx
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


class MedGemmaService:
    """
    Client for the local MedGemma-4B GGUF server.
    
    The 'Mouth' of the AI.
    Connects to a local llama.cpp server running MedGemma-4B-GGUF model
    to generate natural language answers based on medical context from RAG.
    
    This approach:
    - Eliminates GPU memory constraints
    - Allows model to run on any GPU efficiently
    - Separates LLM serving from main application
    
    Prerequisites:
    - Run: llama-server -m medgemma-4b.gguf --port 8080
    """
    _instance = None
    # Fixed port to 8080 (standard llama.cpp port)
    SERVER_URL = "http://127.0.0.1:8090/completion"
    HEALTH_URL = "http://127.0.0.1:8090/health"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MedGemmaService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Check connection to MedGemma server (non-blocking check)."""
        # We do a quick sync check here, or just log that we are assuming it's up.
        # To avoid blocking startup significantly, we use a very short timeout.
        try:
            with httpx.Client(timeout=2.0) as client:
                client.get(self.HEALTH_URL)
            logger.info("✅ Connected to MedGemma-4B (Local GPU Server)")
        except httpx.RequestError as e:
            logger.warning(f"⚠️ MedGemma Server not reachable at startup ({e}). Ensure llama-server is running on port 8080.")
        except Exception as e:
            logger.warning(f"⚠️ MedGemma health check failed: {e}")

    @classmethod
    def get_instance(cls):
        return cls()

    async def generate_response(self, query: str, context: str) -> str:
        """
        Generates a medical answer using the retrieved context.
        
        Connects to local MedGemma server to generate a response.
        
        Args:
            query: The user's medical question
            context: The retrieved medical context from RAG
            
        Returns:
            A natural language response from the LLM, or error message if server unavailable
        """
        # Prompt specifically formatted for Gemma
        prompt = f"""<start_of_turn>user
You are HeartGuard, a medical AI assistant.
Answer based ONLY on the context provided.

CONTEXT:
{context}

QUESTION:
{query}<end_of_turn>
<start_of_turn>model
"""
        
        payload = {
            "prompt": prompt,
            "temperature": 0.2,
            "n_predict": 2048,
            "stop": ["<end_of_turn>"],
            "cache_prompt": True
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.SERVER_URL, json=payload)
                
                if resp.status_code == 200:
                    return resp.json().get("content", "").strip()
                else:
                    logger.error(f"MedGemma returned status {resp.status_code}: {resp.text}")
                    return f"Error: AI Server returned {resp.status_code}"
                    
        except httpx.RequestError as e:
            logger.error(f"Connection error to MedGemma: {e}")
            return "Error connecting to AI Brain. Is the server running?"
        except Exception as e:
            logger.error(f"Unexpected error in MedGemma generation: {e}")
            return f"Error generating response: {e}"

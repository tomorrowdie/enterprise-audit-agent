"""
Model Configuration Factory
Phase 1: Cloud APIs (Claude/Gemini)
Phase 2: Local Ollama models
"""
import os
from langchain_community.chat_models import ChatOllama

# Try importing Google Generative AI
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


def get_llm(use_local=True, local_model="llama3", api_key=None, cloud_model_name="gemini-2.5-flash"):
    """
    Factory function with Dynamic Model Selection.
    Defaults to gemini-2.5-flash for Cloud.

    PRIORITY:
    1. Local Ollama (if use_local=True)
    2. Google Gemini (if api_key provided)
    """
    # 1. Local Mode
    if use_local:
        print(f"[LOCAL] Using Local Ollama Model: {local_model}")
        return ChatOllama(model=local_model, temperature=0, format="json")

    # 2. Cloud Mode (Gemini)
    if GOOGLE_AVAILABLE:
        # Check if key was passed from Sidebar OR .env
        final_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if final_key and "your_" not in final_key:  # Check against placeholder
            # === FIX: Use the passed model name ===
            print(f"[CLOUD] Using Google Cloud Model: {cloud_model_name}")
            return ChatGoogleGenerativeAI(
                model=cloud_model_name,
                google_api_key=final_key,
                temperature=0
            )

    print("[WARNING] Cloud not configured (Missing/Invalid Key). Falling back to Ollama.")
    return ChatOllama(model=local_model, temperature=0, format="json")

# src/llm.py
import os
from typing import Optional

from src.llm_ollama import ollama_generate

def _auto_provider() -> str:
    # Priority: explicit flag wins
    p = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if p in ("openai", "ollama"):
        return p
    # Default behavior:
    # - If Streamlit Cloud secrets provide OPENAI_API_KEY, can still force openai via LLM_PROVIDER.
    # - Otherwise default to ollama (local)
    return "ollama"

def llm_generate(prompt: str, *, model: str = "qwen2.5:7b", temperature: float = 0.2, provider: Optional[str] = None) -> str:
    provider = (provider or _auto_provider()).lower()

    if provider == "openai":
        # If caller passed an Ollama-ish model name, swap to your OpenAI default
        if ":" in model or model.lower().startswith("qwen"):
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    # Default: Ollama
    return ollama_generate(prompt, model=model, temperature=temperature)

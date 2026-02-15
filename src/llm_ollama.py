import requests
from typing import Optional
import os

OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_generate(prompt: str, model: str = "qwen2.5:7b", temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()

if __name__ == "__main__":
    print(ollama_generate("Say hi in one sentence."))
"""
llm_engine.py — Hybrid Local GPU (RTX 5060 Ti) & Cloud LLM Analyst

Provides dual-mode AI analysis for NBA player props and game matchups:
  1. Local GPU Mode: Checks for local Ollama / OpenAI-compatible endpoint (e.g. Llama 3.1 8B, Qwen 2.5 14B)
     running on the NVIDIA GeForce RTX 5060 Ti 16GB. (100% offline, zero API fees, no rate limits).
  2. Cloud Fallback: Seamlessly falls back to Groq API (or OpenAI) if local server is not active.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Also load from .streamlit/secrets.toml if GROQ_API_KEY not yet in environment
if not os.getenv("GROQ_API_KEY"):
    secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            import toml
            sec = toml.loads(secrets_path.read_text(encoding="utf-8"))
            if "GROQ_API_KEY" in sec:
                os.environ["GROQ_API_KEY"] = sec["GROQ_API_KEY"]
        except Exception:
            # Fallback simple line parsing
            for line in secrets_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GROQ_API_KEY"):
                    val = line.split("=", 1)[-1].strip().strip('"').strip("'")
                    os.environ["GROQ_API_KEY"] = val
                    break

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
LOCAL_OPENAI_ENDPOINT = os.getenv("LOCAL_OPENAI_ENDPOINT", "http://localhost:1234/v1")


def check_local_ollama() -> Tuple[bool, Optional[str]]:
    """
    Checks if Ollama is running locally and returns (is_available, default_model_name).
    """
    try:
        resp = requests.get(f"{OLLAMA_ENDPOINT}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", [])]
            if models:
                # Prefer high-performance sports analysis models
                for preferred in ["qwen2.5:14b", "qwen2.5", "llama3.1:8b", "llama3.1", "mistral"]:
                    for m in models:
                        if preferred in m.lower():
                            return True, m
                return True, models[0]
            return True, "llama3.1"
    except Exception:
        pass
    return False, None


def check_local_openai() -> Tuple[bool, Optional[str]]:
    """
    Checks if a local OpenAI-compatible server (LM Studio, vLLM, etc.) is active on port 1234.
    """
    try:
        resp = requests.get(f"{LOCAL_OPENAI_ENDPOINT}/models", timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("id") for m in data.get("data", [])]
            if models:
                return True, models[0]
    except Exception:
        pass
    return False, None


def get_active_llm_provider() -> Dict[str, Any]:
    """
    Detect the active LLM backend and return status info for the UI.
    """
    # 1. Check local Ollama on RTX 5060 Ti
    has_ollama, ollama_model = check_local_ollama()
    if has_ollama and ollama_model:
        return {
            "mode": "local_ollama",
            "name": f"RTX 5060 Ti GPU (Ollama: {ollama_model})",
            "model": ollama_model,
            "is_local": True,
        }

    # 2. Check local LM Studio / OpenAI-compatible
    has_local_ai, local_ai_model = check_local_openai()
    if has_local_ai and local_ai_model:
        return {
            "mode": "local_openai",
            "name": f"RTX 5060 Ti GPU (Local: {local_ai_model})",
            "model": local_ai_model,
            "is_local": True,
        }

    # 3. Fallback to Groq Cloud API
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return {
            "mode": "cloud_groq",
            "name": "Groq Cloud API (llama-3.3-70b-versatile)",
            "model": "llama-3.3-70b-versatile",
            "is_local": False,
        }

    return {
        "mode": "none",
        "name": "No LLM Configured",
        "model": None,
        "is_local": False,
    }


def query_llm(
    prompt: str,
    system_prompt: str = "You are a professional NBA sports betting quantitative analyst.",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Query the active LLM (preferring local RTX 5060 Ti GPU, falling back to Groq cloud).
    Returns dict: {"response": str, "provider": str, "model": str, "success": bool}
    """
    provider = get_active_llm_provider()
    mode = provider["mode"]

    # --- Mode 1: Local Ollama on GPU ---
    if mode == "local_ollama":
        try:
            payload = {
                "model": provider["model"],
                "prompt": f"{system_prompt}\n\n{prompt}",
                "stream": False,
                "options": {"temperature": temperature},
            }
            resp = requests.post(f"{OLLAMA_ENDPOINT}/api/generate", json=payload, timeout=45)
            if resp.status_code == 200:
                answer = resp.json().get("response", "").strip()
                return {
                    "response": answer,
                    "provider": provider["name"],
                    "model": provider["model"],
                    "success": True,
                }
        except Exception as e:
            logger.warning(f"Local Ollama generation failed: {e}. Attempting cloud fallback...")

    # --- Mode 2: Local OpenAI / LM Studio on GPU ---
    if mode == "local_openai":
        try:
            payload = {
                "model": provider["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }
            resp = requests.post(f"{LOCAL_OPENAI_ENDPOINT}/chat/completions", json=payload, timeout=45)
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"].strip()
                return {
                    "response": answer,
                    "provider": provider["name"],
                    "model": provider["model"],
                    "success": True,
                }
        except Exception as e:
            logger.warning(f"Local OpenAI generation failed: {e}. Attempting cloud fallback...")

    # --- Mode 3: Groq Cloud API Fallback ---
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            completion = client.chat.completions.create(
                model=provider.get("model") or "llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=600,
            )
            return {
                "response": completion.choices[0].message.content.strip(),
                "provider": "Groq Cloud API",
                "model": "llama-3.3-70b-versatile",
                "success": True,
            }
        except Exception as e:
            logger.error(f"Groq Cloud generation failed: {e}")
            return {
                "response": f"AI generation error: {e}",
                "provider": "Groq Cloud API",
                "model": "N/A",
                "success": False,
            }

    return {
        "response": (
            "No LLM provider available. Either run Ollama locally on your RTX 5060 Ti "
            "(ollama run llama3.1) or configure GROQ_API_KEY in your .env file."
        ),
        "provider": "None",
        "model": "None",
        "success": False,
    }

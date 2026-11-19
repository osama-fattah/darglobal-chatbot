"""
OpenRouter client limited to free models.

Free models on OpenRouter come and go and are rate-limited, so on startup we
read the live model list, keep only the free ones, rank them by a preference
list, and fall back to the next model when one errors or is rate-limited.
"""
from __future__ import annotations

import logging
import os
import time

import httpx

log = logging.getLogger("llm")

API = "https://openrouter.ai/api/v1"

# Tried in this order when available as ":free". Override with OPENROUTER_MODELS.
PREFERRED = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen3-235b-a22b:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "meta-llama/llama-4-maverick:free",
    "qwen/qwen-2.5-72b-instruct:free",
]


class OpenRouter:
    def __init__(self) -> None:
        self.key = os.getenv("OPENROUTER_API_KEY", "").strip()
        env_models = [m.strip() for m in os.getenv("OPENROUTER_MODELS", "").split(",") if m.strip()]
        self.models: list[str] = env_models or list(PREFERRED)
        self.last_model: str | None = None
        self._client = httpx.Client(
            timeout=60,
            headers={
                "Authorization": f"Bearer {self.key}",
                "HTTP-Referer": os.getenv("APP_URL", "https://huggingface.co/spaces"),
                "X-Title": "Dar Global & Wasalt Assistant",
            },
        )

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def discover_free_models(self) -> None:
        """Keep preferred models that are live and free, then add other free ones."""
        if os.getenv("OPENROUTER_MODELS"):
            return
        try:
            r = self._client.get(f"{API}/models")
            r.raise_for_status()
            live = r.json().get("data", [])
        except Exception as e:
            log.warning("Could not list OpenRouter models, using defaults: %s", e)
            return
        free = {
            m["id"]: m.get("context_length", 0)
            for m in live
            if m["id"].endswith(":free")
            and str(m.get("pricing", {}).get("prompt", "0")) in ("0", "0.0")
            and "text" in (m.get("architecture", {}).get("output_modalities") or ["text"])
        }
        ranked = [m for m in PREFERRED if m in free]
        # then well-known open model families, then anything else free
        families = ("llama", "deepseek", "qwen", "gemma", "mistral", "gpt-oss", "nemotron", "glm", "kimi")
        fam = [m for m in free if m not in ranked and any(f in m.lower() for f in families)
               and not any(x in m.lower() for x in ("-vl", "vision", "embed", "guard"))]
        fam.sort(key=lambda m: -free[m])
        others = fam + sorted((m for m in free if m not in ranked and m not in fam), key=lambda m: -free[m])
        if ranked or others:
            self.models = ranked + others[:8]
        log.info("Free models in rotation: %s", self.models[:6])

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1500) -> str:
        if not self.configured:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        last_err = None
        for model in self.models:
            for attempt in range(2):
                try:
                    r = self._client.post(
                        f"{API}/chat/completions",
                        json={
                            "model": model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                    )
                    if r.status_code == 429 or r.status_code >= 500:
                        last_err = f"{model}: HTTP {r.status_code}"
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    if r.status_code != 200:
                        last_err = f"{model}: HTTP {r.status_code} {r.text[:200]}"
                        break
                    data = r.json()
                    if "error" in data:
                        last_err = f"{model}: {data['error']}"
                        break
                    content = (data["choices"][0]["message"].get("content") or "").strip()
                    if not content:
                        last_err = f"{model}: empty reply"
                        break
                    self.last_model = model
                    return content
                except (httpx.HTTPError, KeyError, ValueError) as e:
                    last_err = f"{model}: {e}"
                    break
            log.warning("Model failed, trying next: %s", last_err)
        raise RuntimeError(f"All free models failed. Last error: {last_err}")

from __future__ import annotations

import json
from typing import Dict, List, Optional

from app.config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig, logger=None):
        self.config = config
        self.logger = logger
        self.backend = config.backend.lower()
        self._openai_client = None
        if self.backend == "openai":
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
                raise ModuleNotFoundError(
                    "openai package is required for OpenAI backend. Install with `pip install openai`."
                ) from exc
            self._openai_client = OpenAI(base_url=config.openai_base_url)

    def _call_openai(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        response = self._openai_client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def _call_ollama(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        try:
            import requests
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise ModuleNotFoundError(
                "requests is required for Ollama backend HTTP calls. Install with `pip install requests`."
            ) from exc

        payload = {
            "model": self.config.ollama_model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        url = f"{self.config.ollama_host.rstrip('/')}/api/chat"
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama request failed: {resp.status_code} {resp.text}")
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 220,
        temperature: float = 0.7,
    ) -> str:
        if self.logger:
            self.logger.debug("Calling %s backend with %d messages", self.backend, len(messages))

        if self.backend == "openai":
            return self._call_openai(messages, max_tokens=max_tokens, temperature=temperature)
        if self.backend == "ollama":
            return self._call_ollama(messages, max_tokens=max_tokens, temperature=temperature)
        raise ValueError(f"Unsupported LLM backend: {self.backend}")

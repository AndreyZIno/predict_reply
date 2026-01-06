from __future__ import annotations

from typing import Dict, Optional

from app.config import AppConfig
from app.generate.llm_client import LLMClient
from app.generate.prompt_builder import build_messages
from app.index.embedder import create_embedder
from app.index.retrieve import build_query_text, retrieve
from app.index.vector_store import create_vector_store


class Responder:
    def __init__(self, app_config: AppConfig, logger=None):
        self.config = app_config
        self.logger = logger
        self.embedder = create_embedder(app_config.index)
        self.store = create_vector_store(app_config.index.vector_backend, app_config.paths.index_dir, reset=False, logger=logger)
        self.llm = LLMClient(app_config.llm, logger=logger)

    def reply(self, prompt: str, recent_context: Optional[str] = None) -> Dict:
        query_text = build_query_text(prompt, recent_context)
        results = retrieve(query_text, self.embedder, self.store, self.config.retrieval)
        messages = build_messages(prompt, recent_context, results, self.config.generation)
        response = self.llm.generate(
            messages,
            max_tokens=self.config.generation.max_tokens,
            temperature=self.config.generation.temperature,
        )
        return {"reply": response, "retrieval": results}

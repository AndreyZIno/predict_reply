from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Optional

from app.config import RetrievalConfig
from app.index.embedder import Embedder
from app.index.vector_store import VectorStore


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _dedupe_results(results: List[Dict], threshold: float) -> List[Dict]:
    deduped: List[Dict] = []
    seen_texts: List[str] = []
    for res in results:
        text = res.get("metadata", {}).get("target_reply") or res.get("document") or ""
        is_dup = any(_similarity(text, seen) >= threshold for seen in seen_texts)
        if is_dup:
            continue
        deduped.append(res)
        seen_texts.append(text)
    return deduped


def build_query_text(prompt: str, recent_context: Optional[str]) -> str:
    if recent_context:
        return f"Recent conversation:\n{recent_context}\n\nPrompt:\n{prompt}"
    return prompt


def retrieve(
    query_text: str,
    embedder: Embedder,
    store: VectorStore,
    config: Optional[RetrievalConfig] = None,
) -> List[Dict]:
    cfg = config or RetrievalConfig()
    embedding = embedder.embed_texts([query_text])[0]
    results = store.query(embedding, top_k=cfg.top_k)
    return _dedupe_results(results, cfg.dedupe_threshold)

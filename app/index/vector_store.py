from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class VectorStore:
    def add(self, embeddings: List[List[float]], texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        raise NotImplementedError

    def query(self, embedding: List[float], top_k: int = 8) -> List[Dict[str, Any]]:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def add(self, embeddings: List[List[float]], texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        for emb, text, metadata, doc_id in zip(embeddings, texts, metadatas, ids):
            self._records.append({"id": doc_id, "embedding": emb, "document": text, "metadata": metadata})

    def _cosine(self, a: List[float], b: List[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        denom_a = math.sqrt(sum(x * x for x in a))
        denom_b = math.sqrt(sum(y * y for y in b))
        if not denom_a or not denom_b:
            return 0.0
        return num / (denom_a * denom_b)

    def query(self, embedding: List[float], top_k: int = 8) -> List[Dict[str, Any]]:
        scored = []
        for record in self._records:
            score = self._cosine(embedding, record["embedding"])
            scored.append({**record, "score": float(score)})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]


class ChromaVectorStore(VectorStore):
    def __init__(self, index_dir: Path, collection_name: str = "persona", reset: bool = False, logger=None):
        try:
            import chromadb
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise ModuleNotFoundError(
                "chromadb is required for Chroma vector store. Install with `pip install chromadb`."
            ) from exc

        self.logger = logger
        self.index_dir = index_dir
        self.collection_name = collection_name
        index_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(index_dir))
        if reset:
            try:
                self.client.delete_collection(name=collection_name)
            except Exception:
                pass
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, embeddings: List[List[float]], texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, embedding: List[float], top_k: int = 8) -> List[Dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "ids"],
        )
        scored = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        dists = result.get("distances", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        for doc_id, doc, dist, meta in zip(ids, docs, dists, metas):
            score = 1 / (1 + float(dist)) if dist is not None else 0.0
            scored.append({"id": doc_id, "document": doc, "metadata": meta, "score": score})
        return scored


def create_vector_store(backend: str, index_dir: Path, reset: bool = False, logger=None) -> VectorStore:
    backend_lower = backend.lower()
    if backend_lower == "chroma":
        return ChromaVectorStore(index_dir=index_dir, reset=reset, logger=logger)
    if backend_lower == "faiss":
        try:
            import faiss  # noqa: F401
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "faiss-cpu is required for the FAISS backend. Install with `pip install faiss-cpu`."
            )
        # A lightweight in-memory fallback; a full persistent FAISS implementation can be added later.
        if logger:
            logger.warning("FAISS backend selected, using in-memory stub. Install faiss-cpu for real index.")
        return InMemoryVectorStore()
    if backend_lower in {"memory", "in_memory"}:
        return InMemoryVectorStore()
    raise ValueError(f"Unsupported vector backend: {backend}")

from __future__ import annotations

from typing import Iterable, List, Optional

from app.config import IndexConfig


class Embedder:
    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    def __init__(self, config: IndexConfig):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise ModuleNotFoundError(
                "openai package is required for OpenAI embeddings. Install with `pip install openai`."
            ) from exc

        self.client = OpenAI()
        self.model = config.model_name
        self.batch_size = config.batch_size

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        inputs = list(texts)
        embeddings: List[List[float]] = []
        for i in range(0, len(inputs), self.batch_size):
            chunk = inputs[i : i + self.batch_size]
            response = self.client.embeddings.create(input=chunk, model=self.model)
            embeddings.extend([item.embedding for item in response.data])
        return embeddings


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, config: IndexConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise ModuleNotFoundError(
                "sentence-transformers is required for local embeddings. "
                "Install with `pip install sentence-transformers`."
            ) from exc

        self.model = SentenceTransformer(config.sentence_transformer_model)
        self.batch_size = config.batch_size

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        vectors = self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True)
        # encode may return a numpy array or list; normalize to list of lists
        return vectors.tolist() if hasattr(vectors, "tolist") else [list(v) for v in vectors]


def create_embedder(config: IndexConfig) -> Embedder:
    backend = config.embedding_backend.lower()
    if backend == "openai":
        return OpenAIEmbedder(config)
    if backend in {"local", "sentence_transformers"}:
        return SentenceTransformerEmbedder(config)
    raise ValueError(f"Unsupported embedding backend: {backend}")

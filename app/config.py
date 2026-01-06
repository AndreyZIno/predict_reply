from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dotenv = None


def maybe_load_dotenv() -> None:
    """Load environment variables from a .env file if python-dotenv is installed."""
    if load_dotenv:
        load_dotenv()


@dataclass
class PathsConfig:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_path: Path = field(default_factory=lambda: Path("data/processed/messages.jsonl"))
    index_dir: Path = field(default_factory=lambda: Path("data/index"))

    def resolve_all(self) -> None:
        self.raw_data_dir = (self.base_dir / self.raw_data_dir).resolve()
        self.processed_data_path = (self.base_dir / self.processed_data_path).resolve()
        self.index_dir = (self.base_dir / self.index_dir).resolve()


@dataclass
class RedactConfig:
    enabled: bool = False
    mask_email: bool = True
    mask_phone: bool = True
    mask_tokens: bool = True
    mask_ids: bool = True


@dataclass
class IngestConfig:
    exclude_channels: List[str] = field(default_factory=list)
    ignore_empty: bool = True
    min_length: int = 0
    drop_urls: bool = False
    drop_attachments_only: bool = True
    redact: RedactConfig = field(default_factory=RedactConfig)
    dry_run: bool = False


@dataclass
class IndexConfig:
    embedding_backend: str = "openai"  # openai | sentence_transformers
    vector_backend: str = "chroma"  # chroma | faiss
    model_name: str = "text-embedding-3-small"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32


@dataclass
class RetrievalConfig:
    top_k: int = 8
    dedupe_threshold: float = 0.95
    context_window: int = 10


@dataclass
class GenerationConfig:
    persona_name: Optional[str] = None
    length: str = "medium"  # short | medium | long
    tone: str = "casual"  # casual | neutral | professional
    emoji_level: str = "low"  # none | low | normal
    honesty: bool = True
    max_tokens: int = 220
    temperature: float = 0.7


@dataclass
class LLMConfig:
    backend: str = "openai"  # openai | ollama
    model: str = "gpt-4o-mini"
    openai_base_url: Optional[str] = None
    ollama_model: str = "llama3"
    max_retries: int = 3


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    debug: bool = False

    def prepare(self) -> None:
        maybe_load_dotenv()
        self.paths.resolve_all()
        self.paths.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.index_dir.mkdir(parents=True, exist_ok=True)
        self.paths.processed_data_path.parent.mkdir(parents=True, exist_ok=True)


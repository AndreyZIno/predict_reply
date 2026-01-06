from __future__ import annotations

import json
from pathlib import Path
from typing import List

from app.config import AppConfig
from app.index.embedder import create_embedder
from app.index.vector_store import create_vector_store
from app.ingest.document_builder import build_documents


def load_processed_messages(path: Path) -> List[dict]:
    messages: List[dict] = []
    with path.open() as f:
        for line in f:
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return messages


def build_vector_index(
    processed_path: Path,
    app_config: AppConfig,
    my_author_id: str,
    reset: bool = False,
    logger=None,
) -> int:
    if not my_author_id:
        raise ValueError("my_author_id is required to build conversation-pair documents.")

    messages = load_processed_messages(processed_path)
    docs = build_documents(messages, my_author_id=my_author_id, retrieval_config=app_config.retrieval)
    if logger:
        logger.info("Loaded %d messages and built %d docs", len(messages), len(docs))

    embedder = create_embedder(app_config.index)
    store = create_vector_store(app_config.index.vector_backend, app_config.paths.index_dir, reset=reset, logger=logger)

    batch_size = app_config.index.batch_size
    for i in range(0, len(docs), batch_size):
        chunk = docs[i : i + batch_size]
        texts = [doc["text"] for doc in chunk]
        metadatas = [doc["metadata"] for doc in chunk]
        ids = [doc["id"] for doc in chunk]
        embeddings = embedder.embed_texts(texts)
        store.add(embeddings=embeddings, texts=texts, metadatas=metadatas, ids=ids)
        if logger:
            logger.debug("Indexed batch %s-%s", i, min(i + batch_size, len(docs)))

    if logger:
        logger.info("Index build complete.")
    return len(docs)

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from app.config import IngestConfig
from app.utils.redact import redact_text

URL_RE = re.compile(r"https?://\\S+")
EMOJI_OR_SYMBOL_RE = re.compile(r"^[\\W_]+$", re.UNICODE)


def _clean_content(content: str, config: IngestConfig) -> str:
    cleaned = content.replace("\\r", " ").strip()
    if config.drop_urls:
        cleaned = URL_RE.sub("", cleaned).strip()
    cleaned = re.sub(r"\\s+", " ", cleaned).strip()
    return cleaned


def _is_meaningful(text: str, config: IngestConfig) -> bool:
    if not text:
        return False
    if config.min_length and len(text) < config.min_length:
        return False
    # Detect emoji/only-symbol spam
    if not any(ch.isalnum() for ch in text) and EMOJI_OR_SYMBOL_RE.match(text):
        return False
    return True


def normalize_messages(messages: Iterable[Dict], config: IngestConfig) -> List[Dict]:
    normalized: List[Dict] = []
    for msg in messages:
        content = _clean_content(msg.get("content", ""), config)
        attachments = msg.get("attachments") or []

        if config.ignore_empty and not content:
            if config.drop_attachments_only and not msg.get("content"):
                continue
        if not _is_meaningful(content, config):
            continue

        msg_copy = dict(msg)
        msg_copy["content"] = redact_text(content, config.redact)
        msg_copy["attachments"] = attachments
        normalized.append(msg_copy)

    normalized.sort(key=lambda m: m.get("timestamp") or "")
    return normalized


def save_messages_jsonl(messages: Iterable[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for message in messages:
            f.write(json.dumps(message, ensure_ascii=True) + "\n")

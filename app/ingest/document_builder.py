from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from app.config import RetrievalConfig


def _join_context(lines: List[str]) -> str:
    return "\n".join(lines).strip()


def build_documents(
    messages: List[Dict],
    my_author_id: str,
    retrieval_config: Optional[RetrievalConfig] = None,
) -> List[Dict]:
    config = retrieval_config or RetrievalConfig()
    docs: List[Dict] = []
    window = config.context_window

    for idx, message in enumerate(messages):
        if message.get("author_id") != my_author_id:
            continue

        msg_id = message.get("id") or f"msg-{idx}"
        target_reply = message.get("content", "").strip()
        if not target_reply:
            continue

        # Single-message doc
        docs.append(
            {
                "id": f"single-{msg_id}",
                "text": target_reply,
                "metadata": {
                    "doc_type": "single",
                    "source_message_id": msg_id,
                    "channel_id": message.get("channel_id"),
                    "channel_name": message.get("channel_name"),
                    "timestamp": message.get("timestamp"),
                    "target_reply": target_reply,
                },
            }
        )

        start_idx = max(0, idx - window)
        context_messages = messages[start_idx:idx]
        context_lines = []
        for ctx in context_messages:
            ctx_content = ctx.get("content", "").strip()
            if not ctx_content:
                continue
            author = ctx.get("author_name") or "unknown"
            context_lines.append(f"{author}: {ctx_content}")

        context_text = _join_context(context_lines)
        if not context_text:
            continue

        docs.append(
            {
                "id": f"pair-{msg_id}",
                "text": context_text + "\n\n[My reply follows above]",
                "metadata": {
                    "doc_type": "conversation_pair",
                    "source_message_id": msg_id,
                    "channel_id": message.get("channel_id"),
                    "channel_name": message.get("channel_name"),
                    "timestamp": message.get("timestamp"),
                    "context_text": context_text,
                    "target_reply": target_reply,
                },
            }
        )

    return docs

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _parse_timestamp(raw_ts: Any) -> Optional[str]:
    if not raw_ts:
        return None
    if isinstance(raw_ts, (int, float)):
        try:
            ts = float(raw_ts)
            if ts > 10**11:  # handle ms epoch stamps
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts).isoformat()
        except Exception:
            return None
    try:
        # Handles ISO strings and common Discord export formats
        return datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00")).isoformat()
    except Exception:
        return str(raw_ts)


def _safe_text(record: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def _list_field(record: Dict[str, Any], *keys: str) -> List[Any]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, list):
            return value
    return []


def _parse_author(record: Dict[str, Any]) -> Dict[str, str]:
    author = record.get("author") or record.get("Author") or {}
    if not isinstance(author, dict):
        return {"id": "", "name": str(author)}
    discriminator = author.get("discriminator") or author.get("Discriminator")
    name = author.get("username") or author.get("name") or author.get("Name") or ""
    if discriminator and discriminator != "0":
        name = f"{name}#{discriminator}"
    return {
        "id": str(author.get("id") or author.get("ID") or ""),
        "name": name,
    }


def _normalize_record(record: Dict[str, Any], channel_id: str, channel_name: str) -> Dict[str, Any]:
    author = _parse_author(record)
    content = _safe_text(record, "content", "Content", "Contents", "Message")
    mentions = _list_field(record, "mentions", "Mentions")
    attachments = _list_field(record, "attachments", "Attachments")
    reply_to = record.get("reference") or record.get("reply_to") or record.get("ReplyTo")
    reply_to_message_id = None
    if isinstance(reply_to, dict):
        reply_to_message_id = reply_to.get("message_id") or reply_to.get("id") or reply_to.get("MessageID")
    elif isinstance(reply_to, str):
        reply_to_message_id = reply_to

    resolved_channel_id = str(record.get("channel_id") or record.get("channelId") or channel_id)
    resolved_channel_name = (
        record.get("channel_name")
        or record.get("channelName")
        or record.get("userName")
        or channel_name
    )

    return {
        "id": str(record.get("id") or record.get("ID") or record.get("MessageID") or ""),
        "channel_id": resolved_channel_id,
        "channel_name": str(resolved_channel_name),
        "author_id": author.get("id", ""),
        "author_name": author.get("name", ""),
        "timestamp": _parse_timestamp(record.get("timestamp") or record.get("Timestamp")),
        "content": content or "",
        "mentions": mentions,
        "attachments": attachments,
        "reply_to_message_id": reply_to_message_id,
    }


def _iter_json_payloads(source: Path) -> Iterable[Dict[str, Any]]:
    if source.suffix.lower() == ".jsonl":
        with source.open() as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return

    with source.open() as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        if "messages" in payload:
            for msg in payload.get("messages", []):
                yield msg
        else:
            for value in payload.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def _iter_files_from_zip(zip_path: Path) -> Iterable[tuple[str, bytes]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            lower_name = name.lower()
            if not lower_name.endswith(".json"):
                continue
            if "messages" not in lower_name and "_page_" not in lower_name:
                continue
            with zf.open(name) as raw:
                yield name, raw.read()


def _iter_message_records(file: Path, logger=None) -> Iterable[Dict[str, Any]]:
    has_message = False
    for record in _iter_json_payloads(file):
        if not isinstance(record, dict):
            continue
        if not any(k in record for k in ("content", "Content", "Message", "timestamp", "Timestamp")):
            continue
        has_message = True
        yield record
    if logger and not has_message:
        logger.debug("Skipping %s (no message-like records)", file)


def load_messages_from_export(
    path: Path, exclude_channels: Optional[List[str]] = None, logger=None
) -> List[Dict[str, Any]]:
    exclude = {c.lower() for c in (exclude_channels or [])}
    messages: List[Dict[str, Any]] = []

    def should_skip(channel_name: str, channel_id: str) -> bool:
        if not exclude:
            return False
        return channel_name.lower() in exclude or channel_id.lower() in exclude

    if path.suffix.lower() == ".zip":
        for name, data in _iter_files_from_zip(path):
            channel_name = Path(name).stem
            channel_id = channel_name
            if should_skip(channel_name, channel_id):
                continue
            try:
                payload = json.load(io.TextIOWrapper(io.BytesIO(data), encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - best effort parsing
                if logger:
                    logger.warning("Skipping %s: %s", name, exc)
                continue

            file_messages = payload.get("messages") if isinstance(payload, dict) else None
            if not file_messages and isinstance(payload, list):
                file_messages = payload
            if not file_messages:
                continue

            for record in file_messages:
                if not isinstance(record, dict):
                    continue
                messages.append(_normalize_record(record, channel_id, channel_name))
        return messages

    if path.is_dir():
        for file in path.rglob("*.json"):
            if file.name.lower() == "index.json":
                continue
            channel_hint = file.parent.name if file.parent != path else file.stem
            channel_name = channel_hint
            channel_id = channel_hint
            if should_skip(channel_name, channel_id):
                continue
            for record in _iter_message_records(file, logger=logger):
                messages.append(_normalize_record(record, channel_id, channel_name))
        for file in path.rglob("*.jsonl"):
            channel_name = file.stem
            channel_id = channel_name
            if should_skip(channel_name, channel_id):
                continue
            for record in _iter_json_payloads(file):
                messages.append(_normalize_record(record, channel_id, channel_name))
        return messages

    # Fallback for single JSON/JSONL file
    if path.suffix.lower() in {".json", ".jsonl"}:
        channel_name = path.stem
        channel_id = channel_name
        if not should_skip(channel_name, channel_id):
            for record in _iter_json_payloads(path):
                messages.append(_normalize_record(record, channel_id, channel_name))
        return messages

    raise FileNotFoundError(f"Unsupported input: {path}")

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from app.config import AppConfig
from app.generate.responder import Responder
from app.index.build_index import build_vector_index
from app.index.retrieve import build_query_text, retrieve
from app.index.vector_store import create_vector_store
from app.index.embedder import create_embedder
from app.ingest.discord_export_parser import load_messages_from_export
from app.ingest.normalize import normalize_messages, save_messages_jsonl
from app.utils.logging import setup_logging


def _read_context(context: Optional[str], context_file: Optional[str]) -> Optional[str]:
    if context_file:
        return Path(context_file).read_text()
    return context


def _build_app_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig()
    config.debug = args.debug
    config.index.embedding_backend = args.embedding_backend or config.index.embedding_backend
    config.index.vector_backend = args.vector_backend or config.index.vector_backend
    if args.embedding_model:
        config.index.model_name = args.embedding_model
        config.index.sentence_transformer_model = args.embedding_model
    config.retrieval.top_k = args.top_k or config.retrieval.top_k
    if args.persona_name:
        config.generation.persona_name = args.persona_name
    config.generation.length = args.length or config.generation.length
    config.generation.tone = args.tone or config.generation.tone
    config.generation.emoji_level = args.emoji_level or config.generation.emoji_level
    config.generation.honesty = args.honesty
    config.generation.max_tokens = args.max_tokens or config.generation.max_tokens
    config.paths.base_dir = Path(args.root).resolve() if args.root else config.paths.base_dir
    config.prepare()
    return config


def cmd_ingest(args: argparse.Namespace) -> None:
    config = _build_app_config(args)
    config.ingest.exclude_channels = args.exclude_channels or []
    config.ingest.drop_urls = args.drop_urls
    config.ingest.min_length = args.min_length
    config.ingest.dry_run = args.dry_run
    config.ingest.redact.enabled = args.redact

    logger = setup_logging(config.debug, "ingest")
    input_path = Path(args.input).expanduser().resolve()

    logger.info("Loading messages from %s", input_path)
    messages = load_messages_from_export(input_path, exclude_channels=config.ingest.exclude_channels, logger=logger)
    logger.info("Parsed %d raw messages", len(messages))

    normalized = normalize_messages(messages, config.ingest)
    logger.info("Normalized down to %d messages", len(normalized))

    if config.ingest.dry_run:
        logger.info("Dry run complete. No data written.")
        return

    save_messages_jsonl(normalized, config.paths.processed_data_path)
    logger.info("Saved processed messages to %s", config.paths.processed_data_path)

    if args.skip_index:
        logger.info("Skipping index build.")
        return

    build_vector_index(
        processed_path=config.paths.processed_data_path,
        app_config=config,
        my_author_id=args.self_id,
        reset=args.reset_index,
        logger=logger,
    )


def cmd_search(args: argparse.Namespace) -> None:
    config = _build_app_config(args)
    logger = setup_logging(config.debug, "search")

    recent_context = _read_context(args.context, args.context_file)
    query_text = build_query_text(args.query, recent_context)

    embedder = create_embedder(config.index)
    store = create_vector_store(config.index.vector_backend, config.paths.index_dir, reset=False, logger=logger)

    results = retrieve(query_text, embedder, store, config.retrieval)
    if not results:
        print("No results found. Did you build the index?")
        return

    for idx, res in enumerate(results, start=1):
        meta = res.get("metadata", {})
        context_text = meta.get("context_text") or res.get("document") or ""
        target_reply = meta.get("target_reply") or res.get("document") or ""
        print(f"[{idx}] score={res.get('score'):.3f} channel={meta.get('channel_name')} ts={meta.get('timestamp')}")
        print("Context:")
        print(context_text[:500])
        print("Reply:")
        print(target_reply[:300])
        print("-" * 40)


def cmd_reply(args: argparse.Namespace) -> None:
    config = _build_app_config(args)
    logger = setup_logging(config.debug, "reply")

    recent_context = _read_context(args.context, args.context_file)
    responder = Responder(config, logger=logger)
    result = responder.reply(args.prompt, recent_context=recent_context)
    print(result["reply"])

    if args.show_retrieval:
        print("\n--- Retrieved examples ---")
        for idx, res in enumerate(result["retrieval"], start=1):
            meta = res.get("metadata", {})
            print(f"[{idx}] score={res.get('score'):.3f} channel={meta.get('channel_name')} ts={meta.get('timestamp')}")
            print("Context:")
            print((meta.get("context_text") or res.get("document") or "")[:400])
            print("Reply:")
            print((meta.get("target_reply") or res.get("document") or "")[:200])
            print("-" * 30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discord Persona Responder (RAG)")
    parser.add_argument("--root", help="Project root (defaults to repository root)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--embedding-backend", help="Embedding backend (openai or sentence_transformers)")
    parser.add_argument("--embedding-model", help="Embedding model name")
    parser.add_argument("--vector-backend", help="Vector store backend (chroma or faiss)")
    parser.add_argument("--persona-name", help="Optional persona name for prompt")
    parser.add_argument("--top-k", type=int, help="Override retrieval top_k")
    parser.add_argument("--length", choices=["short", "medium", "long"], help="Generation length")
    parser.add_argument("--tone", choices=["casual", "neutral", "professional"], help="Generation tone")
    parser.add_argument("--emoji-level", choices=["none", "low", "normal"], help="Emoji usage level")
    honesty_group = parser.add_mutually_exclusive_group()
    honesty_group.add_argument("--honesty", dest="honesty", action="store_true", default=True, help="Ask clarifying questions if unsure")
    honesty_group.add_argument("--no-honesty", dest="honesty", action="store_false", help="Disable honesty guardrail")
    parser.add_argument("--max-tokens", type=int, help="Max tokens for generation")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Parse export/JSONL and build the index")
    ingest.add_argument("input", help="Path to Discord export (zip/folder) or normalized JSON/JSONL")
    ingest.add_argument("--self-id", required=True, help="Author ID representing you")
    ingest.add_argument("--exclude-channels", nargs="*", help="Channel IDs or names to skip")
    ingest.add_argument("--drop-urls", action="store_true", help="Remove URLs from messages")
    ingest.add_argument("--min-length", type=int, default=0, help="Drop very short messages under this length")
    ingest.add_argument("--redact", action="store_true", help="Enable basic PII redaction")
    ingest.add_argument("--dry-run", action="store_true", help="Run parsing without writing files or building index")
    ingest.add_argument("--skip-index", action="store_true", help="Do not build the vector index after ingest")
    ingest.add_argument("--reset-index", action="store_true", help="Reset the vector store before indexing")

    search = subparsers.add_parser("search", help="Search the vector index for relevant examples")
    search.add_argument("--query", required=True, help="Query text / prompt")
    search.add_argument("--context", help="Recent context text")
    search.add_argument("--context-file", help="Path to a text file containing recent context")

    reply = subparsers.add_parser("reply", help="Generate a reply in your voice")
    reply.add_argument("--prompt", required=True, help="Prompt to respond to")
    reply.add_argument("--context", help="Recent context text")
    reply.add_argument("--context-file", help="Path to a text file containing recent context")
    reply.add_argument("--show-retrieval", action="store_true", help="Print retrieved examples for debugging")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "reply":
        cmd_reply(args)
    else:  # pragma: no cover
        parser.print_help()


if __name__ == "__main__":
    main()

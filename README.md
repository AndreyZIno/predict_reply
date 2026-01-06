# Discord Persona Responder (RAG)

Local-first CLI that ingests your Discord history, builds a retrieval index, and generates replies that sound like you.

## Features
- Parse Discord exports (zip/folder) or normalized JSON/JSONL.
- Normalize and redact messages; exclude channels and attachments-only noise.
- Build single-message and conversation-pair documents for better retrieval.
- Embed with OpenAI or local sentence-transformers; store in Chroma (default) or FAISS.
- Generate replies with OpenAI chat or local Ollama/llama.cpp (via HTTP).
- Debug search to inspect retrieved contexts.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in OPENAI_API_KEY or point to your local model
```

## CLI
All commands run from the repository root:
- Ingest + index: `python -m app.main ingest <export.zip|folder|messages.jsonl> --self-id <your_discord_user_id>`
  - Useful flags: `--exclude-channels general random`, `--drop-urls`, `--min-length 3`, `--redact`, `--skip-index`, `--reset-index`, `--embedding-backend sentence_transformers`, `--embedding-model all-MiniLM-L6-v2`, `--vector-backend chroma|faiss`.
- Search retrieved examples: `python -m app.main search --query "talk about GPUs" [--context-file recent.txt]`
- Generate a reply: `python -m app.main reply --prompt "Sure, let's meet" [--context-file thread.txt] [--show-retrieval]`

Global options: `--debug`, `--persona-name`, `--top-k`, `--length short|medium|long`, `--tone casual|neutral|professional`, `--emoji-level none|low|normal`, `--no-honesty` (to disable clarifying-question behavior).

## Data expectations
- Normalized JSONL (what the pipeline writes) uses one message per line with fields: `id`, `channel_id`, `channel_name`, `author_id`, `author_name`, `timestamp`, `content`, `mentions`, `reply_to_message_id`, `attachments`.
- Official Discord exports: point `ingest` at the zip or extracted folder; it will scan `messages/*.json`. If the format differs, use the JSONL shape above.
- Direct Messages export folders like `Direct Messages/<user>_xyz/<user>_page_1.json` (flat lists with `channel_id`, `userName`, etc.) are now handled automatically; just point `ingest` at the `Direct Messages` folder.

## Indexing strategy
- Builds two doc types for your messages (author == `--self-id`):
  - `single`: just your message text.
  - `conversation_pair`: last N messages (default 10) as context_text + your reply as target_reply.
- Retrieval defaults: top_k=8, dedupe near-duplicates (SequenceMatcher), cosine similarity in-memory fallback if Chroma/FAISS missing.

## Safety and privacy
- `--redact` masks emails, phones, token-like strings, and long numeric IDs.
- Prompts instruct the model not to quote sensitive info verbatim and to paraphrase.
- No prompts or data are logged unless you enable `--debug`.

## Paths
- Raw inputs: `data/raw/` (you can store exports here).
- Processed messages: `data/processed/messages.jsonl`.
- Vector store persistence: `data/index/` (Chroma).

## Next steps / eval ideas
- Add a small eval script that hides your reply and checks model similarity.
- Hook up a lightweight web UI for local use.
- Extend FAISS backend to persistent on-disk storage if needed.

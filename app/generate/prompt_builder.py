from __future__ import annotations

from typing import Dict, List, Optional

from app.config import GenerationConfig


def _length_guidance(length: str) -> str:
    mapping = {
        "short": "Keep replies under 20 words; 1-2 short sentences max.",
        "medium": "Keep replies under 80 words.",
        "long": "You may elaborate up to 200 words.",
    }
    return mapping.get(length, "")


def _tone_guidance(tone: str) -> str:
    mapping = {
        "casual": "Use a relaxed, conversational voice with chat slang when natural.",
        "neutral": "Keep it concise and direct; avoid filler or polite fluff.",
        "professional": "Sound concise, confident, and professional.",
    }
    return mapping.get(tone, "")


def _emoji_guidance(level: str) -> str:
    mapping = {
        "none": "Do not use emojis.",
        "low": "Use emojis sparingly, only if natural.",
        "normal": "Use emojis as you typically would.",
    }
    return mapping.get(level, "")


def build_examples_block(examples: List[Dict]) -> str:
    if not examples:
        return "No prior examples provided."

    rendered = []
    for idx, example in enumerate(examples, start=1):
        metadata = example.get("metadata", {})
        context = metadata.get("context_text") or example.get("document") or ""
        reply = metadata.get("target_reply") or example.get("document") or ""
        rendered.append(
            f"Example {idx}\nContext:\n{context}\nMy reply:\n{reply}"
        )
    return "\n\n".join(rendered)


def build_system_prompt(config: GenerationConfig) -> str:
    parts = [
        "You are an assistant that writes responses exactly as the user would.",
        "Mimic their tone, phrasing, and brevity. Avoid quoting private info verbatim.",
        "Default to concise, neutral replies; avoid exclamation marks and avoid emojis unless the user used them.",
        "Keep DM-style slang and abbreviations from the examples; lower-case is fine.",
        "Always answer the prompt directly; do not change the subject or repeat the prompt.",
        "If asked to do something, state whether you'll do it and when, in one short sentence.",
        "Do not repeat the prompt or context verbatim; produce a fresh reply in the user's voice.",
        "If unsure, ask a short follow-up instead of guessing.",
        "Do not output personal data from memory or examples; paraphrase or omit.",
        _length_guidance(config.length),
        _tone_guidance(config.tone),
        _emoji_guidance(config.emoji_level),
    ]
    if config.persona_name:
        parts.append(f"Persona name: {config.persona_name}")
    if config.honesty:
        parts.append("If context is insufficient, ask a clarifying question.")
    return " ".join([p for p in parts if p])


def build_messages(
    prompt: str,
    recent_context: Optional[str],
    examples: List[Dict],
    config: GenerationConfig,
) -> List[Dict[str, str]]:
    system_prompt = build_system_prompt(config)
    examples_block = build_examples_block(examples)
    current_block = f"Current context:\n{recent_context}\n\nPrompt:\n{prompt}" if recent_context else f"Prompt:\n{prompt}"

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "You are replying exactly as the user would. Do not repeat the prompt or context; write a new reply in their style.\n"
                "Answer the ask directly (yes/no/when/etc.), stay brief, and keep any slang/abbreviations from the examples.\n"
                "Return exactly one short sentence (<20 words) that answers the ask. Do not include anything else.\n"
                f"Here are examples of how I reply:\n{examples_block}\n\n"
                "Now respond to the current prompt.\n"
                f"{current_block}\n"
                "Return only one best reply, nothing else."
            ),
        },
    ]

from __future__ import annotations

from typing import Dict, List, Optional

from app.config import GenerationConfig


def _length_guidance(length: str) -> str:
    mapping = {
        "short": "Keep replies under 80 words.",
        "medium": "Keep replies under 160 words.",
        "long": "You may elaborate up to 220 words.",
    }
    return mapping.get(length, "")


def _tone_guidance(tone: str) -> str:
    mapping = {
        "casual": "Use a relaxed, conversational voice.",
        "neutral": "Keep the tone even and factual.",
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
            "content": f"Here are examples of how I reply:\n{examples_block}\n\nNow respond to the current prompt.\n{current_block}\nReturn only one best reply.",
        },
    ]

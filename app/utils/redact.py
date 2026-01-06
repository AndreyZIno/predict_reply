import re
from typing import Optional

from app.config import RedactConfig

EMAIL_RE = re.compile(r"[\\w.+-]+@[\\w-]+\\.[\\w.-]+")
PHONE_RE = re.compile(r"(?:\\+?\\d[\\s-]?){7,15}")
TOKEN_RE = re.compile(r"(?i)(?:api|secret|token|key)[=:]\\s*[A-Za-z0-9-_]{10,}")
ID_RE = re.compile(r"\\b\\d{15,}\\b")


def _mask(match: re.Match, label: str) -> str:
    return f"[{label}_redacted]"


def redact_text(text: str, config: Optional[RedactConfig]) -> str:
    if not config or not config.enabled:
        return text

    redacted = text
    if config.mask_email:
        redacted = EMAIL_RE.sub(lambda m: _mask(m, "email"), redacted)
    if config.mask_phone:
        redacted = PHONE_RE.sub(lambda m: _mask(m, "phone"), redacted)
    if config.mask_tokens:
        redacted = TOKEN_RE.sub(lambda m: _mask(m, "token"), redacted)
    if config.mask_ids:
        redacted = ID_RE.sub(lambda m: _mask(m, "id"), redacted)
    return redacted

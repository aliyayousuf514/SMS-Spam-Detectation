
import re


def clean_text(text: str) -> str:
    """Normalize an SMS while preserving useful security clues."""

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text

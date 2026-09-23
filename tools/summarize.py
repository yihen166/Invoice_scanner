"""Final pipeline step -- turn structured data + validation result into a
short human-readable summary, via Gemini."""
from __future__ import annotations

from api.models import ExtractedInvoice, ValidationResult
from vision.gemini_client import generate_summary as _generate_summary


def summarize(data: ExtractedInvoice, validation: ValidationResult) -> tuple[str, int]:
    result = _generate_summary(
        data.model_dump(mode="json"), validation.model_dump(mode="json")
    )
    return result.text.strip(), result.tokens_used

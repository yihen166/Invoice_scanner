"""Part 3 -- LLM Structured Extraction.

Calls Gemini to turn raw text into JSON, then validates it against
`ExtractedInvoice` (Pydantic). If the model's JSON is malformed or doesn't
fit the schema, we retry once with the parse error fed back to the model --
a small guardrail against one-off formatting slips, without looping forever.
"""
from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from api.models import ExtractedInvoice
from vision.gemini_client import extract_structured_data, GeminiCallResult

logger = logging.getLogger("agent.llm_extract")

SCHEMA_HINT = """{
  "document_type": "invoice" | "receipt" | "unknown",
  "vendor": string | null,
  "invoice_number": string | null,
  "invoice_date": "YYYY-MM-DD" | null,
  "due_date": "YYYY-MM-DD" | null,
  "total_amount": number | null,
  "currency": string | null,
  "line_items": [
    {"description": string, "quantity": number | null, "unit_price": number | null, "amount": number | null}
  ]
}"""


class ExtractionFailedError(RuntimeError):
    pass


def _parse(raw: str) -> ExtractedInvoice:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(cleaned)
    return ExtractedInvoice.model_validate(data)


def llm_extract(raw_text: str) -> tuple[ExtractedInvoice, int]:
    """Returns (structured_data, tokens_used). Raises ExtractionFailedError on repeated failure."""
    tokens_used = 0
    last_error: Exception | None = None

    for attempt in range(2):
        result: GeminiCallResult = extract_structured_data(raw_text, SCHEMA_HINT)
        tokens_used += result.tokens_used
        try:
            return _parse(result.text), tokens_used
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning("LLM output failed schema validation (attempt %d): %s", attempt + 1, exc)
            # Feed the failure back in on the retry by nudging the input text.
            raw_text = (
                f"{raw_text}\n\n[Your previous JSON response was invalid: {exc}. "
                "Return corrected JSON only.]"
            )

    raise ExtractionFailedError(f"LLM failed to produce valid structured data: {last_error}")

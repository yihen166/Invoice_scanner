"""Wrapper around the Gemini API.

One provider covers three roles in this pipeline:
  - VLM: read scanned/image documents directly (Part 2 fallback)
  - LLM: convert raw text into structured JSON, and write the summary (Part 3 & 4)
  - Embeddings: power the RAG-style vendor-history lookup (tools/vendor_history.py)

Centralizing the SDK calls here means the rest of the codebase never touches
`google.genai` directly, and retry/logging/token-counting live in one place.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

logger = logging.getLogger("agent.gemini")

MODEL_TEXT = "gemini-3.6-flash"
MODEL_VISION = "gemini-3.6-flash"  # same model, multimodal
EMBEDDING_MODEL = "gemini-embedding-001"


@dataclass
class GeminiCallResult:
    text: str
    tokens_used: int
    latency_ms: int


class GeminiUnavailableError(RuntimeError):
    """Raised when the Gemini API cannot be reached or returns no usable output."""


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiUnavailableError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def _is_retryable(exc: BaseException) -> bool:
    # Retry on anything that isn't our own "missing key" config error.
    return not isinstance(exc, GeminiUnavailableError)


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception(_is_retryable),
)
def _call(contents, config: Optional[types.GenerateContentConfig] = None, model: str = MODEL_TEXT) -> GeminiCallResult:
    start = time.perf_counter()
    client = _client()
    try:
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )
    except GeminiUnavailableError:
        raise
    except Exception as exc:  # network / API errors -> retried by tenacity
        if "UNAVAILABLE" in str(exc) or "503" in str(exc):
            logger.warning("Gemini is overloaded (503), will retry if attempts remain: %s", exc)
        else:
            logger.warning("Gemini call failed, will retry if attempts remain: %s", exc)
        raise
    latency_ms = int((time.perf_counter() - start) * 1000)
    text = response.text or ""
    tokens = 0
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        tokens = getattr(usage, "total_token_count", 0) or 0
    return GeminiCallResult(text=text, tokens_used=tokens, latency_ms=latency_ms)


def ocr_document(file_bytes: bytes, mime_type: str) -> GeminiCallResult:
    """Use Gemini's vision capability to read text out of an image or scanned PDF page."""
    part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    prompt = (
        "Transcribe all readable text from this document image exactly as it "
        "appears, preserving line breaks and table-like structure. Do not "
        "summarize or interpret -- output raw text only."
    )
    return _call(contents=[part, prompt], model=MODEL_VISION)


def extract_structured_data(raw_text: str, json_schema_hint: str) -> GeminiCallResult:
    """Ask Gemini to convert raw document text into the ExtractedInvoice JSON shape."""
    prompt = f"""You are a financial document data-extraction engine.

Extract invoice/receipt information from the document text below and return
ONLY a single JSON object -- no markdown fences, no commentary -- matching
exactly this shape:

{json_schema_hint}

Rules:
- If a field is not present in the text, use null (or an empty list for line_items).
- total_amount and line item amounts must be plain numbers, not strings.
- invoice_date and due_date must be "YYYY-MM-DD" if determinable, else null.
- document_type must be one of: "invoice", "receipt", "unknown".

Document text:
---
{raw_text}
---
"""
    config = types.GenerateContentConfig(response_mime_type="application/json")
    return _call(contents=[prompt], config=config)


def generate_summary(structured_json: dict, validation_json: dict) -> GeminiCallResult:
    prompt = f"""Write a short (2-4 sentence) plain-English summary of this financial
document for a busy reviewer. Mention vendor, amount, currency, and due date
if known. If validation flagged any issues, mention them plainly at the end.

Structured data:
{json.dumps(structured_json, default=str)}

Validation result:
{json.dumps(validation_json, default=str)}
"""
    return _call(contents=[prompt])


def quick_prompt(prompt: str) -> GeminiCallResult:
    """General-purpose single-prompt call, for ad-hoc tools that don't need
    a dedicated function of their own."""
    return _call(contents=[prompt])


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception(_is_retryable),
)
def embed_text(text: str) -> list[float]:
    """Embed text for the RAG-style vendor-history vector store
    (tools/vendor_history.py). Truncated defensively -- the model accepts
    a large input, but invoice text is short and this keeps latency/cost
    predictable.
    """
    client = _client()
    try:
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text[:8000])
    except GeminiUnavailableError:
        raise
    except Exception as exc:
        logger.warning("Gemini embedding call failed, will retry if attempts remain: %s", exc)
        raise
    if not response.embeddings:
        raise GeminiUnavailableError("Embedding response contained no vectors")
    return list(response.embeddings[0].values)

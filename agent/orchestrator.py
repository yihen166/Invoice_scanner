"""Part 4 -- Simple AI Agent Workflow.

`DocumentAgent.run()` is the whole pipeline:

    extract_text -> llm_extract -> validate_invoice -> summarize

It's a plain function-calling sequence rather than a framework-driven
"reasoning" agent, because the workflow is fixed and known ahead of time --
there's nothing for a general-purpose planner to decide. What makes it an
*agent* rather than a script is that each step is an independent, testable
tool the orchestrator calls and inspects the result of (e.g. it always
calls validate_invoice before summarize, per the spec), and it owns
error handling / observability across the whole run.

After a successful run, it also indexes the invoice into the RAG-style
vendor-history store (tools/vendor_history.py) so future uploads from the
same vendor can be compared against it -- this is deliberately best-effort
and non-fatal; a failed embedding should never fail the document's own
pipeline run.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from api.models import ExtractedInvoice, ValidationResult
from tools.extract_text import extract_text
from tools.llm_extract import llm_extract
from tools.validate import validate_invoice
from tools.summarize import summarize
from tools.vendor_history import store_invoice_embedding

logger = logging.getLogger("agent.orchestrator")


@dataclass
class PipelineResult:
    raw_text: str
    structured_data: ExtractedInvoice
    validation: ValidationResult
    summary: str
    tokens_used: int
    processing_time_ms: int
    step_timings_ms: dict = field(default_factory=dict)


class DocumentAgent:
    """Orchestrates the document-intelligence pipeline for a single file."""

    def run(self, document_id: str, file_path: str) -> PipelineResult:
        start = time.perf_counter()
        timings: dict[str, int] = {}
        total_tokens = 0

        step_start = time.perf_counter()
        extraction = extract_text(file_path)
        timings["extract_text"] = int((time.perf_counter() - step_start) * 1000)
        total_tokens += extraction.tokens_used
        logger.info(
            "document_id=%s step=extract_text method=%s chars=%d",
            document_id, extraction.method, len(extraction.raw_text),
        )

        step_start = time.perf_counter()
        structured_data, tokens = llm_extract(extraction.raw_text)
        timings["llm_extract"] = int((time.perf_counter() - step_start) * 1000)
        total_tokens += tokens
        logger.info("document_id=%s step=llm_extract tokens=%d", document_id, tokens)

        step_start = time.perf_counter()
        validation = validate_invoice(structured_data)
        timings["validate_invoice"] = int((time.perf_counter() - step_start) * 1000)
        logger.info(
            "document_id=%s step=validate_invoice is_valid=%s issues=%d",
            document_id, validation.is_valid, len(validation.issues),
        )

        step_start = time.perf_counter()
        summary_text, tokens = summarize(structured_data, validation)
        timings["summarize"] = int((time.perf_counter() - step_start) * 1000)
        total_tokens += tokens

        total_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "document_id=%s pipeline_complete processing_time_ms=%d llm_tokens_used=%d",
            document_id, total_ms, total_tokens,
        )

        self._index_for_vendor_history(document_id, structured_data, extraction.raw_text)

        return PipelineResult(
            raw_text=extraction.raw_text,
            structured_data=structured_data,
            validation=validation,
            summary=summary_text,
            tokens_used=total_tokens,
            processing_time_ms=total_ms,
            step_timings_ms=timings,
        )

    def _index_for_vendor_history(self, document_id: str, data: ExtractedInvoice, raw_text: str) -> None:
        try:
            store_invoice_embedding(
                document_id=document_id,
                vendor=data.vendor,
                raw_text=raw_text,
                total_amount=data.total_amount,
                currency=data.currency,
                invoice_date=data.invoice_date,
            )
        except Exception as exc:  # noqa: BLE001 -- RAG indexing must never fail the pipeline
            logger.warning("document_id=%s vendor_history_indexing_failed error=%s", document_id, exc)

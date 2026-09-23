"""Part 1 -- Document Upload API.

POST /documents      upload a PDF/image, kicks off the agent pipeline in the
                      background, returns immediately with a document_id.
GET  /documents/{id} poll for status/result.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

load_dotenv()

from api.models import DocumentStatus, ExtractedInvoice, UploadResponse
from agent.orchestrator import DocumentAgent
from storage import db
from tools.llm_extract import ExtractionFailedError
from tools.currency_convert import convert_currency, CurrencyConversionError
from tools.fraud_detection import detect_invoice_fraud
from tools.vendor_history import compare_with_vendor_history
from vision.gemini_client import GeminiUnavailableError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api")

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"

app = FastAPI(title="Document Intelligence Agent", version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()  # creates both the documents table and the vendor-history vector table


def _process_document(document_id: str, file_path: str) -> None:
    """Runs in the background after the API has already responded."""
    try:
        agent = DocumentAgent()
        result = agent.run(document_id, file_path)
        db.update_document(
            document_id,
            status=DocumentStatus.completed.value,
            raw_text=result.raw_text,
            structured_data=result.structured_data.model_dump(mode="json"),
            validation=result.validation.model_dump(mode="json"),
            summary=result.summary,
            processing_time_ms=result.processing_time_ms,
            llm_tokens_used=result.tokens_used,
        )
    except (ExtractionFailedError, GeminiUnavailableError) as exc:
        logger.error("document_id=%s pipeline_failed error=%s", document_id, exc)
        db.update_document(document_id, status=DocumentStatus.failed.value, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - top-level safety net for the background task
        logger.exception("document_id=%s pipeline_failed_unexpected", document_id)
        db.update_document(document_id, status=DocumentStatus.failed.value, error=f"Unexpected error: {exc}")


@app.post("/documents", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    dest_path = UPLOAD_DIR / f"{document_id}{ext}"
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    dest_path.write_bytes(contents)

    created_at = datetime.now(timezone.utc).isoformat()
    db.create_document(document_id, file.filename or dest_path.name, created_at)
    background_tasks.add_task(_process_document, document_id, str(dest_path))

    return UploadResponse(document_id=document_id, status=DocumentStatus.processing)


@app.get("/documents/{document_id}")
def get_document(document_id: str):
    record = db.get_document(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document_id not found")
    return JSONResponse(content=record)


@app.get("/documents")
def list_documents():
    return db.list_documents()


def _completed_record_or_404(document_id: str) -> dict:
    record = db.get_document(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document_id not found")
    if record["status"] != DocumentStatus.completed.value:
        raise HTTPException(
            status_code=409,
            detail=f"document_id is '{record['status']}', not completed yet",
        )
    return record


def _uploaded_file_path(document_id: str, filename: str) -> str:
    ext = Path(filename or "").suffix.lower() or ".pdf"
    return str(UPLOAD_DIR / f"{document_id}{ext}")


@app.post("/documents/{document_id}/convert-currency")
def convert_document_currency(document_id: str, to_currency: str):
    record = _completed_record_or_404(document_id)
    structured = record.get("structured_data") or {}
    try:
        result = convert_currency(
            amount=structured.get("total_amount"),
            from_currency=structured.get("currency"),
            to_currency=to_currency,
        )
    except CurrencyConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.post("/documents/{document_id}/fraud-detection")
def detect_document_fraud(document_id: str):
    record = _completed_record_or_404(document_id)
    structured = ExtractedInvoice.model_validate(record.get("structured_data") or {})
    file_path = _uploaded_file_path(document_id, record["filename"])
    return detect_invoice_fraud(document_id, file_path, structured, record.get("raw_text") or "")


@app.post("/documents/{document_id}/compare-vendor-history")
def compare_document_with_vendor_history(document_id: str):
    """RAG example query from the spec: 'Compare this invoice with previous
    invoices from the same vendor.'"""
    record = _completed_record_or_404(document_id)
    structured = ExtractedInvoice.model_validate(record.get("structured_data") or {})
    return compare_with_vendor_history(
        document_id, structured.vendor, record.get("raw_text") or "", structured.total_amount
    )


@app.get("/health")
def health():
    return {"status": "ok"}

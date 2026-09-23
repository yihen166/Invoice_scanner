"""Pydantic schemas shared across the API, agent, and tools.

Keeping these in one place is what lets Part 3 (LLM structured extraction)
enforce "schema validation" for free: the LLM's JSON output is parsed
straight into `ExtractedInvoice`, and Pydantic rejects anything that
doesn't fit the shape (wrong types, missing required fields, etc).
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    invoice = "invoice"
    receipt = "receipt"
    unknown = "unknown"


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class ExtractedInvoice(BaseModel):
    """The structured shape the LLM must produce (Part 3)."""

    document_type: DocumentType = DocumentType.unknown
    vendor: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None  # kept as str; validated separately
    due_date: Optional[str] = None
    total_amount: Optional[float] = None
    # No length constraint here: malformed currency codes should surface as a
    # validate_invoice() warning, not a hard schema-parse failure.
    currency: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("currency")
    @classmethod
    def upper_currency(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if v else v


class ValidationIssue(BaseModel):
    field: str
    message: str
    severity: str  # "error" | "warning"


class ValidationResult(BaseModel):
    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class DocumentStatus(str, Enum):
    processing = "processing"
    completed = "completed"
    failed = "failed"


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    raw_text: Optional[str] = None
    structured_data: Optional[ExtractedInvoice] = None
    validation: Optional[ValidationResult] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: Optional[int] = None
    llm_tokens_used: Optional[int] = None
    created_at: str


class UploadResponse(BaseModel):
    document_id: str
    status: DocumentStatus

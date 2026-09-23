"""Part 5 -- Data Validation Tool.

Deliberately implemented in plain Python, not another LLM call: totals math,
required-field checks, and currency codes are deterministic and should not
be left to a model that can hallucinate a "close enough" answer. The agent
calls this tool before generating the summary, as the spec requires.
"""
from __future__ import annotations

from datetime import date

from api.models import ExtractedInvoice, ValidationIssue, ValidationResult

REQUIRED_FIELDS = ["vendor", "invoice_number", "invoice_date", "total_amount", "currency"]
VALID_ISO_CURRENCY_LEN = 3
TOTAL_TOLERANCE = 0.01  # allow tiny rounding differences


def _check_missing_fields(data: ExtractedInvoice) -> list[ValidationIssue]:
    issues = []
    for field_name in REQUIRED_FIELDS:
        if getattr(data, field_name) in (None, ""):
            issues.append(
                ValidationIssue(field=field_name, message="Required field is missing", severity="error")
            )
    return issues


def _check_currency(data: ExtractedInvoice) -> list[ValidationIssue]:
    if data.currency and len(data.currency) != VALID_ISO_CURRENCY_LEN:
        return [
            ValidationIssue(
                field="currency",
                message=f"'{data.currency}' does not look like a 3-letter ISO currency code",
                severity="warning",
            )
        ]
    return []

def _check_dates(data: ExtractedInvoice) -> list[ValidationIssue]:
    issues = []
    parsed_invoice_date = None
    for field_name in ("invoice_date", "due_date"):
        value = getattr(data, field_name)
        if not value:
            continue
        try:
            parsed = date.fromisoformat(value)
            if field_name == "invoice_date":
                parsed_invoice_date = parsed
        except ValueError:
            issues.append(
                ValidationIssue(
                    field=field_name, message=f"'{value}' is not a valid YYYY-MM-DD date", severity="error"
                )
            )
    if data.due_date and parsed_invoice_date:
        try:
            due = date.fromisoformat(data.due_date)
            if due < parsed_invoice_date:
                issues.append(
                    ValidationIssue(
                        field="due_date",
                        message="due_date is earlier than invoice_date",
                        severity="warning",
                    )
                )
        except ValueError:
            pass  # already reported above
    return issues


def _check_totals(data: ExtractedInvoice) -> list[ValidationIssue]:
    if not data.line_items or data.total_amount is None:
        return []
    computed = sum(item.amount for item in data.line_items if item.amount is not None)
    if abs(computed - data.total_amount) > TOTAL_TOLERANCE:
        return [
            ValidationIssue(
                field="total_amount",
                message=f"Sum of line items ({computed:.2f}) does not match total_amount ({data.total_amount:.2f})",
                severity="warning",
            )
        ]
    return []


def validate_invoice(data: ExtractedInvoice) -> ValidationResult:
    issues = [
        *_check_missing_fields(data),
        *_check_currency(data),
        *_check_dates(data),
        *_check_totals(data),
    ]
    is_valid = not any(issue.severity == "error" for issue in issues)
    return ValidationResult(is_valid=is_valid, issues=issues)

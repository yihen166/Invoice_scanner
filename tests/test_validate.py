from api.models import ExtractedInvoice, LineItem
from tools.validate import validate_invoice


def _base_invoice(**overrides) -> ExtractedInvoice:
    data = dict(
        document_type="invoice",
        vendor="ABC Supplies Sdn Bhd",
        invoice_number="INV-10234",
        invoice_date="2026-01-15",
        due_date="2026-02-15",
        total_amount=3250.00,
        currency="MYR",
        line_items=[LineItem(description="Widgets", quantity=10, unit_price=325.0, amount=3250.0)],
    )
    data.update(overrides)
    return ExtractedInvoice(**data)


def test_valid_invoice_passes():
    result = validate_invoice(_base_invoice())
    assert result.is_valid
    assert result.issues == []


def test_missing_required_field_is_error():
    result = validate_invoice(_base_invoice(vendor=None))
    assert not result.is_valid
    assert any(i.field == "vendor" and i.severity == "error" for i in result.issues)


def test_bad_currency_is_warning_not_error():
    result = validate_invoice(_base_invoice(currency="MYRR"))
    assert result.is_valid  # warnings don't block validity
    assert any(i.field == "currency" and i.severity == "warning" for i in result.issues)


def test_invalid_date_format_is_error():
    result = validate_invoice(_base_invoice(invoice_date="15-01-2026"))
    assert not result.is_valid
    assert any(i.field == "invoice_date" for i in result.issues)


def test_due_date_before_invoice_date_is_warning():
    result = validate_invoice(_base_invoice(invoice_date="2026-02-15", due_date="2026-01-15"))
    assert result.is_valid
    assert any(i.field == "due_date" and "earlier" in i.message for i in result.issues)


def test_mismatched_total_is_warning():
    result = validate_invoice(
        _base_invoice(line_items=[LineItem(description="Widgets", amount=100.0)], total_amount=3250.0)
    )
    assert result.is_valid
    assert any(i.field == "total_amount" for i in result.issues)

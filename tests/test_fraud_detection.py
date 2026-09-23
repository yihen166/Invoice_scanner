from unittest.mock import patch

import pymupdf
import pytest
from PIL import Image

from api.models import ExtractedInvoice, LineItem
from tools.fraud_detection import detect_invoice_fraud, RISK_THRESHOLD

NO_HISTORY = {"history_found": False, "past_invoice_count": 0, "message": "No history yet."}


def _invoice(**overrides) -> ExtractedInvoice:
    data = dict(
        vendor="ABC Supplies Sdn Bhd",
        invoice_number="INV-10234",
        invoice_date="2026-01-15",
        total_amount=3250.0,
        currency="MYR",
        line_items=[LineItem(description="10x industrial widgets", amount=3250.0)],
    )
    data.update(overrides)
    return ExtractedInvoice(**data)


def _make_pdf(path, creation_date: str | None = None, producer: str | None = None):
    doc = pymupdf.open()
    doc.new_page()
    if producer:
        doc.set_metadata({"producer": producer, "creator": producer})
    if creation_date:
        doc.set_metadata({**doc.metadata, "creationDate": creation_date})
    doc.save(path)
    doc.close()


@patch("tools.fraud_detection._search_hits_for", return_value=3)
@patch("tools.vendor_history.compare_with_vendor_history", return_value=NO_HISTORY)
def test_complete_consistent_invoice_scores_low_and_not_flagged(mock_history, mock_search, tmp_path):
    pdf_path = tmp_path / "invoice.pdf"
    _make_pdf(pdf_path, creation_date="D:20260115090000+08'00'", producer="QuickBooks Online")
    raw_text = (
        "ABC Supplies Sdn Bhd, 123 Jalan Ampang, 50450 Kuala Lumpur\n"
        "GST No: 123456789\nBank Account No: 1234567890\nPhone: +60312345678"
    )
    result = detect_invoice_fraud("doc_1", str(pdf_path), _invoice(), raw_text)
    assert result["overall_score"] < RISK_THRESHOLD
    assert result["flagged"] is False
    assert len(result["checks"]) == 4


@patch("tools.fraud_detection._search_hits_for", return_value=0)
@patch("tools.vendor_history.compare_with_vendor_history", return_value=NO_HISTORY)
def test_sparse_invoice_with_bad_metadata_scores_high_and_flagged(mock_history, mock_search, tmp_path):
    pdf_path = tmp_path / "invoice.pdf"
    # Created ~8 months after the invoice_date it claims -- a real red flag.
    _make_pdf(pdf_path, creation_date="D:20260901090000+08'00'")
    raw_text = "Sample Company Inc\nItem 1\nTotal: 100"  # no address/tax id/bank/contact
    result = detect_invoice_fraud(
        "doc_2",
        str(pdf_path),
        _invoice(
            vendor="Sample Company Inc", invoice_number=None, total_amount=None, currency=None,
            line_items=[LineItem(description="Item 1", amount=100.0)],
        ),
        raw_text,
    )
    assert result["overall_score"] >= RISK_THRESHOLD
    assert result["flagged"] is True


@patch("tools.fraud_detection._search_hits_for", return_value=None)
@patch("tools.vendor_history.compare_with_vendor_history", return_value=NO_HISTORY)
def test_search_failure_does_not_crash(mock_history, mock_search, tmp_path):
    pdf_path = tmp_path / "invoice.pdf"
    _make_pdf(pdf_path)
    result = detect_invoice_fraud("doc_3", str(pdf_path), _invoice(), "some text")
    assert 0 <= result["overall_score"] <= 100


@patch("tools.fraud_detection._search_hits_for", return_value=2)
@patch("tools.vendor_history.compare_with_vendor_history", return_value=NO_HISTORY)
def test_image_without_exif_raises_metadata_risk(mock_history, mock_search, tmp_path):
    img_path = tmp_path / "invoice.jpg"
    Image.new("RGB", (10, 10)).save(img_path)  # no EXIF
    raw_text = "ABC Supplies Sdn Bhd, 123 Jalan Ampang, 50450 Kuala Lumpur"
    result = detect_invoice_fraud("doc_4", str(img_path), _invoice(), raw_text)
    metadata_check = next(c for c in result["checks"] if c["name"] == "Document Metadata Analysis")
    assert metadata_check["score"] >= 30


@patch("tools.vendor_history.compare_with_vendor_history")
def test_large_deviation_from_vendor_history_raises_risk(mock_history, tmp_path):
    pdf_path = tmp_path / "invoice.pdf"
    _make_pdf(pdf_path)
    mock_history.return_value = {
        "history_found": True,
        "past_invoice_count": 3,
        "average_past_total": 1000.0,
        "current_total": 9000.0,
        "deviation_from_average_pct": 800.0,
        "past_invoices": [],
    }
    with patch("tools.fraud_detection._search_hits_for", return_value=2):
        result = detect_invoice_fraud("doc_5", str(pdf_path), _invoice(total_amount=9000.0), "text")
    history_check = next(c for c in result["checks"] if c["name"] == "Vendor History Consistency")
    assert history_check["score"] >= 60


def test_disclaimer_and_threshold_always_present(tmp_path):
    pdf_path = tmp_path / "invoice.pdf"
    _make_pdf(pdf_path)
    with patch("tools.fraud_detection._search_hits_for", return_value=1), \
         patch("tools.vendor_history.compare_with_vendor_history", return_value=NO_HISTORY):
        result = detect_invoice_fraud("doc_6", str(pdf_path), _invoice(), "text")
    assert "heuristic" in result["disclaimer"].lower()
    assert result["threshold"] == RISK_THRESHOLD

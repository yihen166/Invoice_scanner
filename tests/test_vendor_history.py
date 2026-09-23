from unittest.mock import patch

from tools.vendor_history import (
    _cosine_similarity,
    compare_with_vendor_history,
    find_similar_past_invoices,
    store_invoice_embedding,
)


def test_cosine_similarity_identical_vectors_is_one():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_does_not_divide_by_zero():
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


@patch("tools.vendor_history.embed_text", side_effect=RuntimeError("no api key"))
def test_store_embedding_failure_is_silent(mock_embed):
    # Should not raise, even though embed_text always fails here -- and
    # since it fails before any DB write, no real database is touched.
    store_invoice_embedding("doc_1", "Acme", "some text", 100.0, "MYR", "2026-01-01")


@patch("tools.vendor_history.embed_text", side_effect=RuntimeError("no api key"))
def test_find_similar_returns_empty_on_embedding_failure(mock_embed):
    result = find_similar_past_invoices("doc_1", "Acme", "some text")
    assert result == []


@patch("tools.vendor_history.find_similar_past_invoices", return_value=[])
def test_compare_with_no_history_reports_not_found(mock_find):
    result = compare_with_vendor_history("doc_1", "Acme", "text", 100.0)
    assert result["history_found"] is False
    assert result["past_invoice_count"] == 0


@patch("tools.vendor_history.find_similar_past_invoices")
def test_compare_computes_deviation_from_average(mock_find):
    mock_find.return_value = [
        {"document_id": "doc_a", "total_amount": 100.0, "currency": "MYR", "invoice_date": "2026-01-01", "similarity": 0.9},
        {"document_id": "doc_b", "total_amount": 120.0, "currency": "MYR", "invoice_date": "2026-02-01", "similarity": 0.8},
    ]
    result = compare_with_vendor_history("doc_c", "Acme", "text", 300.0)
    assert result["history_found"] is True
    assert result["average_past_total"] == 110.0
    assert result["deviation_from_average_pct"] > 0
    assert len(result["past_invoices"]) == 2

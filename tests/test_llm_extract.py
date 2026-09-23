from unittest.mock import patch

from tools.llm_extract import llm_extract
from vision.gemini_client import GeminiCallResult

VALID_JSON = """{
  "document_type": "invoice",
  "vendor": "ABC Supplies Sdn Bhd",
  "invoice_number": "INV-10234",
  "invoice_date": "2026-01-15",
  "due_date": "2026-02-15",
  "total_amount": 3250.0,
  "currency": "MYR",
  "line_items": []
}"""


@patch("tools.llm_extract.extract_structured_data")
def test_parses_valid_json_first_try(mock_call):
    mock_call.return_value = GeminiCallResult(text=VALID_JSON, tokens_used=120, latency_ms=500)
    data, tokens = llm_extract("some raw invoice text")
    assert data.vendor == "ABC Supplies Sdn Bhd"
    assert tokens == 120
    assert mock_call.call_count == 1


@patch("tools.llm_extract.extract_structured_data")
def test_retries_once_on_malformed_json(mock_call):
    mock_call.side_effect = [
        GeminiCallResult(text="not json at all", tokens_used=50, latency_ms=400),
        GeminiCallResult(text=VALID_JSON, tokens_used=110, latency_ms=450),
    ]
    data, tokens = llm_extract("some raw invoice text")
    assert data.invoice_number == "INV-10234"
    assert tokens == 160  # both attempts counted
    assert mock_call.call_count == 2

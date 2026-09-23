from unittest.mock import Mock, patch

import pytest
import requests

from tools.currency_convert import convert_currency, CurrencyConversionError


def test_same_currency_short_circuits_no_network_call():
    result = convert_currency(100.0, "MYR", "myr")
    assert result == {"amount": 100.0, "from": "MYR", "to": "MYR", "rate": 1.0, "converted": 100.0, "as_of": None}


@patch("tools.currency_convert.requests.get")
def test_successful_conversion(mock_get):
    mock_get.return_value = Mock(
        status_code=200,
        json=lambda: {"amount": 3250.0, "base": "MYR", "date": "2026-09-22", "rates": {"USD": 690.06}},
    )
    mock_get.return_value.raise_for_status = lambda: None
    result = convert_currency(3250.0, "MYR", "USD")
    assert result["converted"] == 690.06
    assert result["as_of"] == "2026-09-22"
    assert result["rate"] == pytest.approx(690.06 / 3250.0)


def test_missing_amount_raises():
    with pytest.raises(CurrencyConversionError):
        convert_currency(None, "MYR", "USD")


@patch("tools.currency_convert.requests.get")
def test_network_error_wrapped(mock_get):
    mock_get.side_effect = requests.ConnectionError("no network")
    with pytest.raises(CurrencyConversionError):
        convert_currency(100.0, "MYR", "USD")

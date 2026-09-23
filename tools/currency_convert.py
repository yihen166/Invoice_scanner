"""Optional tool: convert an extracted invoice total into another currency.

Not part of the core auto-pipeline (the spec's example schema has one
currency field, not a conversion step) -- this is invoked on demand, e.g.
from the CLI, after extraction has already produced a total_amount and
currency.

Uses api.frankfurter.app: free, no API key, backed by European Central Bank
reference rates. Covers ~30 major currencies including MYR, USD, SGD, EUR,
GBP, JPY, CNY, etc. Note: ECB does not publish rates on EUR holidays, so
`as_of` may lag by a day or two on weekends/holidays -- fine for this use
case, not a substitute for a live trading rate.
"""
from __future__ import annotations

import requests

FRANKFURTER_URL = "https://api.frankfurter.app/latest"
TIMEOUT_SECONDS = 10


class CurrencyConversionError(RuntimeError):
    pass


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    if amount is None:
        raise CurrencyConversionError("No amount to convert (total_amount is missing)")

    from_ccy = (from_currency or "").strip().upper()
    to_ccy = (to_currency or "").strip().upper()
    if not from_ccy or not to_ccy:
        raise CurrencyConversionError("Both source and target currency codes are required")

    if from_ccy == to_ccy:
        return {
            "amount": amount, "from": from_ccy, "to": to_ccy,
            "rate": 1.0, "converted": amount, "as_of": None,
        }

    try:
        resp = requests.get(
            FRANKFURTER_URL,
            params={"amount": amount, "from": from_ccy, "to": to_ccy},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        converted = data["rates"][to_ccy]
    except requests.RequestException as exc:
        raise CurrencyConversionError(f"Network error contacting exchange rate service: {exc}") from exc
    except (KeyError, ValueError) as exc:
        raise CurrencyConversionError(
            f"'{to_ccy}' or '{from_ccy}' is not a currency this service supports"
        ) from exc

    return {
        "amount": amount,
        "from": from_ccy,
        "to": to_ccy,
        "rate": round(converted / amount, 6) if amount else None,
        "converted": round(converted, 2),
        "as_of": data.get("date"),
    }

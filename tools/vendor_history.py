"""RAG System (bonus) -- store past invoices' embeddings and let both the
fraud detector and an explicit query answer:
"Compare this invoice with previous invoices from the same vendor."

Storage is a plain SQLite table (see storage/db.py) with cosine similarity
computed in Python -- not a dedicated vector database. That's a deliberate
scale/simplicity tradeoff for a take-home repo (see ENGINEERING_NOTES.md);
the interface here (`store_invoice_embedding`, `find_similar_past_invoices`)
is what would stay the same if a real vector DB were swapped in later.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from storage import db
from vision.gemini_client import embed_text

SIMILARITY_THRESHOLD = 0.55  # cosine similarity floor to count as "the same vendor"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _canonical_text(vendor: Optional[str], raw_text: str) -> str:
    # Weighted toward vendor identity so embeddings cluster by vendor,
    # not just by generic invoice boilerplate.
    return f"Vendor: {vendor or 'unknown'}\n\n{raw_text[:2000]}"


def store_invoice_embedding(
    document_id: str, vendor: Optional[str], raw_text: str,
    total_amount: Optional[float], currency: Optional[str], invoice_date: Optional[str],
) -> None:
    """Best-effort: a failed embedding just means this document won't be
    findable as "history" later -- it should never fail the main pipeline."""
    try:
        embedding = embed_text(_canonical_text(vendor, raw_text))
    except Exception:
        return
    db.store_invoice_vector(
        document_id=document_id, vendor=vendor, embedding=embedding,
        total_amount=total_amount, currency=currency, invoice_date=invoice_date,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def find_similar_past_invoices(
    document_id: str, vendor: Optional[str], raw_text: str, top_k: int = 5,
) -> list[dict]:
    try:
        query_embedding = embed_text(_canonical_text(vendor, raw_text))
    except Exception:
        return []

    candidates = db.list_invoice_vectors(exclude_document_id=document_id)
    scored = [
        {**c, "similarity": _cosine_similarity(query_embedding, c["embedding"])}
        for c in candidates
    ]
    scored = [c for c in scored if c["similarity"] >= SIMILARITY_THRESHOLD]
    scored.sort(key=lambda c: c["similarity"], reverse=True)
    return scored[:top_k]


def compare_with_vendor_history(
    document_id: str, vendor: Optional[str], raw_text: str, current_total: Optional[float],
) -> dict:
    """The RAG example query from the spec, as a standalone tool (also used
    internally by the 'Vendor History Consistency' fraud check)."""
    similar = find_similar_past_invoices(document_id, vendor, raw_text)
    if not similar:
        return {
            "vendor": vendor,
            "history_found": False,
            "past_invoice_count": 0,
            "message": "No sufficiently similar past invoices found for this vendor yet.",
        }

    totals = [c["total_amount"] for c in similar if c["total_amount"] is not None]
    avg_total = sum(totals) / len(totals) if totals else None
    deviation_pct = (
        round(abs(current_total - avg_total) / avg_total * 100, 1)
        if current_total is not None and avg_total else None
    )

    return {
        "vendor": vendor,
        "history_found": True,
        "past_invoice_count": len(similar),
        "average_past_total": round(avg_total, 2) if avg_total is not None else None,
        "current_total": current_total,
        "deviation_from_average_pct": deviation_pct,
        "past_invoices": [
            {
                "document_id": c["document_id"],
                "invoice_date": c["invoice_date"],
                "total_amount": c["total_amount"],
                "currency": c["currency"],
                "similarity": round(c["similarity"], 3),
            }
            for c in similar
        ],
    }

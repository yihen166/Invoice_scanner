"""Thin SQLite persistence layer.

Chosen over in-memory storage so documents/results survive an API restart,
without pulling in a full ORM or external DB service for a take-home.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "documents.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    status TEXT NOT NULL,
    raw_text TEXT,
    structured_data TEXT,
    validation TEXT,
    summary TEXT,
    error TEXT,
    processing_time_ms INTEGER,
    llm_tokens_used INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS invoice_vectors (
    document_id TEXT PRIMARY KEY,
    vendor TEXT,
    embedding TEXT NOT NULL,
    total_amount REAL,
    currency TEXT,
    invoice_date TEXT,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def create_document(document_id: str, filename: str, created_at: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO documents (document_id, filename, status, created_at) "
            "VALUES (?, ?, 'processing', ?)",
            (document_id, filename, created_at),
        )


def update_document(document_id: str, **fields) -> None:
    """Update arbitrary columns. dict/list values are JSON-encoded."""
    if not fields:
        return
    cols = []
    values = []
    for key, value in fields.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        cols.append(f"{key} = ?")
        values.append(value)
    values.append(document_id)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE documents SET {', '.join(cols)} WHERE document_id = ?",
            values,
        )


def get_document(document_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        for json_field in ("structured_data", "validation"):
            if result.get(json_field):
                result[json_field] = json.loads(result[json_field])
        return result


def list_documents(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT document_id, filename, status, created_at FROM documents "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# --- RAG-style vendor-history vector store -----------------------------
# A plain SQLite table, not a dedicated vector DB (Chroma/FAISS/pgvector):
# at this scale (a personal take-home demo, dozens-to-hundreds of
# invoices) a linear scan with cosine similarity is plenty fast and keeps
# the repo dependency-light and easy to run on Windows with no extra
# build tooling. See tools/vendor_history.py for the similarity search
# logic; if this needed to scale to millions of documents, swapping in a
# real vector DB there would be a drop-in replacement.

def store_invoice_vector(
    document_id: str, vendor: Optional[str], embedding: list[float],
    total_amount: Optional[float], currency: Optional[str],
    invoice_date: Optional[str], created_at: str,
) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO invoice_vectors "
            "(document_id, vendor, embedding, total_amount, currency, invoice_date, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (document_id, vendor, json.dumps(embedding), total_amount, currency, invoice_date, created_at),
        )


def list_invoice_vectors(exclude_document_id: Optional[str] = None) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM invoice_vectors").fetchall()
    result = []
    for row in rows:
        record = dict(row)
        if exclude_document_id and record["document_id"] == exclude_document_id:
            continue
        record["embedding"] = json.loads(record["embedding"])
        result.append(record)
    return result

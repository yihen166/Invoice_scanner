# Document Intelligence Agent

An AI agent that takes an uploaded invoice/receipt (PDF or image), extracts
its text, converts it to structured JSON via an LLM, validates the result,
and produces a short summary.

## Pipeline

```
POST /documents
   │
   ▼
extract_text     — native PDF text (PyMuPDF) or Gemini vision for scans/images
   │
   ▼
llm_extract       — Gemini converts text → structured JSON (schema-validated via Pydantic)
   │
   ▼
validate_invoice  — pure-code checks: missing fields, currency format, dates, totals
   │
   ▼
summarize         — Gemini writes a short human-readable report
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env

```

## Run

```bash
uvicorn api.main:app --reload
```

API docs (Swagger UI) at http://localhost:8000/docs

## Try it

```bash
curl -F "file=@sample_invoice.pdf" http://localhost:8000/documents
# -> {"document_id": "doc_abc123...", "status": "processing"}

curl http://localhost:8000/documents/doc_abc123...
# -> full record: raw_text, structured_data, validation, summary
```

`GET /documents` lists recent uploads. `GET /health` is a liveness check.

## Interactive CLI (optional, easier than curl)

Once the server is running (`uvicorn api.main:app --reload`), open a
**second** terminal in the same folder and run:

```bash
python cli.py
```

It will ask for a filename, upload it, show a spinner while it processes,
then walk through the pipeline stages interactively (raw text → ask to
convert to structured data → structured JSON → validation + summary), and
finally offer a menu of optional tools: show total value, convert currency,
run the AI/synthetic-invoice risk check, or compare against past invoices
from the same vendor. This talks to the running API
over HTTP — it's a convenience wrapper, not a replacement for the API.

## Optional tools (beyond the core spec)

- **Currency conversion** — `POST /documents/{id}/convert-currency?to_currency=USD`.
  Uses the free Frankfurter (ECB) exchange-rate API, no key required.
- **AI - Fraud Detection (invoice)** — `POST /documents/{id}/fraud-detection`.
  Four independently-scored checks (0-100 risk each), combined into one
  overall score. **>= 50 is flagged**, below is not:
  1. **Company & Address Verification** — best-effort web search for the
     vendor name (no API key), plus a structural check on whether the
     address text looks complete.
  2. **Document Metadata Analysis** — for PDFs: producer/creator tags and
     whether the file's internal creation date lines up with the invoice
     date it claims. For images: EXIF camera tags.
  3. **Information Completeness** — how many of the fields a genuine
     invoice normally carries (tax ID, address, bank details, contact
     info, etc.) are actually present.
  4. **Vendor History Consistency (RAG)** — looks up past invoices from
     the same vendor (see below) and flags a total that deviates sharply
     from that vendor's history.

  See the disclaimer in its own response — this is a heuristic signal for
  human review, not a verified fraud detector.

- **Vendor history (RAG)** — `POST /documents/{id}/compare-vendor-history`.
  Every successfully-processed invoice is embedded (Gemini's embedding
  model) and stored in a SQLite-backed vector table. This endpoint answers
  the spec's example RAG query directly: "compare this invoice with
  previous invoices from the same vendor" — returning matched past
  invoices (by embedding similarity) and how this one's total compares to
  that vendor's historical average. Feeds into fraud check 4 above too.

## Tests

```bash
pytest
```

All tests run **without** a Gemini API key — validation and native-PDF
extraction are pure code, and the LLM-extraction tests mock the Gemini call.
The only thing that genuinely requires `GEMINI_API_KEY` is running the live
API end-to-end against a real document.

## Repo layout

```
api/            FastAPI app: upload + retrieval endpoints, Pydantic schemas
agent/          orchestrator.py — the 4-step pipeline (Part 4)
tools/          extract_text, llm_extract, validate, summarize, currency_convert,
                fraud_detection, vendor_history (RAG) — one file per tool
vision/         Gemini client wrapper (used as VLM, LLM, and embeddings)
storage/        SQLite persistence (documents + vendor-history vectors)
tests/          unit tests (validation, extraction, LLM-parsing/retry logic)
```

See `ENGINEERING_NOTES.md` for design rationale, limitations, and what I'd
change for production.

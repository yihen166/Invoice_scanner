# Engineering Notes

## Architecture decisions

**One LLM provider for two roles.** Gemini 2.0 Flash is multimodal, so the
same client handles both the "VLM" role (Part 2, reading scanned/image
documents) and the "LLM structured extraction" role (Part 3). This avoids
running two model providers (e.g. Tesseract + GPT) for what is really one
capability — reading a document — and keeps API-key management, retry
logic, and token accounting in one place (`vision/gemini_client.py`).

**Text-based PDFs skip the LLM entirely.** `tools/extract_text.py` tries
PyMuPDF's native text layer first and only calls the vision model if that
comes back empty/too short (a scanned PDF) or the upload is a raw image.
Most real invoices are digital PDFs, so this keeps the common case fast and
free of LLM cost/latency, and keeps the VLM path exercised only when it's
actually needed.

**A plain function-calling orchestrator, not an agent framework.**
`agent/orchestrator.py`'s `DocumentAgent.run()` calls four tools in a fixed
sequence: `extract_text → llm_extract → validate_invoice → summarize`. I
didn't reach for LangChain/CrewAI-style ReAct loops because the workflow is
known ahead of time — there's no decision for a general planner to make
about *which* tool to call next. What makes it agent-like rather than a
plain script is that each step is an independently testable tool with a
typed contract, the orchestrator enforces the one hard ordering constraint
the spec calls out (`validate_invoice` must run before `summarize`), and it
owns cross-cutting concerns (timing, token counting, error boundaries)
across the whole run rather than leaving them scattered in each tool.

**Schema validation via Pydantic, not manual dict-checking.** The LLM's
JSON output is parsed straight into `ExtractedInvoice` (`api/models.py`).
Wrong types or malformed shapes raise a `ValidationError` immediately,
which `tools/llm_extract.py` catches and uses to retry once with the error
fed back to the model — a cheap guardrail against one-off JSON formatting
slips, capped at one retry so a genuinely broken response fails fast
instead of looping.

**Validation is pure code, deliberately not another LLM call.** Whether a
total matches the sum of line items, whether a date parses, whether a
currency code is 3 letters — these are deterministic checks. Asking an LLM
to "check" them risks a model that's *agreeable* rather than *correct*.
`tools/validate.py` separates issues into `error` (blocks `is_valid`, e.g.
missing required fields) vs. `warning` (informational, e.g. totals off by
a few cents, currency looks malformed) so the summary step can mention
concerns without hard-failing the whole document.

**SQLite over in-memory storage.** A dict would have been simpler, but
survives a restart poorly and doesn't reflect how you'd actually run this.
SQLite needs zero setup and is still trivial to read (`storage/db.py`).

**Upload returns immediately; processing happens in a background task.**
Matches the spec's example response (`{"status": "processing"}`) and means
a slow OCR/LLM call doesn't block the HTTP request. The client polls
`GET /documents/{id}` for the result.

## Prompt design

- **OCR prompt** (`ocr_document`): explicitly told to transcribe only —
  "do not summarize or interpret" — because early testing without that
  line risked the vision model "helpfully" cleaning up or reformatting
  numbers, which is exactly the kind of silent corruption you don't want
  in a financial pipeline.
- **Extraction prompt** (`extract_structured_data`): the JSON shape is
  spelled out inline in the prompt *and* enforced by `response_mime_type:
  "application/json"` in the Gemini call config, plus a second, independent
  check via Pydantic on the response. Belt and suspenders — the model's
  "JSON mode" reduces malformed output but doesn't guarantee the *shape* is
  right (right field names, right types).
- **Retry prompt**: on a schema failure, the previous (invalid) output and
  the specific Pydantic error are appended to the input, so the retry is
  targeted ("your last JSON was invalid: <error>") rather than a blind
  second attempt with the same prompt.
- **Summary prompt**: given both the structured data *and* the validation
  result, with an explicit instruction to mention flagged issues — so a
  document with a totals mismatch doesn't get a falsely clean-sounding
  summary.

## Optional tools added beyond the core spec

**`cli.py`** is a convenience layer, not a replacement for the API — the
assessment explicitly asks for an upload API, so that stays the source of
truth. The CLI just polls it and presents results progressively (raw text,
then structured data, then validation+summary) instead of requiring manual
`curl` calls at each step.

**Currency conversion** (`tools/currency_convert.py`) calls the free
Frankfurter API (ECB reference rates, no key). Deliberately *not* part of
the core auto-pipeline — the spec's expected output has a single `currency`
field, not a conversion — so it's exposed as an on-demand endpoint/CLI
option instead.

**AI - Fraud Detection (invoice)** (`tools/fraud_detection.py`) replaced an
earlier, vaguer "synthetic document" heuristic with four concrete, named,
separately-scored checks (company/address verification, document metadata
analysis, information completeness, vendor history consistency), combined
into one overall score with an explicit 50-point flag threshold. I still
want to be explicit about the same limitation as before: **there is no
reliable, general-purpose way to prove a document was AI-generated or
fraudulent**, and none of these four checks does that individually either:
- The web search for company existence has no way to confirm legitimacy
  either way -- a hit doesn't prove the invoice is real, and a miss is
  common for small/local businesses with no web presence.
- Metadata (PDF producer/creator, image EXIF) can be stripped or forged by
  legitimate tools too. The one genuinely strong signal here is a PDF's
  internal creation timestamp landing long after the invoice date it
  claims -- that's hard to fake accidentally and is weighted accordingly.
- Missing fields (no tax ID, no bank details) are sometimes just how a
  particular business's invoices look.
- A vendor-history deviation can be a legitimate one-off large order, and
  a first-time vendor scores low/neutral rather than risky -- absence of
  history isn't evidence of fraud.

The response always carries a `disclaimer` field, and the CLI's report
frames a flagged result as "worth a human review," not a verdict. If this
were going into a real fraud-screening product, I'd want labeled training
data and a proper classifier, and even then I'd keep a human in the loop
before treating any of this as a decision-maker.

**Vendor history / RAG** (`tools/vendor_history.py`) is the bonus "RAG
System" from the spec, implemented as a SQLite-backed vector store rather
than a dedicated vector database (Chroma/FAISS/pgvector). At this scale --
a personal take-home demo, dozens-to-hundreds of invoices -- a linear scan
with cosine similarity in plain Python is fast enough and keeps the repo
dependency-light and easy to run on Windows with no extra build tooling.
Every successfully-processed invoice is embedded (`gemini-embedding-001`)
and indexed in the background as part of `DocumentAgent.run()`; indexing
is deliberately best-effort and non-fatal -- a failed embedding call never
fails the document's own pipeline, it just means that document won't be
findable as "history" later. `compare_with_vendor_history()` answers the
spec's exact example query ("compare this invoice with previous invoices
from the same vendor") and doubles as fraud check 4. If this needed to
scale up, swapping in a real vector DB would only touch this one file --
the interface (`store_invoice_embedding`, `find_similar_past_invoices`)
would stay the same.

## Limitations

- OCR quality for scanned/handwritten documents depends entirely on the
  underlying vision model; there's no confidence score surfaced from
  Gemini itself, so a bad transcription can silently produce a
  plausible-looking but wrong structured result.
- Only one retry on extraction failure — sufficient to smooth over
  formatting slips, not repeated model errors.
- Background processing is in-process (FastAPI `BackgroundTasks`), which
  is fine for a demo but won't survive a server restart mid-processing or
  scale across multiple workers.
- No auth/rate-limiting on the upload endpoint.
- Vendor-history matching only sees invoices this system has itself
  processed, not a vendor's full real-world invoice history -- a vendor's
  very first upload here always looks like "no history," even if they've
  sent this business a hundred invoices on paper.
- The CLI's animated stage labels (e.g. "Extracting text...", "Checking
  company & address...") rotate on a fixed timer for a smooth visual, not
  from real per-step progress signals from the server -- the underlying
  HTTP call is a single request/response, so the labels are a cosmetic
  approximation of what's likely happening, not a live trace.
- Currency conversion (mentioned as an example validation tool in the
  spec) isn't implemented — validation flags a malformed currency *code*
  but doesn't call an FX API to convert amounts.
- Line-item-level validation (e.g. quantity × unit_price = amount) isn't
  checked, only the invoice-level total vs. sum of line items.

## What I'd improve for production

- Move background processing to a real task queue (Celery/RQ + Redis or
  SQS) so it survives restarts and scales horizontally.
- Add a confidence/hallucination check: cross-reference a couple of
  extracted fields (e.g. total_amount) against a second, cheaper
  regex/heuristic pass over the raw text, and flag disagreement.
- Structured JSON logging shipped somewhere queryable (not just stdout),
  keyed by `document_id`, so `processing_time_ms` and `llm_tokens_used`
  can be tracked over time and per-step, not just at the end of a run.
- Auth on the API, and per-user rate limits.
- Store uploaded files in object storage (S3) instead of local disk.
- Expand test coverage to include a mocked end-to-end API test (upload →
  poll) in the committed test suite, alongside a couple of low-quality
  scanned-document fixtures to exercise the VLM fallback path specifically.

"""Interactive CLI for the Document Intelligence Agent.

This is a convenience layer on top of the API -- it does not replace it
(the assessment requires a real HTTP upload API, which lives in api/main.py
and keeps working with plain curl/Postman/etc). Run this in a *second*
terminal while `uvicorn api.main:app --reload` is running in the first.

Usage:
    python cli.py
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import requests

API_BASE = "http://localhost:8000"
POLL_INTERVAL_SECONDS = 1.5
POLL_TIMEOUT_SECONDS = 90


# ---------------------------------------------------------------------------
# Animated loading
# ---------------------------------------------------------------------------

def _advanced_spinner(stop_event: threading.Event, stages: list[str]) -> None:
    """A braille-frame spinner that rotates through a list of stage labels,
    so a multi-step wait doesn't look frozen or restart-flicker on every
    poll. Runs continuously for the whole duration of the wrapped call."""
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    start = time.time()
    frame_idx = 0
    stage_idx = 0
    last_stage_switch = start
    stage_interval = 0.9
    while not stop_event.is_set():
        now = time.time()
        if now - last_stage_switch > stage_interval:
            stage_idx = (stage_idx + 1) % len(stages)
            last_stage_switch = now
        frame = frames[frame_idx % len(frames)]
        elapsed = now - start
        print(f"\r{frame} {stages[stage_idx]} ({elapsed:0.1f}s)" + " " * 10, end="", flush=True)
        frame_idx += 1
        time.sleep(0.08)
    print("\r" + " " * 70 + "\r", end="", flush=True)


def _with_advanced_spinner(stages: list[str], fn, *args, **kwargs):
    stop_event = threading.Event()
    t = threading.Thread(target=_advanced_spinner, args=(stop_event, stages), daemon=True)
    t.start()
    try:
        return fn(*args, **kwargs)
    finally:
        stop_event.set()
        t.join()


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _risk_bar(score: int, width: int = 20) -> str:
    filled = round(width * score / 100)
    return "█" * filled + "░" * (width - filled)


def _risk_label(score: int) -> str:
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MEDIUM"
    return "LOW"


def render_fraud_report(result: dict) -> str:
    lines = [
        "",
        "=" * 46,
        "   AI - Fraud Detection (Invoice) Report",
        "=" * 46,
    ]
    for i, check in enumerate(result["checks"], start=1):
        lines.append(f"[{i}] {check['name']}")
        lines.append(f"    {_risk_bar(check['score'])}  {check['score']:>3}/100  ({_risk_label(check['score'])})")
        for finding in check["findings"]:
            lines.append(f"      - {finding}")
        lines.append("")
    lines.append("-" * 46)
    overall = result["overall_score"]
    lines.append(f"OVERALL RISK SCORE:  {_risk_bar(overall)}  {overall:>3}/100")
    lines.append(f"Threshold: {result['threshold']}  ->  {result['verdict']}")
    lines.append("-" * 46)
    lines.append(result["disclaimer"])
    lines.append("=" * 46)
    return "\n".join(lines)


def render_vendor_history(result: dict) -> str:
    lines = ["", "-" * 46, "   Vendor History Comparison (RAG)", "-" * 46]
    if not result.get("history_found"):
        lines.append(result.get("message", "No history found."))
        lines.append("-" * 46)
        return "\n".join(lines)

    lines.append(f"Vendor: {result.get('vendor') or 'unknown'}")
    lines.append(f"Similar past invoices found: {result['past_invoice_count']}")
    if result.get("average_past_total") is not None:
        lines.append(f"Historical average total: {result['average_past_total']}")
        lines.append(f"This invoice's total:     {result.get('current_total')}")
        if result.get("deviation_from_average_pct") is not None:
            lines.append(f"Deviation from average:  {result['deviation_from_average_pct']}%")
    lines.append("")
    lines.append("Past invoices (most similar first):")
    for inv in result.get("past_invoices", []):
        lines.append(
            f"  - {inv['document_id']}  date={inv.get('invoice_date')}  "
            f"total={inv.get('total_amount')} {inv.get('currency') or ''}  "
            f"similarity={inv['similarity']}"
        )
    lines.append("-" * 46)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

def _check_server() -> bool:
    try:
        requests.get(f"{API_BASE}/health", timeout=3)
        return True
    except requests.RequestException:
        return False


def _resolve_file(raw_input_path: str) -> Path:
    path = Path(raw_input_path.strip().strip('"'))
    if path.exists():
        return path
    # Common case: user typed just a filename, expecting it in the same
    # folder as this script.
    candidate = Path(__file__).parent / raw_input_path.strip().strip('"')
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Couldn't find '{raw_input_path}' (checked as given, and next to cli.py)"
    )


def upload_document(file_path: Path) -> str:
    stop_event = threading.Event()
    t = threading.Thread(
        target=_advanced_spinner, args=(stop_event, [f"Uploading {file_path.name}..."]), daemon=True
    )
    t.start()
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(f"{API_BASE}/documents", files={"file": (file_path.name, f)})
        resp.raise_for_status()
    finally:
        stop_event.set()
        t.join()
    data = resp.json()
    print(f"\nUpload: {file_path.name}")
    print(json.dumps(data, indent=2))
    return data["document_id"]


def wait_for_completion(document_id: str) -> dict:
    """One continuous animation for the whole wait, rather than restarting a
    spinner on every poll -- reads as a smooth 'processing' animation
    instead of flickering every couple of seconds."""
    stages = [
        "Extracting text from document...",
        "Running LLM structured extraction...",
        "Validating extracted data...",
        "Generating summary...",
        "Indexing for vendor history (RAG)...",
    ]
    stop_event = threading.Event()
    t = threading.Thread(target=_advanced_spinner, args=(stop_event, stages), daemon=True)
    t.start()
    try:
        deadline = time.time() + POLL_TIMEOUT_SECONDS
        while time.time() < deadline:
            record = requests.get(f"{API_BASE}/documents/{document_id}").json()
            if record["status"] != "processing":
                return record
            time.sleep(POLL_INTERVAL_SECONDS)
        raise TimeoutError(f"Document {document_id} did not finish within {POLL_TIMEOUT_SECONDS}s")
    finally:
        stop_event.set()
        t.join()


def _ask_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please type y or n.")


def show_raw_text(record: dict) -> None:
    print("\n--- Part 2: Extracted Text ---")
    print(json.dumps({"document_id": record["document_id"], "raw_text": record["raw_text"]}, indent=2))


def show_structured_data(record: dict) -> None:
    print("\n--- Part 3: LLM Structured Extraction ---")
    print(json.dumps(record["structured_data"], indent=2))


def show_validation_and_summary(record: dict) -> None:
    print("\n--- Part 4/5: Validation + Summary ---")
    print(json.dumps({
        "document_id": record["document_id"],
        "validation": record["validation"],
        "summary": record["summary"],
    }, indent=2))
    print(f"\nprocessing_time_ms={record['processing_time_ms']}  llm_tokens_used={record['llm_tokens_used']}")


def tool_menu(document_id: str, record: dict) -> None:
    structured = record.get("structured_data") or {}
    while True:
        print("\n--- Optional tools ---")
        print("1) Show total value")
        print("2) Convert currency")
        print("3) AI - Fraud Detection (invoice)")
        print("4) Compare with past invoices from this vendor (RAG)")
        print("5) Done")
        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            total = structured.get("total_amount")
            currency = structured.get("currency")
            print(f"\nTotal value: {total} {currency}" if total is not None else "\nNo total_amount was extracted.")

        elif choice == "2":
            target = input("Convert to which currency code (e.g. USD, SGD, EUR): ").strip().upper()
            try:
                resp = requests.post(
                    f"{API_BASE}/documents/{document_id}/convert-currency",
                    params={"to_currency": target},
                )
                resp.raise_for_status()
                print(json.dumps(resp.json(), indent=2))
            except requests.HTTPError as exc:
                print(f"Conversion failed: {exc.response.json().get('detail', exc)}")

        elif choice == "3":
            resp = _with_advanced_spinner(
                [
                    "Checking company & address...",
                    "Inspecting document metadata...",
                    "Checking information completeness...",
                    "Looking up vendor history...",
                    "Calculating overall risk...",
                ],
                requests.post,
                f"{API_BASE}/documents/{document_id}/fraud-detection",
            )
            print(render_fraud_report(resp.json()))

        elif choice == "4":
            resp = _with_advanced_spinner(
                ["Retrieving past invoices from this vendor...", "Comparing with history..."],
                requests.post,
                f"{API_BASE}/documents/{document_id}/compare-vendor-history",
            )
            print(render_vendor_history(resp.json()))

        elif choice == "5":
            return
        else:
            print("Please choose 1, 2, 3, 4, or 5.")


def main() -> None:
    print("Document Intelligence Agent -- interactive CLI")
    if not _check_server():
        print(
            "\nCan't reach the API at http://localhost:8000.\n"
            "Start it first, in another terminal, from the repo folder:\n"
            "    python -m uvicorn api.main:app --reload\n"
        )
        sys.exit(1)

    raw_path = input("Enter the PDF/image filename (in this folder, or a full path): ")
    try:
        file_path = _resolve_file(raw_path)
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)

    document_id = upload_document(file_path)

    try:
        record = wait_for_completion(document_id)
    except TimeoutError as exc:
        print(str(exc))
        sys.exit(1)

    if record["status"] == "failed":
        print(f"\nProcessing failed: {record.get('error')}")
        sys.exit(1)

    show_raw_text(record)

    if _ask_yes_no("\nConvert to structured data?"):
        show_structured_data(record)
        show_validation_and_summary(record)
        tool_menu(document_id, record)
    else:
        print("\nDone -- raw text only, as requested.")


if __name__ == "__main__":
    main()

"""AI - Fraud Detection (invoice).

Four independent, separately-scored checks, combined into one overall risk
score (0-100, higher = more suspicious). Overall >= 50 is flagged as
"possible AI-generated / fraudulent"; below 50 is "not flagged."

IMPORTANT CAVEAT: none of these checks *prove* fraud or AI-generation.
Each is a weak, explainable signal. Treat "flagged" as "worth a human
review," never as a verdict. This is surfaced in the returned disclaimer,
not just here.

Checks:
  1. Company & Address Verification -- best-effort web search for the
     vendor name (DuckDuckGo's public HTML search, no API key), plus a
     structural check on whether the extracted address looks complete
     (street indicator + postal code). No search API can *confirm* a
     business is legitimate; absence of hits is weak evidence at best,
     especially for small/local businesses with no web presence.
  2. Document Metadata Analysis -- PDFs: producer/creator tags and
     whether the file's internal creation date lines up with the
     invoice_date it claims (a file "created" long after the date it
     represents is a real red flag). Images: EXIF data -- real
     photos/scans usually carry camera Make/Model tags; many synthetic
     or screenshotted images don't.
  3. Information Completeness -- how many of the fields a genuine
     invoice normally carries (tax ID, address, bank/payment details,
     contact info, line items, etc.) are actually present.
  4. Vendor History Consistency (RAG) -- looks up past invoices from the
     same vendor via tools/vendor_history.py's embedding-similarity
     search, and checks whether this invoice's total is consistent with
     that vendor's history.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from PIL.ExifTags import TAGS

from api.models import ExtractedInvoice

RISK_THRESHOLD = 50
SEARCH_TIMEOUT_SECONDS = 8

AI_IMAGE_TOOL_HINTS = ["stable diffusion", "midjourney", "dall-e", "dalle", "comfyui", "diffusers"]
GENERATOR_TOOL_HINTS = [
    "wkhtmltopdf", "weasyprint", "reportlab", "headless", "puppeteer",
    "chromium", "skia/pdf", "tcpdf", "fpdf", "latex", "pandoc",
]
KNOWN_ACCOUNTING_SOFTWARE = [
    "quickbooks", "xero", "sap", "sage", "freshbooks", "zoho", "wave",
    "netsuite", "myob", "odoo",
]
ADDRESS_STREET_WORDS = r"\b(street|road|jalan|ave|avenue|blvd|lorong|taman|st\.)\b"
POSTAL_CODE_PATTERN = r"\b\d{4,6}\b"


@dataclass
class CheckResult:
    name: str
    score: int  # 0-100, higher = more suspicious
    findings: list[str]
    caveat: str


def _clamp(score: float) -> int:
    return max(0, min(100, round(score)))


# ---------------------------------------------------------------------------
# Check 1: Company & Address Verification
# ---------------------------------------------------------------------------

def _search_hits_for(query: str) -> Optional[int]:
    """Returns an approximate result count, or None if the search itself failed."""
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; InvoiceFraudCheck/1.0)"},
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return len(re.findall(r'class="result__a"', resp.text))
    except requests.RequestException:
        return None


def check_company_and_address(data: ExtractedInvoice) -> CheckResult:
    findings: list[str] = []
    scores: list[float] = []

    vendor = (data.vendor or "").strip()
    if vendor:
        hits = _search_hits_for(vendor)
        if hits is None:
            findings.append("Online search could not be performed (network issue) -- skipped")
            scores.append(25)
        elif hits == 0:
            findings.append(f"No web search results found for vendor name '{vendor}'")
            scores.append(55)
        else:
            findings.append(f"Found {hits} web search result(s) mentioning '{vendor}'")
            scores.append(10)
    else:
        findings.append("No vendor name was extracted, so no search could be run")
        scores.append(40)

    return CheckResult(
        name="Company & Address Verification",
        score=_clamp(sum(scores) / len(scores)),
        findings=findings,
        caveat=(
            "A missing/absent web presence is common for small local "
            "businesses and is not proof of anything; a found presence is "
            "not proof of legitimacy either."
        ),
    )


def _address_structure_score(raw_text: str) -> tuple[int, str]:
    has_street = bool(re.search(ADDRESS_STREET_WORDS, raw_text, re.IGNORECASE))
    has_postal = bool(re.search(POSTAL_CODE_PATTERN, raw_text))
    if has_street and has_postal:
        return 5, "Address text includes a street indicator and a postal code"
    if has_street or has_postal:
        return 35, "Address text is only partially present (missing street or postal code)"
    return 65, "No recognizable address (street name / postal code) found in the document text"


# ---------------------------------------------------------------------------
# Check 2: Document Metadata Analysis
# ---------------------------------------------------------------------------

def _parse_pdf_date(raw: str) -> Optional[datetime]:
    # PDF date format: D:YYYYMMDDHHmmSS(+HH'mm' | -HH'mm' | Z)
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", raw or "")
    if not match:
        return None
    y, mo, d, h, mi, s = (int(g) if g else 0 for g in match.groups())
    try:
        return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
    except ValueError:
        return None


def _check_pdf_metadata(file_path: Path, invoice_date_str: Optional[str]) -> CheckResult:
    findings: list[str] = []
    scores: list[float] = []
    try:
        import pymupdf
        with pymupdf.open(file_path) as doc:
            meta = {k: (v or "") for k, v in (doc.metadata or {}).items()}
    except Exception:
        return CheckResult(
            name="Document Metadata Analysis", score=30,
            findings=["Could not read PDF metadata"],
            caveat="Metadata can be stripped or forged; this is a weak signal either way.",
        )

    producer_creator = f"{meta.get('producer', '')} {meta.get('creator', '')}".lower()
    if not meta.get("producer") and not meta.get("creator"):
        findings.append("No producer/creator metadata at all (common in stripped or synthetic files)")
        scores.append(45)
    elif any(t in producer_creator for t in KNOWN_ACCOUNTING_SOFTWARE):
        findings.append(f"Producer/creator matches known accounting software ({producer_creator.strip()})")
        scores.append(5)
    elif any(t in producer_creator for t in GENERATOR_TOOL_HINTS):
        findings.append(f"Producer/creator is a generic document-generation tool ({producer_creator.strip()})")
        scores.append(40)
    else:
        findings.append(f"Producer/creator: {producer_creator.strip() or 'unknown'}")
        scores.append(20)

    creation_dt = _parse_pdf_date(meta.get("creationDate", ""))
    if creation_dt is None:
        findings.append("PDF has no internal creation timestamp")
        scores.append(30)
    elif invoice_date_str:
        try:
            claimed_date = datetime.fromisoformat(invoice_date_str).replace(tzinfo=timezone.utc)
            gap_days = (creation_dt - claimed_date).days
            if gap_days > 30:
                findings.append(
                    f"File was created {gap_days} days AFTER the invoice date it claims ({invoice_date_str})"
                )
                scores.append(75)
            elif gap_days < -1:
                findings.append("File's internal creation date is before its own claimed invoice date")
                scores.append(30)
            else:
                findings.append("File creation date is consistent with the claimed invoice date")
                scores.append(5)
        except ValueError:
            scores.append(20)

    return CheckResult(
        name="Document Metadata Analysis",
        score=_clamp(sum(scores) / len(scores)) if scores else 30,
        findings=findings,
        caveat="Metadata can be edited or stripped by legitimate tools too; treat as supporting evidence only.",
    )


def _check_image_exif(file_path: Path) -> CheckResult:
    findings: list[str] = []
    scores: list[float] = []
    try:
        with Image.open(file_path) as img:
            exif_raw = img.getexif()
            exif = {TAGS.get(k, k): v for k, v in exif_raw.items()} if exif_raw else {}
    except Exception:
        return CheckResult(
            name="Document Metadata Analysis", score=30,
            findings=["Could not read image metadata"],
            caveat="Metadata can be stripped or forged; this is a weak signal either way.",
        )

    if not exif:
        findings.append("Image has no EXIF metadata at all (common for screenshots, downloads, or generated images)")
        scores.append(45)
    else:
        if exif.get("Make") or exif.get("Model"):
            findings.append(f"Image has camera device tags ({exif.get('Make', '')} {exif.get('Model', '')})".strip())
            scores.append(5)
        else:
            findings.append("Image has EXIF data but no camera Make/Model tags")
            scores.append(30)

        software = str(exif.get("Software", "")).lower()
        if any(tool in software for tool in AI_IMAGE_TOOL_HINTS):
            findings.append(f"EXIF Software tag references a known AI image tool ({software})")
            scores.append(85)

    return CheckResult(
        name="Document Metadata Analysis",
        score=_clamp(sum(scores) / len(scores)),
        findings=findings,
        caveat="Absence of EXIF is common even for genuine documents (many scanners/exports strip it).",
    )


def check_document_metadata(file_path: str, invoice_date_str: Optional[str]) -> CheckResult:
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        return _check_pdf_metadata(path, invoice_date_str)
    return _check_image_exif(path)


# ---------------------------------------------------------------------------
# Check 3: Information Completeness
# ---------------------------------------------------------------------------

def check_information_completeness(data: ExtractedInvoice, raw_text: str) -> CheckResult:
    components = {
        "vendor name": bool(data.vendor),
        "invoice number": bool(data.invoice_number),
        "invoice date": bool(data.invoice_date),
        "total amount": data.total_amount is not None,
        "currency": bool(data.currency),
        "line items": bool(data.line_items),
        "tax/business ID": bool(re.search(r"\b(tax id|gst|vat|sst|ein|tin)\b", raw_text, re.IGNORECASE)),
        "address": bool(re.search(ADDRESS_STREET_WORDS, raw_text, re.IGNORECASE)),
        "bank/payment details": bool(re.search(r"\b(bank|account no|iban|swift)\b", raw_text, re.IGNORECASE)),
        "contact info": bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+|\b\+?\d[\d\s-]{7,}\d\b", raw_text)),
    }
    missing = [name for name, present in components.items() if not present]
    score = _clamp((len(missing) / len(components)) * 100)
    findings = (
        [f"Missing: {', '.join(missing)}"] if missing
        else ["All commonly-expected invoice fields are present"]
    )
    return CheckResult(
        name="Information Completeness",
        score=score,
        findings=findings,
        caveat="Some genuine invoices legitimately omit fields (e.g. no bank details on a receipt).",
    )


# ---------------------------------------------------------------------------
# Check 4: Vendor History Consistency (RAG)
# ---------------------------------------------------------------------------

def check_vendor_history(document_id: str, data: ExtractedInvoice, raw_text: str) -> CheckResult:
    from tools.vendor_history import compare_with_vendor_history

    comparison = compare_with_vendor_history(document_id, data.vendor, raw_text, data.total_amount)

    if not comparison["history_found"]:
        return CheckResult(
            name="Vendor History Consistency", score=15,
            findings=["No prior invoices on record for this vendor yet -- nothing to compare against"],
            caveat="A first-time vendor is not inherently suspicious; it just means there's no history yet.",
        )

    findings = [f"Found {comparison['past_invoice_count']} similar past invoice(s) from this vendor"]
    deviation = comparison.get("deviation_from_average_pct")
    if deviation is not None:
        findings.append(
            f"Total amount deviates {deviation}% from this vendor's historical average "
            f"({comparison['average_past_total']} {data.currency or ''})".strip()
        )
        score = 70 if deviation > 200 else 40 if deviation > 75 else 10
    else:
        score = 15

    return CheckResult(
        name="Vendor History Consistency", score=score, findings=findings,
        caveat="A large deviation can be a legitimate one-off large order, not necessarily fraud.",
    )


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

def detect_invoice_fraud(document_id: str, file_path: str, data: ExtractedInvoice, raw_text: str) -> dict:
    check1 = check_company_and_address(data)
    # Fold address-structure signal into check 1's score/findings.
    addr_score, addr_finding = _address_structure_score(raw_text)
    check1.findings.append(addr_finding)
    check1.score = _clamp((check1.score + addr_score) / 2)

    check2 = check_document_metadata(file_path, data.invoice_date)
    check3 = check_information_completeness(data, raw_text)
    check4 = check_vendor_history(document_id, data, raw_text)

    checks = [check1, check2, check3, check4]
    overall = _clamp(sum(c.score for c in checks) / len(checks))
    flagged = overall >= RISK_THRESHOLD

    return {
        "overall_score": overall,
        "flagged": flagged,
        "verdict": (
            "Flagged: possible AI-generated / fraudulent invoice"
            if flagged else
            "Not flagged: looks like a genuine invoice"
        ),
        "threshold": RISK_THRESHOLD,
        "checks": [
            {"name": c.name, "score": c.score, "findings": c.findings, "caveat": c.caveat}
            for c in checks
        ],
        "disclaimer": (
            "Heuristic estimate only, not a verified detector. Treat a "
            "flagged result as 'worth a human review', not proof."
        ),
    }

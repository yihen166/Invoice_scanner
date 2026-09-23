from pathlib import Path

import pymupdf
import pytest

from tools.extract_text import extract_text


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "ABC Supplies Sdn Bhd\nInvoice INV-10234\nTotal: MYR 3250.00")
    path = tmp_path / "sample.pdf"
    doc.save(path)
    doc.close()
    return path


def test_native_pdf_extraction_used_for_text_pdf(sample_pdf: Path):
    result = extract_text(str(sample_pdf))
    assert result.method == "pdf_native"
    assert "INV-10234" in result.raw_text
    assert result.tokens_used == 0  # no LLM call needed


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_text(str(tmp_path / "does_not_exist.pdf"))

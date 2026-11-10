"""Real synthetic PDF fixtures verify extraction; network responses are mocked."""

import re

import httpx
import pymupdf
import pytest

from pda.fulltext import ExtractedPaper, PdfStore, TextUnit, chunk_document, extract_pdf
from pda.summary_config import SummaryError, SummarySettings


def scientific_pdf():
    document = pymupdf.open()
    sections = [
        "1 Introduction",
        "2 Methods",
        "3 Results",
        "References",
        "Appendix A Additional results",
    ]
    for index, section in enumerate(sections):
        page = document.new_page()
        page.insert_text((60, 22), "Recurring running header", fontsize=9)
        page.insert_text((60, 810), str(index + 1), fontsize=9)
        page.insert_text((60, 90), section, fontsize=15)
        text = (
            "BIBLIOGRAPHY_ONLY_TOKEN Author and publication details. " * 12
            if section == "References"
            else f"SECTION_{index}_FINDING Eq. (1) and Table 2 refer to Figure 3. "
            + "This paragraph contains synthetic scientific evidence for a software extraction test. "
            * 18
        )
        assert page.insert_textbox(pymupdf.Rect(60, 130, 530, 720), text, fontsize=10) >= 0
    raw = document.tobytes()
    document.close()
    return raw


def test_extracts_all_sections_equation_references_and_appendix_but_removes_bibliography(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(scientific_pdf())
    document = extract_pdf(path, SummarySettings(cache_dir=tmp_path))
    assert document.pages == 5 and document.bibliography_removed
    assert document.removed_artifact_lines >= 5
    assert "Recurring running header" not in document.text
    assert "BIBLIOGRAPHY_ONLY_TOKEN" not in document.text
    assert "SECTION_0_FINDING" in document.text and "SECTION_4_FINDING" in document.text
    assert all(
        term in document.text
        for term in (
            "Introduction",
            "Methods",
            "Results",
            "Eq. (1)",
            "Table 2",
            "Figure 3",
            "Appendix",
        )
    )


def test_unnumbered_supplementary_heading_after_references_is_not_lost(tmp_path):
    document = pymupdf.open(stream=scientific_pdf(), filetype="pdf")
    page = document.new_page()
    page.insert_text((60, 90), "Attention Visualizations", fontsize=15)
    page.insert_textbox(
        pymupdf.Rect(60, 130, 530, 720),
        "SUPPLEMENT_FINDING Figure 9 shows a documented supplementary observation. " * 12,
        fontsize=10,
    )
    # Remove the explicit appendix so only the unnumbered post-reference heading can resume text.
    document.delete_page(4)
    path = tmp_path / "supplement.pdf"
    document.save(path)
    document.close()
    extracted = extract_pdf(path, SummarySettings())
    assert "SUPPLEMENT_FINDING" in extracted.text
    assert "Attention Visualizations" in extracted.text
    assert "BIBLIOGRAPHY_ONLY_TOKEN" not in extracted.text


def test_download_uses_versioned_pdf_url_and_reuses_source_for_other_queries(tmp_path):
    requests = []
    raw = scientific_pdf()

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(200, content=raw, headers={"content-type": "application/pdf"})

    store = PdfStore(SummarySettings(cache_dir=tmp_path), httpx.MockTransport(handler))
    first, sha = store.get("1706.03762v7")
    second, other_sha = store.get("1706.03762v7")
    assert first == second and sha == other_sha
    assert requests == ["https://arxiv.org/pdf/1706.03762v7"]
    assert first.read_bytes() == raw


@pytest.mark.parametrize("failure", ["http", "timeout", "not_pdf", "too_large"])
def test_pdf_download_failures_never_create_a_source_cache(tmp_path, monkeypatch, failure):
    monkeypatch.setattr("pda.fulltext.time.sleep", lambda _: None)

    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("test timeout")
        return httpx.Response(
            404 if failure == "http" else 200,
            content=b"%PDF-" + b"x" * 2048 if failure == "too_large" else b"not pdf",
        )

    store = PdfStore(
        SummarySettings(cache_dir=tmp_path, max_pdf_bytes=1024), httpx.MockTransport(handler)
    )
    with pytest.raises(SummaryError):
        store.get("1706.03762v7")
    assert not list(tmp_path.rglob("*.pdf"))


@pytest.mark.parametrize(
    "identifier", ["../outside", "1706.03762", "https://example.com/paper", "1706.03762v0"]
)
def test_source_cache_requires_safe_versioned_arxiv_id(tmp_path, identifier):
    store = PdfStore(
        SummarySettings(cache_dir=tmp_path),
        httpx.MockTransport(lambda _: pytest.fail("Unexpected network request")),
    )
    with pytest.raises(SummaryError):
        store.get(identifier)


@pytest.mark.parametrize("kind", ["broken", "insufficient", "scanned", "page_limit", "text_limit"])
def test_extraction_failures_and_processing_limits_are_explicit(tmp_path, kind):
    path = tmp_path / "paper.pdf"
    settings = SummarySettings(cache_dir=tmp_path)
    if kind == "broken":
        path.write_bytes(b"%PDF- broken")
    elif kind in {"page_limit", "text_limit"}:
        path.write_bytes(scientific_pdf())
        settings = settings.model_copy(
            update={"max_pages": 2} if kind == "page_limit" else {"max_text_chars": 800}
        )
    else:
        with pymupdf.open() as document:
            page = document.new_page()
            if kind == "scanned":
                page.draw_rect(pymupdf.Rect(30, 30, 100, 100))
            else:
                page.insert_text((60, 90), "Only an abstract.")
            document.save(path)
    with pytest.raises(SummaryError):
        extract_pdf(path, settings)


def test_chunking_preserves_every_word_and_section_including_long_sentences():
    units = [
        TextUnit(f"Section {i}", i + 1, " ".join(f"token_{i}_{j}" for j in range(150)))
        for i in range(4)
    ]
    document = ExtractedPaper(units, 4, False, 0)
    chunks = chunk_document(document, 500, 100)
    assert all(len(chunk.encode()) <= 500 for chunk in chunks)
    restored = " ".join(re.sub(r"\[Section:[^\n]*\]\n", "", chunk) for chunk in chunks)
    assert restored.split() == " ".join(unit.text for unit in units).split()
    assert "Section 3" in chunks[-1]


def test_normal_paragraphs_and_sentences_are_kept_together_when_they_fit():
    paragraph = "First sentence with findings. Second sentence with limitations."
    document = ExtractedPaper([TextUnit("Results", 1, paragraph)], 1, False, 0)
    chunks = chunk_document(document, 500, 10)
    assert len(chunks) == 1 and paragraph in chunks[0]


def test_chunk_limits_fail_instead_of_discarding_the_tail():
    document = ExtractedPaper([TextUnit("Results", 1, "word " * 2000)], 1, False, 0)
    with pytest.raises(SummaryError, match="no sections were omitted"):
        chunk_document(document, 500, 2)
    with pytest.raises(SummaryError, match="unsplittable"):
        chunk_document(ExtractedPaper([TextUnit("Results", 1, "x" * 2000)], 1, False, 0), 500, 10)

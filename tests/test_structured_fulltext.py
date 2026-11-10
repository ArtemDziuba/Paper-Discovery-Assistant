"""Real synthetic layouts and controlled native API output test the PyMuPDF 2 adapter."""

import pymupdf
import pytest

from pda.fulltext import ExtractedPaper, TextUnit, chunk_document, extract_pdf
from pda.summary_config import SummaryError, SummarySettings


def structured_page(blocks):
    text = ""
    boxes = []
    for index, (kind, content) in enumerate(blocks):
        start = len(text)
        text += content + "\n\n"
        boxes.append({"index": index, "class": kind, "pos": [start, len(text)]})
    return {"metadata": {"page_number": 1}, "text": text, "page_boxes": boxes}


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((60, 90), "Digital source text.")
        doc.save(path)
    return path


def test_native_multicolumn_reading_order(tmp_path):
    path = tmp_path / "columns.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((60, 70), "1 Introduction", fontsize=15)
        for x, y, marker in [
            (60, 110, "LEFT_FIRST"),
            (60, 380, "LEFT_SECOND"),
            (320, 110, "RIGHT_FIRST"),
            (320, 380, "RIGHT_SECOND"),
        ]:
            paragraph = marker + " " + "Scientific evidence and documented observations. " * 10
            assert (
                page.insert_textbox(pymupdf.Rect(x, y, x + 220, y + 230), paragraph, fontsize=10)
                >= 0
            )
        doc.save(path)
    result = extract_pdf(path, SummarySettings())
    positions = [
        result.text.index(m) for m in ("LEFT_FIRST", "LEFT_SECOND", "RIGHT_FIRST", "RIGHT_SECOND")
    ]
    assert positions == sorted(positions)


def test_native_table_preserves_column_and_row_associations(tmp_path):
    path = tmp_path / "table.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((60, 70), "3 Experimental Results", fontsize=15)
        paragraph = (
            "Synthetic scientific text for parser validation with documented findings. " * 18
        )
        assert page.insert_textbox(pymupdf.Rect(60, 95, 530, 300), paragraph, fontsize=10) >= 0
        columns = [60, 220, 390, 530]
        for x in columns:
            page.draw_line((x, 340), (x, 610))
        for row in range(10):
            y = 340 + row * 30
            page.draw_line((60, y), (530, y))
        rows = [
            ("Method", "Dataset", "Score"),
            *[(f"METHOD{i}", f"DATA{i}", str(10 + i)) for i in range(8)],
        ]
        for row, cells in enumerate(rows):
            for x, cell in zip(columns, cells):
                page.insert_text((x + 8, 360 + row * 30), cell, fontsize=10)
        doc.save(path)
    result = extract_pdf(path, SummarySettings())
    tables = [u.text for u in result.units if u.kind == "table"]
    assert tables
    table = "\n".join(tables)
    assert "|Method|Dataset|Score|" in table
    assert all(f"|METHOD{i}|DATA{i}|{10 + i}|" in table for i in range(8))


def test_inline_reference_boundary_keeps_acknowledgements_and_resumes_appendix(source, monkeypatch):
    record = structured_page(
        [
            ("section-header", "## Results"),
            ("text", "SUPPORTED_FINDING " + "Scientific evidence and limitations. " * 40),
            (
                "text",
                "**Acknowledgements** Thanks to reviewers. **References** [1] BIB_ONLY_TOKEN. 2020.",
            ),
            ("section-header", "## Attention Visualizations"),
            ("caption", "APPENDIX_FINDING Figure 5 illustrates a supplementary finding."),
        ]
    )
    monkeypatch.setattr(pymupdf.Document, "to_markdown", lambda *args, **kwargs: [record])
    result = extract_pdf(source, SummarySettings())
    assert result.bibliography_removed
    assert "Thanks to reviewers" in result.text and "SUPPORTED_FINDING" in result.text
    assert "BIB_ONLY_TOKEN" not in result.text and "APPENDIX_FINDING" in result.text
    assert result.units[-1].kind == "caption"


def test_native_spans_keep_unclassified_text_unknown_types_and_fenced_code(source, monkeypatch):
    record = structured_page(
        [
            ("text", "UNCOVERED_FINDING " + "Documented observations. " * 40),
            ("future-element-type", "UNRECOGNIZED_KIND_FINDING"),
            ("text", "```python\nfirst_call()\n\nsecond_call()\n```"),
        ]
    )
    record["page_boxes"].pop(0)  # Native metadata omitted a span; retain that text gap.
    record["page_boxes"].reverse()  # Respect reading indices, not dictionary storage order.
    monkeypatch.setattr(pymupdf.Document, "to_markdown", lambda *args, **kwargs: [record])
    result = extract_pdf(source, SummarySettings())
    assert "UNCOVERED_FINDING" in result.text and "UNRECOGNIZED_KIND_FINDING" in result.text
    code = [u for u in result.units if u.kind == "code"]
    assert len(code) == 1 and "first_call()\n\nsecond_call()" in code[0].text


@pytest.mark.parametrize(
    "problem", ["missing_page", "wrong_page", "missing_layout", "bad_span", "overlap"]
)
def test_incomplete_or_malformed_native_output_fails_without_legacy_fallback(
    source, monkeypatch, problem
):
    record = structured_page([("text", "Paper evidence " * 100), ("text", "Conclusions")])
    pages = [record]
    if problem == "missing_page":
        pages = []
    elif problem == "wrong_page":
        record["metadata"]["page_number"] = 2
    elif problem == "missing_layout":
        record.pop("page_boxes")
    elif problem == "bad_span":
        record["page_boxes"][0]["pos"] = [0, len(record["text"]) + 1]
    else:
        record["page_boxes"][1]["pos"][0] = 1
    monkeypatch.setattr(pymupdf.Document, "to_markdown", lambda *args, **kwargs: pages)
    with pytest.raises(SummaryError):
        extract_pdf(source, SummarySettings())


def test_layout_api_failure_is_reported_without_falling_back(source, monkeypatch):
    def fail(*args, **kwargs):
        assert kwargs["use_ocr"] is False
        assert kwargs["write_images"] is False and kwargs["embed_images"] is False
        raise RuntimeError("Native layout failed")

    monkeypatch.setattr(pymupdf.Document, "to_markdown", fail)
    with pytest.raises(SummaryError, match="structured PDF extraction failed"):
        extract_pdf(source, SummarySettings())


def test_large_table_chunks_repeat_headers_and_keep_each_complete_row_once():
    header = "|Method|Dataset|Finding|\n|---|---|---|"
    rows = [f"|ROW{i}|DATA{i}|Supported observation {i}|" for i in range(35)]
    table = header + "\n" + "\n".join(rows)
    doc = ExtractedPaper([TextUnit("Results", 4, table, "table")], 4, False, 0)
    chunks = chunk_document(doc, 400, 100)
    assert len(chunks) > 1 and all(len(c.encode()) <= 400 for c in chunks)
    assert all(header in chunk and "page 4" in chunk for chunk in chunks)
    observed = [line for chunk in chunks for line in chunk.splitlines() if line.startswith("|ROW")]
    assert observed == rows


def test_small_table_stays_in_one_piece():
    table = "|Method|Score|\n|---|---|\n|MethodA|10|\n|MethodB|20|"
    doc = ExtractedPaper([TextUnit("Results", 1, table, "table")], 1, False, 0)
    assert table in chunk_document(doc, 400, 10)[0]


@pytest.mark.parametrize("kind", ["table_row", "unstructured_table", "formula", "code"])
def test_oversized_structural_elements_fail_instead_of_cutting_cells_or_syntax(kind):
    if kind == "table_row":
        unit = TextUnit("Results", 1, "|Column|\n|---|\n|" + "value " * 100 + "|", "table")
    elif kind == "unstructured_table":
        unit = TextUnit("Results", 1, "<table>" + "value " * 100 + "</table>", "table")
    else:
        unit = TextUnit("Methods", 1, "structured element " * 100, kind)
    with pytest.raises(SummaryError):
        chunk_document(ExtractedPaper([unit], 1, False, 0), 400, 10)

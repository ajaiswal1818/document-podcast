from pathlib import Path

import fitz

from podcast.document.parser import DocumentParser, chunk_text, extract_pdf


def test_document_parser_reads_text_file(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello from the document parser", encoding="utf-8")

    parser = DocumentParser(file_path)

    assert parser.parse() == "hello from the document parser"


def test_chunk_text_splits_long_content() -> None:
    text = "\n\n".join(["A" * 2000, "B" * 2000, "C" * 2000])

    chunks = chunk_text(text, max_chars=2500)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 3000 for chunk in chunks)


def test_extract_pdf_reads_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hello local podcast!")
    document.save(pdf_path)
    document.close()

    extracted = extract_pdf(str(pdf_path))

    assert "Hello local podcast!" in extracted

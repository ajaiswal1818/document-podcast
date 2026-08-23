"""Document parsing primitives."""

from __future__ import annotations

from pathlib import Path

import fitz


def extract_pdf(path: str | Path) -> str:
    """Extract plain text from a PDF document page by page."""
    pdf_path = Path(path)
    document = fitz.open(pdf_path)

    pages: list[str] = []
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")
        if text and text.strip():
            pages.append(f"\n--- PAGE {page_number} ---\n{text}")

    document.close()
    return "\n".join(pages)


def chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    """Split long text into manageable chunks for local LLM processing."""
    if not text or not text.strip():
        return []

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return [text.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for paragraph in paragraphs:
        if current and current_size + len(paragraph) > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0

        current.append(paragraph)
        current_size += len(paragraph)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


class DocumentParser:
    """Simple document parser for PDF and text files."""

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)

    def parse(self) -> str:
        """Return the text contents of the document."""
        if not self.source_path.exists():
            raise FileNotFoundError(f"Document not found: {self.source_path}")

        if self.source_path.suffix.lower() == ".pdf":
            return extract_pdf(self.source_path)

        return self.source_path.read_text(encoding="utf-8")

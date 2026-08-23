"""Document parsing primitives."""

from __future__ import annotations

from pathlib import Path


class DocumentParser:
    """Simple document parser interface for future PDF/text extraction."""

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)

    def parse(self) -> str:
        """Return the text contents of the document."""
        if not self.source_path.exists():
            raise FileNotFoundError(f"Document not found: {self.source_path}")

        return self.source_path.read_text(encoding="utf-8")

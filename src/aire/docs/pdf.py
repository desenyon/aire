"""PDF to text/pages Document pipeline (lazy ``pypdf``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aire.core.content import DocumentContent, TextContent
from aire.core.errors import ConfigurationError
from aire.data.dataset import Dataset
from aire.data.types import Record


class PDFPage(BaseModel):
    page: int
    text: str


class PDFDocument(BaseModel):
    path: str
    pages: list[PDFPage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    def to_document_content(self) -> DocumentContent:
        return DocumentContent(
            data=self.text.encode("utf-8"),
            media_type="text/plain",
            page_count=len(self.pages),
            metadata={**self.metadata, "path": self.path},
        )

    def to_records(self, *, per_page: bool = True) -> list[Record]:
        if per_page:
            return [
                Record(
                    text=p.text,
                    metadata={"page": p.page, "path": self.path, **self.metadata},
                )
                for p in self.pages
                if p.text.strip()
            ]
        return [
            Record(text=self.text, metadata={"path": self.path, "pages": len(self.pages), **self.metadata})
        ]


def _require_pypdf() -> Any:
    if importlib.util.find_spec("pypdf") is None:
        raise ConfigurationError(
            "pypdf is required for PDF loading: pip install 'aire[pypdf]'",
            code="docs.pypdf_missing",
            context={"extra": "aire[pypdf]", "package": "pypdf"},
        )
    import pypdf  # type: ignore[import-not-found]

    return pypdf


def load_pdf(path: str | Path, *, password: str | None = None) -> PDFDocument:
    """Extract text per page from a PDF file."""
    pypdf = _require_pypdf()
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise ConfigurationError(
            f"PDF not found: {pdf_path}",
            code="docs.pdf_not_found",
            context={"path": str(pdf_path)},
        )
    reader = pypdf.PdfReader(str(pdf_path), password=password)
    pages: list[PDFPage] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - keep pipeline resilient
            text = ""
        pages.append(PDFPage(page=i + 1, text=text))
    meta: dict[str, Any] = {"page_count": len(pages)}
    if reader.metadata:
        for key in ("title", "author", "subject"):
            value = getattr(reader.metadata, key, None)
            if value:
                meta[key] = str(value)
    return PDFDocument(path=str(pdf_path), pages=pages, metadata=meta)


def pdf_to_dataset(path: str | Path, *, per_page: bool = True, name: str | None = None) -> Dataset:
    doc = load_pdf(path)
    return Dataset(doc.to_records(per_page=per_page), name=name or Path(path).stem)


def pdf_to_text_content(path: str | Path) -> TextContent:
    return TextContent(text=load_pdf(path).text)


def describe() -> dict[str, Any]:
    return {
        "kind": "pdf",
        "available": importlib.util.find_spec("pypdf") is not None,
        "install": "pip install 'aire[pypdf]'",
        "outputs": ["PDFDocument", "Dataset", "DocumentContent", "TextContent"],
    }

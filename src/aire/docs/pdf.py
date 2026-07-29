"""PDF to text/pages Document pipeline (lazy ``pypdf``).

By default extracts the embedded text layer only. Pass ``ocr=True`` to attempt
page rendering + OCR via optional ``pillow`` + ``pytesseract`` (+ ``pypdfium2``
rasterizer) when the text layer is empty (scanned PDFs).
"""

from __future__ import annotations

import importlib.util
import warnings
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
    ocr: bool = False


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
                    metadata={
                        "page": p.page,
                        "path": self.path,
                        "ocr": p.ocr,
                        **self.metadata,
                    },
                )
                for p in self.pages
                if p.text.strip()
            ]
        return [
            Record(
                text=self.text,
                metadata={"path": self.path, "pages": len(self.pages), **self.metadata},
            )
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


def load_pdf(  # noqa: C901
    path: str | Path,
    *,
    password: str | None = None,
    raise_on_empty: bool = False,
    ocr: bool = False,
    ocr_lang: str = "eng",
) -> PDFDocument:
    """Extract text per page from a PDF.

    ``ocr=False`` (default): text layer only.
    ``ocr=True``: when a page has no text, rasterize via pypdfium2 and run
    pytesseract (requires ``pip install 'aire[ocr]'``).
    """
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
        except Exception:
            text = ""
        used_ocr = False
        if ocr and not text.strip():
            ocr_text = _ocr_page_index(pdf_path, i, lang=ocr_lang)
            if ocr_text.strip():
                text = ocr_text
                used_ocr = True
        pages.append(PDFPage(page=i + 1, text=text, ocr=used_ocr))
    meta: dict[str, Any] = {
        "page_count": len(pages),
        "text_layer_only": not ocr,
        "ocr": ocr,
        "ocr_pages": sum(1 for p in pages if p.ocr),
    }
    if reader.metadata:
        for key in ("title", "author", "subject"):
            value = getattr(reader.metadata, key, None)
            if value:
                meta[key] = str(value)
    doc = PDFDocument(path=str(pdf_path), pages=pages, metadata=meta)
    if not doc.text.strip():
        suffix = (
            "; OCR also returned nothing)"
            if ocr
            else "; pass ocr=True for scanned pages)"
        )
        message = f"PDF text extract is empty for {pdf_path} (text layer empty{suffix}"
        if raise_on_empty:
            raise ConfigurationError(
                message,
                code="docs.pdf_empty_extract",
                context={"path": str(pdf_path), "pages": len(pages), "ocr": ocr},
            )
        warnings.warn(message, UserWarning, stacklevel=2)
    return doc


def _ocr_page_index(pdf_path: Path, page_index: int, *, lang: str) -> str:
    """Rasterize one PDF page and OCR it. Requires ocr extra."""
    missing = [
        name
        for name, mod in (
            ("pillow", "PIL"),
            ("pytesseract", "pytesseract"),
            ("pypdfium2", "pypdfium2"),
        )
        if importlib.util.find_spec(mod) is None
    ]
    if missing:
        raise ConfigurationError(
            "OCR requires pillow, pytesseract, and pypdfium2: pip install 'aire[ocr]'",
            code="docs.ocr_missing",
            context={"extra": "aire[ocr]", "missing": missing},
        )
    import pypdfium2 as pdfium  # type: ignore[import-not-found]
    import pytesseract  # type: ignore[import-not-found]

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc[page_index]
        bitmap = page.render(scale=2).to_pil()
        return str(pytesseract.image_to_string(bitmap, lang=lang))
    finally:
        doc.close()


def pdf_to_dataset(
    path: str | Path,
    *,
    per_page: bool = True,
    name: str | None = None,
    ocr: bool = False,
) -> Dataset:
    doc = load_pdf(path, ocr=ocr)
    return Dataset(doc.to_records(per_page=per_page), name=name or Path(path).stem)


def pdf_to_text_content(path: str | Path, *, ocr: bool = False) -> TextContent:
    return TextContent(text=load_pdf(path, ocr=ocr).text)


def describe() -> dict[str, Any]:
    return {
        "kind": "pdf",
        "available": importlib.util.find_spec("pypdf") is not None,
        "install": "pip install 'aire[pypdf]'",
        "ocr": {
            "extra": "aire[ocr]",
            "packages": ["pillow", "pytesseract", "pypdfium2"],
        },
        "honesty": "default is text layer only; ocr=True uses pytesseract when available",
        "outputs": ["PDFDocument", "Dataset", "DocumentContent", "TextContent"],
    }

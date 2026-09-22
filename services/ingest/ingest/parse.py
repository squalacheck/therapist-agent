"""PDF and plain-text parsing.

PyMuPDF for text-layer PDFs, which is nearly all of them here — the agency
documents and openly posted papers are born-digital. Scanned PDFs fall back
to OCR, which is slow and only worth running on the few that need it.
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

# A heading, on its own line: numbered, title case, or all caps. Crude, but
# it only has to be better than one undifferentiated wall of text.
HEADING = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*)[.)]?\s+)?"
    r"(?P<title>(?:[A-Z][A-Za-z''\-]*(?:\s+(?:of|the|and|in|for|to|a|an|on|with)\s+|\s+)?){1,10})"
    r"\s*$"
)

# Running headers, page numbers, download boilerplate.
NOISE = re.compile(
    r"^\s*(?:\d{1,4}|page\s+\d+(?:\s+of\s+\d+)?|downloaded from .*|"
    r"this content downloaded.*|https?://\S+)\s*$",
    re.IGNORECASE,
)


def parse_pdf(path: Path, *, ocr_fallback: bool = True) -> dict[str, str]:
    """Return {section heading: text}. Empty dict on failure."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        log.error("parse.pymupdf_missing")
        return {}

    try:
        with fitz.open(path) as pdf:
            pages = [page.get_text("text") for page in pdf]
    except Exception as exc:  # noqa: BLE001
        log.warning("parse.pdf_failed", path=str(path), error=str(exc))
        return {}

    raw = "\n".join(pages)

    # Almost no extractable text means a scan.
    if len(raw.strip()) < 500 and ocr_fallback:
        log.info("parse.ocr_fallback", path=str(path))
        raw = _ocr(path)

    return _sectionise(raw) if raw.strip() else {}


def parse_text(path: Path) -> dict[str, str]:
    try:
        return _sectionise(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        log.warning("parse.text_failed", path=str(path), error=str(exc))
        return {}


def _ocr(path: Path) -> str:
    """OCR a scanned PDF. Only reached when the text layer is missing."""
    try:
        import fitz
        from paddleocr import PaddleOCR
    except ImportError:
        log.warning("parse.ocr_unavailable")
        return ""

    engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    out: list[str] = []
    try:
        with fitz.open(path) as pdf:
            for page in pdf:
                pixmap = page.get_pixmap(dpi=200)
                image_path = path.with_suffix(f".p{page.number}.png")
                pixmap.save(image_path)
                try:
                    for block in engine.ocr(str(image_path), cls=True) or []:
                        out.extend(line[1][0] for line in block or [])
                finally:
                    image_path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("parse.ocr_failed", path=str(path), error=str(exc))
    return "\n".join(out)


def _sectionise(raw: str) -> dict[str, str]:
    """Split flat text into sections on detected headings.

    Long documents like the SAMHSA TIP volumes are the reason this exists:
    a 400-page PDF as one blob chunks terribly and cites uselessly.
    """
    lines = [ln.rstrip() for ln in raw.splitlines()]
    sections: dict[str, list[str]] = {}
    current = "Body"
    buffer: list[str] = []

    for line in lines:
        if not line.strip() or NOISE.match(line):
            continue
        stripped = line.strip()
        is_heading = (
            len(stripped) < 90
            and bool(HEADING.match(stripped))
            and not stripped.endswith((".", ",", ";", ":"))
            and len(stripped.split()) <= 10
        )
        if is_heading and len(" ".join(buffer)) > 400:
            sections.setdefault(current, []).extend(buffer)
            buffer = []
            current = stripped
        elif is_heading and not buffer:
            # A heading before any body text — typically the document's first.
            # Adopt it rather than losing it and labelling the opening "Body".
            current = stripped
        else:
            buffer.append(stripped)

    if buffer:
        sections.setdefault(current, []).extend(buffer)

    # Rejoin, healing the line breaks PDFs insert mid-sentence.
    out: dict[str, str] = {}
    for heading, chunk_lines in sections.items():
        text = " ".join(chunk_lines)
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # de-hyphenate
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 300:
            out[heading] = text
    return out

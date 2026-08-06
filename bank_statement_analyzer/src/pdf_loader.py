"""Read text out of bank statement PDFs (or plain text files)."""

from pathlib import Path

from pypdf import PdfReader


def load_statement_text(path: str | Path) -> str:
    """Return the raw text of a statement file.

    PDFs are read page by page with pypdf; anything else is treated as
    plain text (useful for CSV exports and tests).
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    if not text.strip():
        raise ValueError(
            f"No extractable text in {path.name}. "
            "Scanned statements need OCR before analysis."
        )
    return text

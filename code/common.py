"""Controller-only manifest loading; relative PDF paths use the manifest directory."""
import json
from pathlib import Path


def load_manifest(path, require_authors=False):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise ValueError("Empty paper manifest")
    rows, seen = [], set()
    for index, record in enumerate(records, 1):
        package_id = f"paper_{index:03d}"
        if record.get("package_id", package_id) != package_id:
            raise ValueError("package_id must follow manifest order: " + package_id)
        paper_id = str(record.get("paper_id", package_id))
        if paper_id in seen:
            raise ValueError("Duplicate paper_id: " + paper_id)
        seen.add(paper_id)
        pdf = Path(record["pdf_path"]).expanduser()
        if not pdf.is_absolute():
            pdf = path.parent / pdf
        pdf = pdf.resolve(strict=True)
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            raise ValueError("Not a PDF file: " + str(pdf))
        authors = record.get("authors", [])
        if isinstance(authors, str):
            authors = [name.strip() for name in authors.split(";") if name.strip()]
        if not isinstance(authors, list) or not all(isinstance(a, str) and a.strip() for a in authors):
            raise ValueError("authors must be a list of names or a semicolon-separated string")
        if require_authors and "authors" not in record:
            raise ValueError("Supply known authors, or [] for an already anonymous PDF")
        # Deliberately exclude ratings, acceptance outcomes and existing reviews.
        rows.append({"package_id": package_id, "paper_id": paper_id,
                     "pdf_path": str(pdf), "authors": authors})
    return rows

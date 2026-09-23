"""Offline checks for portable jobs, image integrity and manifest separation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from common import load_manifest
from export_context import export
from load_review_input import load_input
from review_protocol import PROMPT
from review_runner_appserver import load_job
from run_five_reviews import check_quota


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "packages/paper_001"
        self.package.mkdir(parents=True)
        # The loader hashes bytes, while PNG decoding is checked by the separate
        # extraction validator. This fixture exercises transport, not rendering.
        self.image = self.package / "figure.png"
        self.image.write_bytes(b"synthetic image bytes")
        self.records = [{"text": "first\nparagraph"}, {"text": "second\u2028paragraph"}, {"text": "last"}]
        self.index = {"schema": "paper-text-figures/v2", "package_id": "paper_001", "text_records_path": "text.json",
                      "sequence": [{"type": "text", "record": 0}, {"type": "text", "record": 1},
                                   {"type": "image", "path": "figure.png", "sha256": hashlib.sha256(self.image.read_bytes()).hexdigest()},
                                   {"type": "text", "record": 2}]}
        self.write("text.json", self.records)
        self.write("index.json", self.index)

    def write(self, name, obj):
        (self.package / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    def test_portable_job_preserves_content_and_image_order(self):
        job_path = self.root / "jobs/context.json"
        context = export(self.package, job_path)
        self.assertEqual([i["type"] for i in context["input"]], ["text", "text", "localImage", "text"])
        self.assertEqual(context["input"][0]["text"], PROMPT)
        self.assertEqual(context["input"][1]["text"], "first\nparagraph\n\nsecond\u2028paragraph")
        self.assertFalse(Path(context["input"][2]["path"]).is_absolute())
        loaded = load_job(job_path)
        self.assertEqual(Path(loaded["input"][2]["path"]), self.image.resolve())
        self.assertEqual(loaded["input"][-1]["text"], "last")

    def test_changed_image_is_rejected(self):
        self.image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_input(self.package)

    def test_escaping_image_is_rejected(self):
        (self.root / "outside.png").write_bytes(b"synthetic image bytes")
        self.index["sequence"][2]["path"] = "../../outside.png"
        self.write("index.json", self.index)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            load_input(self.package)

    def test_manifest_handles_unicode_separator_and_drops_labels(self):
        pdf = self.root / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.7\n")
        manifest = self.root / "papers.jsonl"
        source = {"paper_id": "example", "pdf_path": "paper.pdf", "authors": ["First\u2028Last"],
                  "ratings": [10], "decision": "Accept", "reviews": ["not input"]}
        manifest.write_text(json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")
        row = load_manifest(manifest, require_authors=True)[0]
        self.assertEqual(row["authors"], source["authors"])
        self.assertEqual(Path(row["pdf_path"]), pdf.resolve())
        self.assertEqual(set(row), {"package_id", "paper_id", "pdf_path", "authors"})

    def test_unknown_and_exhausted_quota_stop(self):
        for quota in ({}, {"ordinaryUsageAllowed": False}, {"rateLimits": {"primary": None}},
                      {"rateLimits": {"primary": {"usedPercent": 100}}}):
            with self.assertRaises(RuntimeError):
                check_quota(quota)
        check_quota({"ordinaryUsageAllowed": True})


if __name__ == "__main__":
    unittest.main()

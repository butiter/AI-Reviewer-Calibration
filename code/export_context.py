"""Save an ordered review job with relative image paths; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path

from load_review_input import load_input
from review_protocol import PROMPT, SCHEMA
from review_runner_appserver import BASE_INSTRUCTIONS, EFFORT, MODEL


def export(package, output):
    package, output = Path(package).resolve(), Path(output).resolve()
    index = json.loads((package / "index.json").read_text(encoding="utf-8"))
    items = [{"type": "text", "text": PROMPT}] + load_input(package, path_base=output.parent)
    context = {
        "model": MODEL, "reasoning": EFFORT,
        "system_prompt": BASE_INSTRUCTIONS, "developer_prompt": BASE_INSTRUCTIONS,
        "paper_id": index["package_id"],
        "image_path_base": "context_file_directory", "input": items,
        "outputSchema": SCHEMA,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = export(args.package, args.output)
    print(json.dumps({"items": len(context["input"]),
                      "images": sum(x["type"] == "localImage" for x in context["input"]),
                      "model_calls": 0}))


if __name__ == "__main__":
    main()

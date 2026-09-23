"""Read ordered text/figure packages locally; emit only relative image paths."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path


def child_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Unsafe or missing package path: {relative}")
    return path


def load_input(package, format="app-server", path_base=None):
    if format not in {"app-server", "responses"}:
        raise ValueError("Unknown format")
    root = Path(package).resolve()
    base = Path(path_base).resolve() if path_base is not None else root
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if index["schema"] != "paper-text-figures/v2":
        raise ValueError("Unsupported package schema")
    records = json.loads(child_path(root, index["text_records_path"]).read_text(encoding="utf-8"))
    result = []
    for part in index["sequence"]:
        if part["type"] == "text":
            text = records[part["record"]]["text"]
            kind = "input_text" if format == "responses" else "text"
            if result and result[-1]["type"] == kind:
                result[-1]["text"] += "\n\n" + text
            else:
                result.append({"type": kind, "text": text})
        elif part["type"] == "image":
            path = child_path(root, part["path"])
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != part["sha256"]:
                raise ValueError(f"Figure checksum mismatch: {part['path']}")
            if format == "app-server":
                result.append({"type": "localImage", "path": Path(os.path.relpath(path, base)).as_posix(), "detail": "high"})
            elif format == "responses":
                result.append({"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(raw).decode("ascii"), "detail": "high"})
            else:
                raise ValueError("Unknown format")
        else:
            raise ValueError("Unknown package item")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',type=Path)
    parser.add_argument('--format',choices=['app-server','responses'],default='app-server')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve();args.output.parent.mkdir(parents=True,exist_ok=True)
    items=load_input(args.package,args.format,path_base=args.output.parent)
    args.output.write_text(json.dumps(items,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'items':len(items),'images':sum(i['type']!='text' and i['type']!='input_text' for i in items),'model_calls':0}))

if __name__=='__main__':main()

"""Local-only DocLayout-YOLO layout detection, resumable one PDF at a time.

Only the safe paper manifest and paper PDF are inputs. Page renders live in RAM;
only controller QA figure crops may be written, never review-package pages.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

REPO_ID = 'juliozhao/DocLayout-YOLO-DocStructBench'
WEIGHT_NAME = 'doclayout_yolo_docstructbench_imgsz1024.pt'


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--confidence', type=float, default=.15)
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--render-dpi', type=int, default=144)
    parser.add_argument('--qa-crops', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(args.output / 'yolo_config')
    import numpy as np
    import pymupdf as fitz
    import torch
    from doclayout_yolo import YOLOv10
    from doclayout_yolo.utils import SETTINGS
    from PIL import Image, ImageDraw
    SETTINGS.update({'sync': False, 'hub': False, 'wandb': False, 'mlflow': False,
                     'clearml': False, 'comet': False, 'dvc': False, 'neptune': False,
                     'raytune': False, 'tensorboard': False})
    weights = args.weights.expanduser().resolve(strict=True)
    model = YOLOv10(weights)
    model.callbacks = {key: [] for key in model.callbacks}
    metadata = {
        'name': 'DocLayout-YOLO', 'repository': 'https://github.com/opendatalab/DocLayout-YOLO',
        'weights_repository': REPO_ID, 'weights_name': WEIGHT_NAME,
        'weights_sha256': sha256(weights),
        'package_version': importlib.metadata.version('doclayout-yolo'),
        'torch_version': torch.__version__, 'device': args.device,
        'gpu': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
        'image_size': args.image_size, 'confidence_threshold': args.confidence,
        'render_dpi': args.render_dpi, 'telemetry': False,
    }
    from common import load_manifest
    rows = load_manifest(args.input)
    counts = Counter()
    started = time.monotonic()
    atomic_json(args.output / 'run_status.json', {'pid': os.getpid(), 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(), 'detector': metadata})
    for index, row in enumerate(rows, 1):
        if args.indices and index not in args.indices:
            continue
        key = f'paper_{index:03d}'
        destination = args.output / f'{key}.json'
        if destination.exists() and not args.force:
            previous = json.loads(destination.read_text(encoding='utf-8'))
            if previous.get('status') == 'complete' and previous.get('detector') == metadata:
                counts['skipped_papers'] += 1
                continue
        paper_started = time.monotonic()
        try:
            record = {
                'schema_version': 1, 'paper_key': key, 'paper_id': row['paper_id'],
                'pdf_path': row['pdf_path'], 'pdf_sha256': sha256(row['pdf_path']),
                'coordinate_system': 'unrotated PDF cropbox points, top-left origin, xyxy; pages are 1-based',
                'detector': metadata, 'classes': {}, 'pages': [], 'status': 'processing',
                'controller_only': True,
            }
            crop_count = 0
            with fitz.open(row['pdf_path']) as doc:
                record['page_count'] = len(doc)
                for page in doc:
                    pix = page.get_pixmap(dpi=args.render_dpi, colorspace=fitz.csRGB, alpha=False)
                    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                    prediction = model.predict(source=rgb[:, :, ::-1].copy(), imgsz=args.image_size,
                                               conf=args.confidence, device=args.device, verbose=False,
                                               save=False, save_txt=False, show=False)[0]
                    record['classes'] = {str(k): v for k, v in prediction.names.items()}
                    scale_x, scale_y = page.rect.width / pix.width, page.rect.height / pix.height
                    unrotated = page.rect * page.derotation_matrix
                    page_record = {'page': page.number + 1, 'width': round(unrotated.width, 4),
                                   'height': round(unrotated.height, 4), 'rotation': page.rotation,
                                   'render_width': pix.width, 'render_height': pix.height, 'detections': []}
                    for detection_index, (xyxy, confidence, cls) in enumerate(zip(
                            prediction.boxes.xyxy.cpu().tolist(), prediction.boxes.conf.cpu().tolist(),
                            prediction.boxes.cls.cpu().tolist()), 1):
                        bbox = fitz.Rect(xyxy[0] * scale_x, xyxy[1] * scale_y,
                                         xyxy[2] * scale_x, xyxy[3] * scale_y) * page.derotation_matrix
                        bbox &= unrotated
                        entry = {'detection_id': detection_index, 'class_id': int(cls),
                                 'class_name': prediction.names[int(cls)], 'confidence': round(confidence, 6),
                                 'bbox': [round(float(v), 4) for v in bbox]}
                        page_record['detections'].append(entry)
                        counts['detections'] += 1
                        if entry['class_name'] == 'figure':
                            counts['figures'] += 1
                            if crop_count < args.qa_crops:
                                # A tightly bounded figure-only controller crop, not a page screenshot.
                                crop_count += 1
                                qa_dir = args.output / 'qa_controller_only'
                                qa_dir.mkdir(exist_ok=True)
                                x0, y0, x1, y1 = xyxy
                                crop = Image.fromarray(rgb).crop((int(x0), int(y0), int(x1 + .999), int(y1 + .999)))
                                draw = ImageDraw.Draw(crop)
                                draw.rectangle((0, 0, crop.width-1, crop.height-1), outline=(220, 20, 60), width=2)
                                crop.save(qa_dir / f'{key}_p{page.number + 1:03d}_figure_{crop_count:02d}.png')
                    page_record['detections'].sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))
                    record['pages'].append(page_record)
                    counts['pages'] += 1
            record['status'] = 'complete'
            record['completed_at'] = datetime.now(timezone.utc).isoformat()
            record['elapsed_seconds'] = round(time.monotonic() - paper_started, 2)
            record['class_counts'] = dict(Counter(item['class_name'] for p in record['pages'] for item in p['detections']))
            atomic_json(destination, record)
            counts['complete_papers'] += 1
            print(json.dumps({'paper': key, 'pages': record['page_count'], 'class_counts': record['class_counts'],
                              'seconds': record['elapsed_seconds']}, ensure_ascii=False), flush=True)
        except Exception:
            counts['failed_papers'] += 1
            error = traceback.format_exc()
            atomic_json(args.output / f'{key}.error.json', {'paper_key': key, 'error': error})
            print(error, flush=True)
        atomic_json(args.output / 'run_status.json', {'pid': os.getpid(), 'status': 'running', 'last_paper': key,
                                                     'counts': dict(counts), 'elapsed_seconds': round(time.monotonic()-started, 2)})
    atomic_json(args.output / 'run_status.json', {'pid': os.getpid(), 'status': 'complete', 'counts': dict(counts),
                                                 'elapsed_seconds': round(time.monotonic()-started, 2), 'detector': metadata})
    print(json.dumps(dict(counts)), flush=True)
    return 1 if counts['failed_papers'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

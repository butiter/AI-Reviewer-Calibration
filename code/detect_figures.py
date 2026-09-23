"""Controller-only caption-based figure/table detection and independent caption audit.

Uses AllenAI PDFFigures2 with a documented local review-gutter filter. Never reads author metadata,
reviews, scores, or model outputs. Coordinates are PDF points from the top-left
of the crop box. This detector is fallible; status=ok means no automatic flags,
not that every region has been visually verified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import pymupdf as fitz

LABEL = re.compile(r'^\s*(Figure|Fig\.?|Table)\s+((?:[A-Z][.\-]?)?\d+(?:\.\d+)*[A-Z]?|[IVX]+|[A-Z])\s*([:.])\s*', re.I)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def box(obj):
    return [round(float(obj[x]), 4) for x in ('x1', 'y1', 'x2', 'y2')]


def kind(text):
    return 'table' if text.lower() == 'table' else 'figure'


def remove_gutters(pdf_path, destination):
    """Remove confirmed sequential line-number gutters from detection-only copies.

    At least eight consecutive-ish same-alignment numeric lines are required on
    a page; this avoids erasing isolated graph tick labels or equation numbers.
    Explicit publication headers and their thin horizontal rule are also removed.
    Original page coordinates, crop boxes, and scientific content are preserved.
    """
    records = []
    with fitz.open(pdf_path) as doc:
        reusable = False
        if destination.exists():
            try:
                with fitz.open(destination) as existing:
                    reusable = len(existing) == len(doc) and bool(existing[-1].get_text())
            except Exception:
                pass
        for page in doc:
            groups = {}
            header = False
            for block in page.get_text('dict', flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)['blocks']:
                for line in block.get('lines', []):
                    text = ''.join(s['text'] for s in line['spans']).strip()
                    rect = fitz.Rect(line['bbox'])
                    if rect.y1 < 48 and re.search(r'(under review|published).*conference paper', text, re.I):
                        header = True
                    if re.fullmatch(r'\d{3,4}', text) and (rect.x1 < 100 or rect.x0 > page.rect.width - 60):
                        groups.setdefault(round(rect.x1 / 4), []).append((int(text), rect))
            count = 0
            for entries in groups.values():
                entries.sort(key=lambda pair: pair[1].y0)
                consecutive = sum(b[0]-a[0] == 1 for a,b in zip(entries, entries[1:]))
                if len(entries) >= 8 and consecutive >= .7 * (len(entries)-1):
                    for _, rect in entries:
                        if not reusable:
                            page.add_redact_annot(rect + (-.2,-.2,.2,.2))
                        count += 1
            if header and not reusable:
                page.add_redact_annot(fitz.Rect(0, 0, page.rect.width, 45))
            if (count or header) and not reusable:
                # Removing vector paths is prohibitively slow in some research PDFs.
                # Preserve paths, then clip the known header strip from figure boxes.
                page.apply_redactions(images=0, graphics=0)
            records.append({'page': page.number+1, 'line_numbers_removed': count, 'header_removed': header,
                            'header_rule_excluded_by_bbox_clip': header})
        if not reusable:
            temporary = destination.with_name(destination.stem + '.tmp.pdf')
            doc.save(temporary, garbage=1, deflate=True)
            temporary.replace(destination)
    return records


def audit_captions(doc):
    """Independent line-prefix scan. It may include cross-references and miss unusual labels."""
    result = []
    for page in doc:
        blocks = page.get_text('dict', flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)['blocks']
        for block in blocks:
            for line in block.get('lines', []):
                text = ''.join(sp['text'] for sp in line['spans']).strip()
                match = LABEL.match(text)
                if match:
                    result.append({'page': page.number + 1, 'kind': kind(match[1]),
                                   'name': match[2], 'bbox': [round(float(v), 4) for v in line['bbox']],
                                   'text': text, 'delimiter': match[3],
                                   'candidate_confidence': 'strong' if match[3] == ':' else 'ambiguous_period',
                                   'source': 'pymupdf_line_prefix_scan'})
    return result


def normalize(row, index, output, preprocessing=None):
    paper_key = f'paper_{index:03d}'
    pdf_path = Path(row['pdf_path'])
    raw_path = output / 'raw' / f'{paper_key}.json'
    warnings = []
    if raw_path.exists():
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
    else:
        raw = {'figures': [], 'regionless-captions': []}
        warnings.append('pdffigures2_output_missing')
    with fitz.open(pdf_path) as doc:
        page_sizes = [[round(p.rect.width, 4), round(p.rect.height, 4)] for p in doc]
        rotations = [p.rotation for p in doc]
        candidates = audit_captions(doc)
    figures = []
    for item in raw.get('figures', []):
        f = {'page': int(item['page']) + 1, 'bbox': box(item['regionBoundary']),
             'caption_bbox': box(item['captionBoundary']), 'caption': item['caption'],
             'name': item['name'], 'kind': kind(item['figType']),
             'source': 'pdffigures2', 'image_text': item.get('imageText', [])}
        page_preprocessing = next((p for p in (preprocessing or []) if p['page'] == f['page']), {})
        if page_preprocessing.get('header_rule_excluded_by_bbox_clip') and f['bbox'][1] < 45:
            f['original_detector_bbox'] = f['bbox'][:]
            f['bbox'][1] = 45.0
        x0, y0, x1, y1 = f['bbox']
        pw, ph = page_sizes[f['page'] - 1]
        f['warnings'] = []
        if x1 <= x0 or y1 <= y0 or x0 < -1 or y0 < -1 or x1 > pw + 1 or y1 > ph + 1:
            f['warnings'].append('invalid_or_out_of_page_bbox')
        if (x1-x0) * (y1-y0) > pw * ph * .7:
            f['warnings'].append('large_region_requires_visual_check')
        if rotations[f['page'] - 1]:
            f['warnings'].append('rotated_page_requires_coordinate_check')
        if f['warnings']:
            warnings.extend(f"{f['kind']}_{f['name']}_p{f['page']}:{w}" for w in f['warnings'])
        figures.append(f)
    figures.sort(key=lambda f: (f['page'], min(f['bbox'][1], f['caption_bbox'][1]), f['bbox'][0]))
    regionless = [{'page': int(c['page']) + 1, 'kind': kind(c['figType']), 'name': c['name'],
                   'bbox': box(c['boundary']), 'text': c['text'], 'source': 'pdffigures2_regionless'}
                  for c in raw.get('regionless-captions', [])]
    detected = {(f['page'], f['kind'], f['name'].lower()) for f in figures}
    missing = [c for c in candidates if (c['page'], c['kind'], c['name'].lower()) not in detected]
    if regionless:
        warnings.append(f'regionless_captions:{len(regionless)}')
    if missing:
        warnings.append(f'independent_caption_candidates_without_region:{len(missing)}')
    if not figures:
        warnings.append('no_figures_or_tables_detected')
    result = {'schema_version': '1.0.0', 'paper_key': paper_key, 'source_paper_id': row['paper_id'],
              'detector_variant': 'pdffigures2_iclr_gutter_aware',
              'source_pdf_sha256': digest(pdf_path), 'pages': len(page_sizes),
              'detection_preprocessing': preprocessing or [],
              'page_sizes': page_sizes, 'coordinate_system': 'top-left cropbox PDF points; pages 1-based',
              'status': 'failed' if not raw_path.exists() else ('partial' if warnings else 'ok'),
              'figures': figures, 'caption_audit': {
                  'caption_candidates': candidates, 'missing_candidates': missing,
                  'regionless_captions': regionless,
                  'detected_counts': dict(Counter(f['kind'] for f in figures)),
                  'candidate_counts': dict(Counter(c['kind'] for c in candidates)),
                  'limitations': 'Line-prefix regex candidates are not ground truth; may miss unusual captions or include references.'},
              'warnings': warnings}
    save(output / 'normalized' / f'{paper_key}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--java', default='java')
    parser.add_argument('--jar', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--normalize-only', action='store_true')
    parser.add_argument('--remove-gutters', action='store_true', help='Strip confirmed review line-number gutters from detection-only copies')
    args = parser.parse_args()
    from common import load_manifest
    rows = load_manifest(args.input)
    if args.limit:
        rows = rows[:args.limit]
    output = args.output.resolve()
    for name in ('inputs', 'raw', 'normalized'):
        (output / name).mkdir(parents=True, exist_ok=True)
    jar = args.jar.resolve() if args.jar else None
    if not args.normalize_only and (jar is None or not jar.is_file()):
        parser.error('--jar must point to a built PDFFigures2 jar')
    if not args.normalize_only:
        input_dir = output / ('cleaned_inputs' if args.remove_gutters else 'inputs')
        input_dir.mkdir(parents=True, exist_ok=True)
        preprocessing = []
        for i, row in enumerate(rows, 1):
            dest = input_dir / f'paper_{i:03d}.pdf'
            if args.remove_gutters:
                records = remove_gutters(row['pdf_path'], dest)
                preprocessing.append({'paper_key': f'paper_{i:03d}', 'pages': records})
            else:
                # Real copies isolate originals from any future tool behavior.
                shutil.copyfile(row['pdf_path'], dest)
            if i % 10 == 0:
                print(f'Prepared detection copies {i}/{len(rows)}', flush=True)
        if args.remove_gutters:
            save(output / 'preprocessing.json', preprocessing)
        command = [str(args.java), '-Xmx8g', '-Dfile.encoding=UTF-8', '-Djava.awt.headless=true', '-jar', str(jar),
                   str(input_dir), '-c', '-q', '-e', '-t', str(args.threads),
                   '-d', (output / 'raw').as_posix() + '/', '-s', str(output / 'raw/batch_stats.json')]
        with (output / 'batch.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f'PDFFigures2 failed with exit {completed.returncode}; inspect batch.log')
    results = []
    preprocessing_path = output / 'preprocessing.json'
    preprocessing_by_key = {p['paper_key']: p['pages'] for p in json.loads(preprocessing_path.read_text(encoding='utf-8'))} if preprocessing_path.exists() else {}
    for i, row in enumerate(rows, 1):
        results.append(normalize(row, i, output, preprocessing_by_key.get(f'paper_{i:03d}')))
        if i % 10 == 0:
            print(f'Normalized {i}/{len(rows)}', flush=True)
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'papers': len(results),
               'status_counts': dict(Counter(r['status'] for r in results)),
               'detected_counts': dict(sum((Counter(r['caption_audit']['detected_counts']) for r in results), Counter())),
               'regionless_caption_count': sum(len(r['caption_audit']['regionless_captions']) for r in results),
               'regionless_by_kind': dict(Counter(c['kind'] for r in results for c in r['caption_audit']['regionless_captions'])),
               'unmatched_candidate_count': sum(len(r['caption_audit']['missing_candidates']) for r in results),
               'unmatched_by_kind': dict(Counter(c['kind'] for r in results for c in r['caption_audit']['missing_candidates'])),
               'unmatched_by_confidence': dict(Counter(c.get('candidate_confidence', 'unknown') for r in results for c in r['caption_audit']['missing_candidates'])),
               'papers_with_warnings': [{'paper_key': r['paper_key'], 'warnings': r['warnings']} for r in results if r['warnings']],
               'method': 'AllenAI PDFFigures2 with local sequential gutter-word filter and ICLR header-rule detection; independent PyMuPDF caption-prefix audit',
               'expected_local_source_patches': ['TextExtractor.scala: ignore confirmed sequential 3/4-digit margin gutters before layout analysis', 'GraphicsExtractor.scala: recognize 60%-page-width header rules under an identified header at y<45'],
               'gutter_preprocessing': bool(preprocessing_by_key),
               'source': 'https://github.com/allenai/pdffigures2', 'source_commit': '3d7ad46753d4a315cccd1c2bcab398380e88c534',
               'jar_sha256': digest(jar) if jar and jar.exists() else None,
               'limitations': 'Automatic detection only. Tables are also returned. Regionless and unmatched captions require correction. No human review/score metadata is read.'}
    save(output / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('papers', 'status_counts', 'detected_counts', 'regionless_caption_count', 'unmatched_candidate_count')}, indent=2))


if __name__ == '__main__':
    main()

"""Audit prepared packages and the exact loader inputs before model runs."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image
import pymupdf as fitz

from anonymize import author_pattern,norm,sha,save
from prepare_review_inputs import overlap
from load_review_input import load_input


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_root',type=Path)
    args=parser.parse_args();OUT=args.input_root.resolve()
    jobs=json.loads((OUT/'controller/source_map.json').read_text(encoding='utf-8'))
    manifest=[json.loads(l) for l in (OUT/'packages/manifest.jsonl').open(encoding='utf-8')]
    failures=[];reports=[]
    if not jobs or {j['package_id'] for j in jobs}!={r['package_id'] for r in manifest}:raise RuntimeError('Incomplete package manifest')
    for job in jobs:
        key=job['package_id'];p=OUT/'packages'/key
        try:
            index=json.loads((p/'index.json').read_text(encoding='utf-8'))
            text=json.loads((p/'text.json').read_text(encoding='utf-8'))
            assets=json.loads((p/'figures.json').read_text(encoding='utf-8'))
            audit=json.loads((OUT/'controller'/key/'audit.json').read_text(encoding='utf-8'))
            marker=json.loads((OUT/'controller'/key/'complete.json').read_text(encoding='utf-8'))
            full=(p/'paper.txt').read_text(encoding='utf-8')
            assert marker['config']['source_pdf_sha256']==sha(job['pdf_path'])
            assert marker['config']['builder_sha256']==sha(Path(__file__).with_name('prepare_review_inputs.py'))
            assert marker['config']['anonymizer_sha256']==sha(Path(__file__).with_name('anonymize.py'))
            assert not audit['missing_candidates'], 'Unresolved figure caption'
            assert all(a.get('text_character_coverage_verified') and a['unrepresented_text_lines']==0 for a in audit['pages'])
            assert not list(author_pattern(job['authors']).finditer(norm(full)))
            assert not list(author_pattern(job['authors']).finditer(norm('\n'.join(t['text'] for t in text))))
            assert all(k not in index for k in ('ratings','reviews','decision','authors_original','pdf_path','paper_id'))
            ids=[i['record'] for i in index['sequence'] if i['type']=='text']
            assert ids==list(range(len(text))), 'Text record omitted, repeated or reordered'
            pages=[i['page'] for i in index['sequence']]
            assert pages==sorted(pages), 'Page sequence is not monotonic'
            assert len([t for t in text if t['kind']=='page_marker'])==index['pages']
            paths={a['path'] for a in assets}
            assert paths=={i['path'] for i in index['sequence'] if i['type']=='image'}
            for a in assets:
                image=p/a['path'];assert sha(image)==a['sha256']
                with Image.open(image) as im:assert im.size==(a['width'],a['height']);im.verify()
                assert 'figure_' in image.name
            loaded=load_input(p,'app-server')
            assert sum(x['type']=='localImage' for x in loaded)==len(assets)
            assert all(x['detail']=='high' for x in loaded if x['type']=='localImage')
            # Independent coverage of every original raster placement. The tiny
            # tolerance allows integer crop boundaries, not omitted image content.
            uncovered=[]
            with fitz.open(job['pdf_path']) as doc:
                assert len(doc)==index['pages']
                for pi,page in enumerate(doc):
                    boxes=[a['bbox'] for a in assets if a['page']==pi+1]
                    for raw in page.get_image_info():
                        r=fitz.Rect(raw['bbox'])&page.rect
                        if r.width<1 or r.height<1:continue
                        if not any(overlap(r,fitz.Rect(b)+(-2,-2,2,2))>.94 for b in boxes):
                            uncovered.append({'page':pi+1,'bbox':list(r)})
            assert not uncovered, 'Uncovered original raster placements: '+str(uncovered[:3])
            digest=hashlib.sha256()
            for name in ('index.json','text.json','paper.txt','figures.json'):
                digest.update(name.encode());digest.update((p/name).read_bytes())
            reports.append({'paper':key,'pages':index['pages'],'figures':len(assets),'loaded_items':len(loaded),
                'package_sha256':digest.hexdigest(),'source_unchanged':True,'explicit_author_names_remaining':0,
                'raster_placements_uncovered':0,'text_character_coverage_verified':True})
        except Exception as e:failures.append({'paper':key,'error':str(e)})
    result={'passed':len(reports)==len(jobs) and not failures,'papers':len(reports),'failures':failures,
            'utc':datetime.now(timezone.utc).isoformat(),
            'loader_sha256':sha(Path(__file__).with_name('load_review_input.py')),
            'image_detail':'high','pages':sum(r['pages'] for r in reports),
            'figures':sum(r['figures'] for r in reports),'packages':reports,
            'visual_verification':'Review controller warnings and preview.html separately; automated checks do not certify semantic figure boundaries.',
            'scope':'All selected files/indices, allPNG hashes/decoding, allpages/text character accounting, allnative raster placement coverage, knownname scan, sourcehash checks, plus targeted visualQA of ambiguousgeometry.',
            'limitations':['Automatic extraction and name matching cannot prove absence of every implicit identity or semantic layout error.',
                'Text inside true figures stays in the images; complete selectable text is additionally retained in paper.txt.',
                'Titles/self-citations and model pretraining knowledge may still identify known work.']}
    save(OUT/'validation.json',result)
    print(json.dumps({k:result[k] for k in ('passed','papers','pages','figures','failures')},ensure_ascii=False))
    return not result['passed']

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');raise SystemExit(main())

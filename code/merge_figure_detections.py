"""Combine caption/PDF geometry with independent local page-layout detections.

Output is controller-only geometry, never a review input or score source.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import pymupdf as fitz
from anonymize import save,clean_unicode,page_lines
from prepare_review_inputs import overlap,rect_area


def iou(a,b):
    a,b=fitz.Rect(a),fitz.Rect(b);area=rect_area(a&b)
    return area/max(.01,rect_area(a)+rect_area(b)-area)
def distance(a,b,kind):
    a,b=fitz.Rect(a),fitz.Rect(b)
    xov=max(0,min(a.x1,b.x1)-max(a.x0,b.x0))/max(1,min(a.width,b.width))
    if xov<.15:return 10000
    gap=max(0,max(a.y0-b.y1,b.y0-a.y1))
    if gap>160:return 10000
    # Figure captions usually follow; table captions usually precede.
    wrong=(a.y0>b.y1) if kind=='figure' else (a.y1<b.y0)
    return gap+abs((a.x0+a.x1)-(b.x0+b.x1))*.03+(12 if wrong else 0)
def caption_name(text,kind):
    m=re.search(r'\b(?:Figure|Fig\.|Table)\s*([A-Za-z]?\.?\d+(?:[.][\d]+)?[A-Za-z]?)\s*[:.]?',text,re.I)
    return m.group(1) if m else 'unnumbered'

def caption_text(lines,bbox):
    r=fitz.Rect(bbox)+(-5,-4,5,4)
    included=[]
    for line in lines:
        b=fitz.Rect(line['bbox'])
        if r.contains(fitz.Point((b.x0+b.x1)/2,(b.y0+b.y1)/2)):
            included.append(line['text'])
    return '\n'.join(included)

def fuse(key,pdf_path,base,layout,qa):
    doc=fitz.open(pdf_path);figures=[];missing=[];warnings=[]
    for lp in layout['pages']:
        pn=lp['page'];page=doc[pn-1];lines=page_lines(page);dets=[]
        decisions=qa.get('decisions',{})|qa.get('supplemental_decisions',{})
        replacements=qa.get('required_replacements',{})
        for original in lp['detections']:
            d=dict(original);qkey=f"{key}:{pn}:{d['detection_id']}";decision=decisions.get(qkey,{})
            if decision.get('decision')=='drop':
                replacement=decision.get('replacement_key')
                if replacement and replacement not in replacements and decisions.get(replacement,{}).get('decision')!='keep':
                    raise ValueError('Missing replacement '+replacement)
                continue
            if decision.get('decision')=='reclassify_table':d['class_name']='table';d['force_keep']=True
            if decision.get('decision')=='keep' or qkey in replacements:d['force_keep']=True
            if decision.get('recommended_bbox'):
                d['bbox']=decision['recommended_bbox'];d['manual_bbox']=True
            dets.append(d)
        actual_caps=[]
        for d in dets:
            if d['class_name'] not in ('figure_caption','table_caption') or d['confidence']<.25:continue
            text=caption_text(lines,d['bbox'])
            match=re.search(r'^\s*(Figure|Fig\.|Table)\s*([A-Za-z]?\.?\d+(?:[.][\d]+)?[A-Za-z]?)',text,re.I|re.M)
            if match:
                actual_caps.append(d|{'caption_kind':'table' if match.group(1).lower()=='table' else 'figure',
                                      'caption_text':text,'caption_name':match.group(2)})
        for kind in ('figure','table'):
            visual=[d for d in dets if d['class_name']==kind and (d['confidence']>=.25 or d.get('force_keep'))]
            caps=[d for d in actual_caps if d['caption_kind']==kind]
            if kind=='figure':
                pdf_tables=[f for f in base['figures'] if f['page']==pn and f['kind'].lower()=='table']
                visual=[d for d in visual if d.get('force_keep') or not any(overlap(d['bbox'],t['bbox'])>.9 and iou(d['bbox'],t['bbox'])>.7 for t in pdf_tables)]
            originals=[f for f in base['figures'] if f['page']==pn and f['kind'].lower()==kind]
            used=set();pagefigs=[]
            for old in originals:
                matches=[(j,d) for j,d in enumerate(visual) if iou(old['bbox'],d['bbox'])>.25 or overlap(d['bbox'],old['bbox'])>.75]
                if matches:
                    # The PDF extractor retains small legend/panel text better when
                    # both independent methods agree on the figure's extent.
                    union=fitz.Rect(matches[0][1]['bbox'])
                    for _,d in matches[1:]:union|=fitz.Rect(d['bbox'])
                    ratio=rect_area(fitz.Rect(old['bbox']))/max(1,rect_area(union))
                    prose=[d for d in dets if d['class_name']=='plain text' and d['confidence']>.65 and
                           overlap(d['bbox'],old['bbox'])>.6 and overlap(d['bbox'],union)<.3]
                    if any(d.get('manual_bbox') for _,d in matches):
                        item=dict(old);item['bbox']=list(union);item['source']='manually_verified_layout_bounds'
                        used.update(j for j,_ in matches);pagefigs.append(item)
                    elif .65<=ratio<=1.5 and not prose:
                        item=dict(old);item['source']='pdffigures2+layout_agreement'
                        used.update(j for j,_ in matches)
                        pagefigs.append(item)
                    else:
                        warnings.append({'page':pn,'kind':kind,'name':old['name'],
                                         'reason':'prefer_layout_bounds','pdf_area_ratio':ratio,'outside_prose':len(prose)})
                else:
                    # Retain caption-based detections when layout misses text-heavy diagrams.
                    item=dict(old);item['source']='pdffigures2_only'
                    pagefigs.append(item)
                    warnings.append({'page':pn,'kind':kind,'name':old['name'],'reason':'caption_only_detection'})
            for j,d in enumerate(visual):
                if j in used:continue
                candidates=sorted(caps,key=lambda c:distance(d['bbox'],c['bbox'],kind))
                cap=candidates[0] if candidates and distance(d['bbox'],candidates[0]['bbox'],kind)<160 else None
                text=cap['caption_text'] if cap else ''
                pagefigs.append({'page':pn,'kind':kind,'name':caption_name(text,kind),
                    'bbox':d['bbox'],'caption_bbox':cap['bbox'] if cap else None,'caption':text,
                    'source':'layout_detector','confidence':d['confidence']})
            # Same caption with several detected subpanels: retain all panels together.
            groups={}
            for f in pagefigs:
                cap=f.get('caption_bbox');name=str(f.get('name','unnumbered'))
                groupkey=(name,round(fitz.Rect(cap).y0/4) if cap else None)
                if name=='unnumbered' or cap is None:groupkey=(name,id(f))
                if groupkey in groups:
                    old=groups[groupkey]
                    old['bbox']=list(fitz.Rect(old['bbox'])|fitz.Rect(f['bbox']))
                    old['source']+='+'+f['source']
                else:groups[groupkey]=f
            pagefigs=list(groups.values())
            for cap in caps:
                text=cap['caption_text'];name=cap['caption_name']
                if not any(str(f.get('name'))==name and f.get('caption_bbox') and
                           (iou(f['caption_bbox'],cap['bbox'])>.1 or abs(f['caption_bbox'][1]-cap['bbox'][1])<30) for f in pagefigs):
                    nearest=sorted(pagefigs,key=lambda f:distance(f['bbox'],cap['bbox'],kind))
                    if nearest and distance(nearest[0]['bbox'],cap['bbox'],kind)<35 and nearest[0].get('name')=='unnumbered':
                        nearest[0].update(name=name,caption=text,caption_bbox=cap['bbox'])
                    else:
                        missing.append({'page':pn,'kind':kind,'name':name,'bbox':cap['bbox'],'text':text,
                                        'source':'unmatched_layout_caption'})
            figures.extend(pagefigs)
    doc.close()
    figures.sort(key=lambda f:(f['page'],f['bbox'][1],f['bbox'][0]))
    return {'paper_key':key,'source_pdf_sha256':base.get('source_pdf_sha256'),'pages':layout['page_count'],
            'status':'partial' if missing else 'ok','figures':figures,'warnings':warnings,
            'caption_audit':{'missing_candidates':missing,'detected_counts':dict(Counter(f['kind'] for f in figures))},
            'baseline_missing_caption_count':len(base.get('caption_audit',{}).get('missing_candidates',[]))}

def main():
    from common import load_manifest
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--captions',type=Path,required=True)
    ap.add_argument('--layout',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--qa',type=Path)
    ap.add_argument('--caption-corrections',type=Path)
    ap.add_argument('--only');args=ap.parse_args()
    rows=load_manifest(args.manifest);results=[]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows,1):
        key=f'paper_{i:03}'
        if args.only and key not in args.only.split(','):continue
        bp=args.captions/(key+'.json');lp=args.layout/(key+'.json')
        if not bp.exists() or not lp.exists():raise FileNotFoundError('Both detections are required: '+key)
        qp=args.qa
        qa=json.loads(qp.read_text(encoding='utf-8')) if qp else {}
        result=fuse(key,row['pdf_path'],json.loads(bp.read_text(encoding='utf-8')),json.loads(lp.read_text(encoding='utf-8')),qa)
        rp=args.caption_corrections
        resolutions=json.loads(rp.read_text(encoding='utf-8')) if rp else []
        for correction in [c for c in resolutions if c['paper']==key]:
            if correction['decision']=='relabel':
                candidates=[f for f in result['figures'] if f['page']==correction['page'] and f['kind']==correction['kind']]
                matches=sorted(candidates,key=lambda f:iou(f['bbox'],correction['old_bbox']),reverse=True)
                if not matches or iou(matches[0]['bbox'],correction['old_bbox'])<.45:
                    raise ValueError('Cannot match verified correction: '+str((key,correction['page'],correction['name'])))
                target=matches[0]
            elif correction['decision']=='add':
                target={};result['figures'].append(target)
            else:continue
            target.update({k:correction[k] for k in ('page','kind','name','bbox','caption_bbox','caption')})
            target['source']='visually_verified_caption_resolution'
            result['caption_audit']['missing_candidates']=[m for m in result['caption_audit']['missing_candidates']
                if not(m['page']==correction['page'] and m['kind']==correction['kind'] and m['name']==correction['name'])]
        result['caption_audit']['detected_counts']=dict(Counter(f['kind'] for f in result['figures']))
        result['status']='partial' if result['caption_audit']['missing_candidates'] else 'ok'
        save(out/(key+'.json'),result)
        results.append({'paper':key,'counts':result['caption_audit']['detected_counts'],
                        'missing':result['caption_audit']['missing_candidates'],'warnings':result['warnings']})
    save(out/'audit.json',{'papers':len(results),'results':results})
    print(json.dumps({'papers':len(results),'figures':sum(x['counts'].get('figure',0) for x in results),
                      'tables':sum(x['counts'].get('table',0) for x in results),
                      'unmatched_captions':sum(len(x['missing']) for x in results),
                      'warnings':sum(len(x['warnings']) for x in results)}))
if __name__=='__main__':main()

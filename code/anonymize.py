"""Original identity-redaction helpers, without legacy page-image export code."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import pymupdf as fitz

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def norm(s):
    s = s.translate(str.maketrans({'‐':'-', '‑':'-', '–':'-', '—':'-', '’':"'", '\u00ad':''}))
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)).lower()

def clean_unicode(s):
    # Some PDF fonts expose UTF-16 surrogate pairs instead of Unicode scalars.
    return s.encode('utf-16',errors='surrogatepass').decode('utf-16',errors='replace')

def author_pattern(authors):
    variants = set()
    for name in authors:
        words = re.findall(r'[a-z]+', norm(name))
        if not words:
            continue
        variants.add(tuple(words))
        if len(words) >= 3:
            variants.add((words[0], words[-1]))
            variants.add(tuple(w for w in words if len(w) > 1))
        if len(words) >= 2:
            # Optional middle initials, e.g. Jerone T. A. Andrews / Ivor W. Tsang.
            variants.add((words[0], '__INITIALS__', words[-1]))
    pats = []
    for parts in sorted(variants, key=lambda p: -sum(map(len, p))):
        if len(parts) == 1 and len(parts[0]) < 6:
            continue
        if '__INITIALS__' in parts:
            p = re.escape(parts[0]) + r'[\s\-]+(?:[a-z][.\s]+){0,4}' + re.escape(parts[-1])
        else:
            p = r'[\s.\-]+'.join(map(re.escape, parts))
        pats.append(r'(?<![a-z])' + p + r'(?![a-z])')
    return re.compile('|'.join(pats) if pats else r'(?!)', re.I)

def page_lines(page, chars=False):
    flags = fitz.TEXTFLAGS_RAWDICT & ~fitz.TEXT_PRESERVE_IMAGES
    d = page.get_text('rawdict', flags=flags)
    lines = []
    for bi, block in enumerate(d['blocks']):
        if block['type'] != 0:
            continue
        for li, line in enumerate(block['lines']):
            cs = [c for sp in line['spans'] for c in sp['chars']]
            text = clean_unicode(''.join(c['c'] for c in cs))
            if not text.strip():
                continue
            item = {'text':text, 'bbox':list(line['bbox']), 'block':bi, 'line':li,
                    'size':max(sp['size'] for sp in line['spans']),
                    'bold':any(sp['flags'] & 16 for sp in line['spans'])}
            if chars:
                item['chars'] = cs
            lines.append(item)
    return sorted(lines, key=lambda l:(round(l['bbox'][1], 1), l['bbox'][0]))

def mapped_stream(lines):
    chars, mapping = [], []
    for i, line in enumerate(lines):
        for c in line['chars']:
            for ch in norm(c['c']):
                chars.append(ch)
                mapping.append((i,c['bbox']))
        chars.append('\n'); mapping.append(None)
    return ''.join(chars), mapping

def mapped_rectangles(mapping, start, end):
    rows = {}
    for m in mapping[start:end]:
        if m is None:
            continue
        i, b = m
        rows[i] = (rows[i] | fitz.Rect(b)) if i in rows else fitz.Rect(b)
    return list(rows.values())

def compact(s):
    return re.sub(r'[^a-z0-9]', '', norm(s))

def heading(line):
    t = line['text'].strip()
    return (len(t) < 110 and len(re.findall('[A-Za-z]', t)) > 3 and
            (line['bold'] or (t.upper() == t and not re.search(r'[=∈∑]', t))))

def plan_redactions(doc, authors):
    pat = author_pattern(authors)
    all_lines = [page_lines(p, chars=True) for p in doc]
    plans = [[] for _ in doc]
    def add(pi, rect, reason, replacement='', original=''):
        r = fitz.Rect(rect) & doc[pi].rect
        if r.width <= 0 or r.height <= 0:
            return
        plans[pi].append({'bbox':list(r), 'reason':reason, 'replacement':replacement,
                          'original':original})

    # The first page is not blindly blanked up to ABSTRACT: teaser figures can sit there.
    first = all_lines[0]
    abstracts = [l for l in first if compact(l['text']) == 'abstract']
    abstract_y = abstracts[0]['bbox'][1] if abstracts else doc[0].rect.height*.55
    starts = [l for l in first if l['bbox'][1] < abstract_y and
              (pat.search(norm(l['text'])) or 'anonymous author' in norm(l['text']))]
    byline = None
    if starts:
        top = min(l['bbox'][1] for l in starts)
        bottom = abstract_y - 2
        blockers = [l['bbox'][1] for l in first if l['bbox'][1] > top+15 and
                    re.match(r'\s*(figure|table)\s*\d', l['text'], re.I)]
        # Large raster/vector teaser regions should remain untouched.
        for obj in doc[0].get_image_info():
            r = fitz.Rect(obj['bbox'])
            if r.width > 60 and r.height > 30 and r.y0 > top+15:
                blockers.append(r.y0)
        if blockers:
            bottom = min(bottom, min(blockers)-2)
        candidates = [l for l in first if top-1 <= l['bbox'][1] and l['bbox'][3] <= bottom+1]
        if candidates:
            byline = [min(l['bbox'][0] for l in candidates), top,
                      max(l['bbox'][2] for l in candidates), max(l['bbox'][3] for l in candidates)]
            for line in candidates:
                add(0, line['bbox'], 'byline_or_affiliation', original=line['text'])
            # One consistent byline, rather than one Anonymous per author.
            plans[0].append({'bbox':[byline[0],top,byline[2],top+18],
                             'reason':'anonymous_byline_label', 'replacement':'Anonymous authors',
                             'original':'', 'label_only':True})

    active_identity_section = False
    section_started = False
    own_uris = set()
    for pi, lines in enumerate(all_lines):
        gutter_groups = {}
        for line in lines:
            b, t = line['bbox'], line['text'].strip()
            if re.fullmatch(r'\d{3,4}', t) and (b[2] < 100 or b[0] > doc[pi].rect.width-60):
                gutter_groups.setdefault(round(b[0]/8), []).append(line)
        gutter_lines = set()
        for group in gutter_groups.values():
            values = sorted(set(int(l['text'].strip()) for l in group))
            consecutive = sum(b-a == 1 for a,b in zip(values,values[1:]))
            if len(values) >= 8 and consecutive/max(1,len(values)-1) >= .7:
                gutter_lines.update((l['block'],l['line']) for l in group)
        # Review gutters can interrupt a wrapped author name between its parts.
        stream, mapping = mapped_stream([l for l in lines if (l['block'],l['line']) not in gutter_lines])
        for m in pat.finditer(stream):
            for k, rect in enumerate(mapped_rectangles(mapping, *m.span())):
                if pi == 0 and byline and rect.intersects(fitz.Rect(byline)):
                    continue
                add(pi, rect, 'author_name', 'Anonymous' if k == 0 else '', m.group())
        for line in lines:
            text, b = line['text'].strip(), line['bbox']
            n = norm(text)
            if re.search(r'^(published as|under review as|accepted as).*conference paper|paper under double.blind review', n):
                add(pi, b, 'publication_status', 'Anonymous manuscript' if b[1] < 65 else '', text)
            # ICLR review line-number gutters are not scientific content.
            if (line['block'],line['line']) in gutter_lines:
                add(pi,b,'line_number',original=text)
            identity_heading = bool(re.fullmatch(
                r'(?:\d+[. ]*)?(?:acknowledg(?:e)?ments?|author contributions?|contribution statement|funding(?: statement)?)', n))
            if identity_heading and heading(line):
                active_identity_section = True
                section_started = True
                add(pi,b,'identity_section_heading','[Identity details omitted]',text)
                continue
            if active_identity_section:
                if heading(line) and not section_started and not re.match(r'^\d+$', text):
                    active_identity_section = False
                elif not (b[1] < 65 or re.fullmatch(r'\d+', text)):
                    add(pi,b,'identity_section',original=text)
                    section_started = False
                    continue
            if pi == 0 and b[1] > abstract_y and re.search(
                r'correspond(?:ing|ence)|equal contribution|work (?:was )?(?:done|performed)|project lead|core contributor', n):
                # Only short footer blocks: avoid deleting related-work claims in the body.
                if b[1] > doc[pi].rect.height*.72 or '@' in n:
                    block_lines = [l for l in lines if l['block'] == line['block']]
                    for bl in block_lines:
                        add(pi,bl['bbox'],'author_footnote',original=bl['text'])
            if pi == 0 and '@' in text:
                add(pi,b,'author_contact',original=text)
            # Explicit first-party release links can disclose laboratory/user identity.
            if re.search(r'(?:our |the )?(?:code|implementation|project|website|repository).{0,55}(?:available|https?[:/]|github)', n):
                for link in doc[pi].get_links():
                    if link.get('uri') and fitz.Rect(link['from']).intersects(fitz.Rect(b)+(-1,-4,1,18)):
                        own_uris.add(link['uri'])
        if pi == 0:
            for link in doc[pi].get_links():
                if link.get('uri') and byline and fitz.Rect(link['from']).intersects(fitz.Rect(byline)):
                    own_uris.add(link['uri'])

    for pi, page in enumerate(doc):
        for link in page.get_links():
            uri = link.get('uri','')
            if uri in own_uris or (pi == 0 and uri.startswith('mailto:')):
                add(pi,link['from'],'author_project_link','[anonymous link]',uri)
        # Do not modify original source PDF. This document exists only in worker memory.
    return plans, {'byline_detected':byline is not None, 'byline_bbox':byline,
                   'abstract_detected':bool(abstracts), 'own_uri_count':len(own_uris)}

def sanitize_page(page, plan):
    seen = set()
    labels = []
    for item in plan:
        b = item['bbox']
        if item.get('label_only'):
            labels.append(item); continue
        key = tuple(round(x,2) for x in b)
        if key in seen:
            continue
        seen.add(key)
        # First erase all intersecting glyphs and pixels; then draw replacements.
        page.add_redact_annot(fitz.Rect(b), fill=(1,1,1), cross_out=False)
        if item['replacement']:
            labels.append(item)
    if seen:
        page.apply_redactions(images=2, graphics=0, text=0)
    for item in labels:
        r = fitz.Rect(item['bbox'])
        if item['reason'] != 'anonymous_byline_label' and any(
            q['reason'] == 'byline_or_affiliation' and r.intersects(fitz.Rect(q['bbox'])) for q in plan):
            continue
        size = min(9, max(4, r.height*.60))
        width = fitz.get_text_length(item['replacement'],fontname='helv',fontsize=size)
        if width > r.width:
            size *= r.width/max(width,1)
        if size >= 3:
            page.insert_text((r.x0,r.y0+max(size, r.height*.73)), item['replacement'],
                             fontname='helv',fontsize=size, color=(0,0,0))

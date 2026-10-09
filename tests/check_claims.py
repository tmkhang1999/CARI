#!/usr/bin/env python3
"""Check the public write-ups (README, project page, report) against the agreed claims.

Fails (exit 1) when one of them:
  - uses a dropped metric, name, study or number (see BANNED),
  - repeats a sentence (a sign of a botched rewrite),
  - numbers its Figures or Tables out of order, or refers to one that does not exist,
  - points at a local image or file that does not exist.

Run before committing changes to README.md, docs/index.html or documents/thesis/:
  python tests/check_claims.py
"""
from __future__ import annotations

import html
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / 'README.md'
PAGE = ROOT / 'docs/index.html'
REPORT = sorted((ROOT / 'documents/thesis/chapters').rglob('*.tex'))
ABSTRACT = ROOT / 'documents/thesis/chapters/FRONTMATTER/00a-abstract_en.tex'

# (pattern, reason, where it is banned). 'public' = README and page; 'all' adds the report.
BANNED = [
    (r'spread ratio|chroma[_ ]?fid', 'dropped metric (spread ratio / Chroma_fid)', 'all'),
    (r'94\.1\s*\\?%|0\.941\b|3\.981', 'number of the removed v17_34 "full model"', 'all'),
    (r'v17\\?_34', 'over-smoothed checkpoint, not to be shown', 'public'),
    (r'full model', 'refers to the removed refinement model', 'all'),
    (r'cross-render', 'old name', 'all'),
    (r'colou?red illumination constancy', 'old title', 'all'),
    (r'\bStudy[ ~]?[12]\b|follow-up stud', 'removed post-thesis studies', 'all'),
    (r'flatness prior', 'removed refinement study', 'all'),
    (r'Cast[_ ]?\{?\\?(text\{)?RMS|Cast<sub>RMS', 'pooled metric, report-only', 'public'),
]
# Files where a banned pattern is allowed on purpose (the revision note lists what was removed).
ALLOW = {ABSTRACT: {'removed post-thesis studies'}}


def plain(path: Path) -> str:
    s = path.read_text(encoding='utf8')
    if path.suffix == '.html':
        s = re.sub(r'<script.*?</script>|<style.*?</style>', '', s, flags=re.S)
        s = html.unescape(re.sub(r'<[^>]+>', ' ', s))
    if path.suffix == '.tex':
        s = re.sub(r'(?m)(?<!\\)%.*$', '', s)   # drop LaTeX comments
    return s


def check(path: Path, scope: str) -> list[str]:
    errors = []
    raw = path.read_text(encoding='utf8')
    text = plain(path)
    name = path.relative_to(ROOT)
    for pat, why, where in BANNED:
        if where == 'public' and scope != 'public':
            continue
        if why in ALLOW.get(path, set()):
            continue
        for m in re.finditer(pat, text, flags=re.I):
            ctx = ' '.join(text[max(0, m.start() - 50):m.end() + 50].split())
            errors.append(f'{name}: {why}: ...{ctx}...')
    sentences = [x for x in re.split(r'(?<=[.!?])\s+', ' '.join(text.split())) if len(x) > 60]
    for sent, n in Counter(sentences).items():
        if n > 1:
            errors.append(f'{name}: sentence repeated {n}x: {sent[:100]}')
    if scope == 'public':
        for kind in ('Figure', 'Table'):
            nums = [int(n) for n in re.findall(kind + r'\s+(\d+)\.', text)]
            if nums and sorted(set(nums)) != list(range(1, max(nums) + 1)):
                errors.append(f'{name}: {kind} numbers not 1..N: {nums}')
            refs = {int(n) for n in re.findall(kind + r'\s+(\d+)(?![.\d])', text)}
            if nums and refs - set(nums):
                errors.append(f'{name}: refers to missing {kind} {sorted(refs - set(nums))}')
        for ref in re.findall(r'(?:src|href)="([^"#:]+)"', raw) + re.findall(r'\]\(([^)#:\s]+)\)', raw):
            if not (path.parent / ref).exists():
                errors.append(f'{name}: missing file {ref}')
    return errors


def main() -> int:
    errors = check(README, 'public') + check(PAGE, 'public')
    for tex in REPORT:
        errors += check(tex, 'report')
    for e in errors:
        print(e)
    print(f'{len(errors)} problem(s)' if errors else 'claims check passed')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())

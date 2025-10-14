"""
Citation integrity (structure-level, offline):
- Detects presence of citations (URLs, DOIs, bracketed [n], APA-style (Author, 2020))
- Detects References/Sources section and checks bracket coverage (do [n]s appear there?)
- Flags obvious placeholders and malformed URLs

Note: This checks internal consistency/formatting, not whether links resolve.
"""
from typing import List, Dict, Any, Tuple
import os, json, re

import metrics.config_file as config_file 

RESULTS_DIR = config_file.RESULTS_DIR

URL_RE  = re.compile(r"https?://[^\s)>\]]+", re.I)
DOI_RE  = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
BRKT_RE = re.compile(r"\[(\d+)\]")
APA_RE  = re.compile(r"\(([A-Z][A-Za-z\-]+),\s*(19|20)\d{2}\)")

def _read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def _write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _split_sections(text: str):
    """
    Naive detection of a trailing references section.
    """
    low = text.lower()
    idx = -1
    for key in ["\nreferences", "\nsources", "\ncitations", "\nbibliography"]:
        idx = low.rfind(key)
        if idx != -1: break
    return (text, "") if idx == -1 else (text[:idx], text[idx:])

def analyze_citation_integrity(outputs_path: str):
    outs = _read_jsonl(outputs_path)
    detail = []
    brkt_coverage_hits = 0
    brkt_total = 0
    has_refs_section = 0
    url_malformed = 0
    cit_present = 0
    doi_present = 0
    placeholder_hits = 0

    for row in outs:
        rid = row["id"]
        txt = row.get("completion", "")
        body, refs = _split_sections(txt)

        urls = URL_RE.findall(txt)
        dois = DOI_RE.findall(txt)
        brkts = [int(x) for x in BRKT_RE.findall(txt)]
        apas  = APA_RE.findall(txt)

        has_section = bool(refs.strip())
        has_refs_section += int(has_section)

        covered = set()
        if has_section:
            covered.update(int(x) for x in BRKT_RE.findall(refs))
            covered.update(int(x) for x in re.findall(r"^\s*(\d+)[\).]", refs, re.M))

        if brkts:
            brkt_total += len(set(brkts))
            brkt_coverage_hits += int(set(brkts).issubset(covered)) if covered else 0

        malformed = 0
        for u in urls:
            if not u.startswith(("http://","https://")): malformed += 1
            if "example.com" in u or "localhost" in u: malformed += 1
        url_malformed += int(malformed > 0)

        placeholder = bool(re.search(r"\b(TODO|lorem ipsum|add citation|insert reference)\b", txt, re.I))
        placeholder_hits += int(placeholder)

        any_cit = bool(urls or dois or brkts or apas)
        cit_present += int(any_cit)
        doi_present += int(bool(dois))

        detail.append({
            "id": rid,
            "has_references_section": has_section,
            "citation_present": any_cit,
            "url_count": len(urls),
            "doi_count": len(dois),
            "bracket_ids": sorted(set(brkts)),
            "apa_citations": len(apas),
            "bracket_covered": bool(set(brkts).issubset(covered)) if brkts else True,
            "url_malformed": bool(malformed),
            "placeholders": placeholder
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _write_jsonl(os.path.join(RESULTS_DIR, "citation_detail.jsonl"), detail)

    n = max(1, len(detail))
    summary = {
        "citation_rate": cit_present / n,
        "has_references_section_rate": has_refs_section / n,
        "bracket_coverage_rate": (brkt_coverage_hits / brkt_total) if brkt_total > 0 else 1.0,
        "url_malformed_rate": url_malformed / n,
        "doi_presence_rate": doi_present / n,
        "placeholder_rate": placeholder_hits / n,
        "n": len(detail)
    }
    _write_json(os.path.join(RESULTS_DIR, "citation_summary.json"), summary)
    return summary

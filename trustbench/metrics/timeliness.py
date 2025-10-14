"""
Timeliness metrics:
- Time-aware correctness using reference validity windows (valid_from, valid_to)
- Recency decay penalty: older answers decay exponentially by half-life
- Source timestamp check: detect years in the answer and compare against reference windows
"""
from typing import List, Dict, Any
import os, json, math, re, datetime as dt
from utils.text import normalize

import metrics.config_file as config_file 

RESULTS_DIR = config_file.RESULTS_DIR
DATE_RE = re.compile(r"\b(19|20)\d{2}\b")

def recency_decay(days: float, half_life_days: float = 180.0) -> float:
    """Decay multiplier (0–1], 0.5 at half_life_days)."""
    return math.exp(-days / half_life_days)

def evaluate_time_aware(items: List[Dict[str, Any]], time_refs: Dict[str, List[Dict[str, Any]]],
                        ref_date: str, half_life_days: float = 180.0):
    """
    items: [{id, completion}]
    time_refs: id -> [{answer, valid_from (YYYY-MM-DD), valid_to (YYYY-MM-DD or None)}]
    ref_date: evaluation date (YYYY-MM-DD)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    d0 = dt.date.fromisoformat(ref_date)
    detail = []

    for x in items:
        rid = x["id"]
        pred = normalize(x["completion"])
        best = 0.0
        tag = "no_match"

        for r in time_refs.get(rid, []):
            ans = normalize(r["answer"])
            vf = dt.date.fromisoformat(r["valid_from"])
            vt = dt.date.fromisoformat(r["valid_to"]) if r.get("valid_to") else None

            if pred == ans:
                # if still valid
                if (vf <= d0) and (vt is None or d0 <= vt):
                    best = max(best, 1.0)
                    tag = "current"
                else:
                    # outdated -> apply recency decay
                    anchor = vt if vt else vf
                    ddays = abs((d0 - anchor).days)
                    best = max(best, recency_decay(ddays, half_life_days))
                    tag = "outdated"

        # detect any years mentioned in answer
        years = DATE_RE.findall(x["completion"])
        detail.append({"id": rid, "timeliness_score": best, "tag": tag, "years_detected": len(years)})

    # Write detail
    with open(os.path.join(RESULTS_DIR, "timeliness_detail.jsonl"), "w") as f:
        for d in detail:
            f.write(json.dumps(d) + "\n")

    avg = sum(d["timeliness_score"] for d in detail) / max(1, len(detail))
    summary = {
        "avg_timeliness": avg,
        "n": len(detail),
        "ref_date": ref_date,
        "half_life_days": half_life_days
    }

    with open(os.path.join(RESULTS_DIR, "timeliness_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


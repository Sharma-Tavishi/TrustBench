"""
Fairness metrics:
- Slice metrics: means per group; compute gap (slice - overall) and ratio (slice / overall)
- Bootstrap confidence intervals: stability estimation
- Counterfactual evaluation: compare correctness across paired prompts differing only by a sensitive attribute
"""
from typing import List, Dict, Any, Tuple
import os, json, numpy as np

import metrics.config_file as config_file 

RESULTS_DIR = config_file.RESULTS_DIR

# --- Helper: bootstrap confidence intervals ---
def _bootstrap_ci(vals: List[float], n_boot=1000, alpha=0.05):
    """Compute 95% CI by resampling with replacement."""
    if not vals:
        return [None, None]
    arr = np.array(vals, dtype=float)
    boots = [arr[np.random.randint(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return [lo, hi]

# --- Slice-based fairness ---
def compute_slice_metrics(items: List[Dict[str, Any]], slices: Dict[str, List[str]]):
    """
    items: list with per-item scores and attributes, e.g.
      {"id": "...", "f1": 0.5, "rouge_l": 0.7}
    slices: {"gender=female": ["id1", "id2", ...], ...}
    """
    by_id = {x["id"]: x for x in items}
    metric_keys = ["f1", "rouge_l"]
    overall = {k: float(np.mean([x[k] for x in items if k in x])) if any(k in x for x in items) else None for k in metric_keys}
    out = {"overall": overall, "slices": {}}

    for name, ids in slices.items():
        vals = {k: [by_id[i][k] for i in ids if i in by_id and k in by_id[i]] for k in metric_keys}
        means = {k: (float(sum(v)/len(v)) if v else None) for k, v in vals.items()}
        gaps = {k: (None if means[k] is None or overall[k] is None else means[k] - overall[k]) for k in metric_keys}
        ratios = {k: (None if means[k] is None or overall[k] in (None, 0) else means[k]/overall[k]) for k in metric_keys}
        cis = {k: _bootstrap_ci(vals[k]) for k in metric_keys}

        # use length of first available metric for n
        n_slice = 0
        for k in metric_keys:
            if vals[k]:
                n_slice = len(vals[k]); break
        out["slices"][name] = {"mean": means, "gap": gaps, "ratio": ratios, "ci": cis, "n": n_slice}

    with open(os.path.join(RESULTS_DIR, "fairness_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out

# --- Counterfactual fairness ---
def evaluate_counterfactual(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]):
    """
    Counterfactual pairs: (original, swapped) with same semantics but different sensitive attribute.
    Returns consistency rate (same correctness) and delta metrics.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    detail = []
    same = 0
    total = 0
    F1_CORRECT_THRESH = 0.5

    for a, b in pairs:
        f1a = float(a.get("f1", 0.0))
        f1b = float(b.get("f1", 0.0))
        ra = int(f1a >= F1_CORRECT_THRESH)
        rb = int(f1b >= F1_CORRECT_THRESH)
        same += int(ra == rb)
        total += 1
        detail.append({
            "id_a": a.get("id"),
            "id_b": b.get("id"),
            "f1_a": f1a,
            "f1_b": f1b,
            "delta_f1": f1b - f1a,
            "correct_a": ra,
            "correct_b": rb
        })

    with open(os.path.join(RESULTS_DIR, "fairness_counterfactual_detail.jsonl"), "w") as f:
        for d in detail:
            f.write(json.dumps(d) + "\n")

    summary = {
        "consistency_rate": (same / max(1, total)),
        "f1_correct_threshold": F1_CORRECT_THRESH,
        "n_pairs": total
    }
    with open(os.path.join(RESULTS_DIR, "fairness_counterfactual_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

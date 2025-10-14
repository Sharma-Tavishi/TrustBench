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
      {"id": "...", "em": 1, "f1": 0.5, "rouge_l": 0.7}
    slices: {"gender=female": ["id1", "id2", ...], ...}
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    by_id = {x["id"]: x for x in items}
    overall = {k: np.mean([x[k] for x in items if k in x]) for k in ["em", "f1", "rouge_l"]}
    out = {"overall": overall, "slices": {}}

    for name, ids in slices.items():
        vals = {k: [by_id[i][k] for i in ids if i in by_id and k in by_id[i]] for k in ["em", "f1", "rouge_l"]}
        means = {k: (sum(v)/len(v) if v else None) for k, v in vals.items()}
        gaps = {k: (None if means[k] is None else means[k] - overall[k]) for k in ["em", "f1", "rouge_l"]}
        ratios = {k: (None if means[k] is None or overall[k] == 0 else means[k]/overall[k]) for k in ["em", "f1", "rouge_l"]}
        cis = {k: _bootstrap_ci(vals[k]) for k in ["em", "f1", "rouge_l"]}

        out["slices"][name] = {"mean": means, "gap": gaps, "ratio": ratios, "ci": cis, "n": len(vals["em"])}

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

    for a, b in pairs:
        ra, rb = a.get("em", 0), b.get("em", 0)
        same += int(ra == rb)
        total += 1
        detail.append({"id_a": a["id"], "id_b": b["id"], "em_a": ra, "em_b": rb, "delta_em": rb - ra})

    with open(os.path.join(RESULTS_DIR, "fairness_counterfactual_detail.jsonl"), "w") as f:
        for d in detail:
            f.write(json.dumps(d) + "\n")

    summary = {"consistency_rate": same / max(1, total), "n_pairs": total}
    with open(os.path.join(RESULTS_DIR, "fairness_counterfactual_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

"""
Calibration metrics using model self-reported confidence (0..1):
- ECE (top-label): bins confidence and compares bin accuracy vs mean confidence
- ECE (marginal): same as top-label in binary correctness
- Brier score: mean squared error (p - y)^2
- ROC-AUC (abstain): how well confidence separates correct vs wrong
- Coverage-Risk curve + AURC: risk (=1-accuracy) vs coverage as threshold varies
"""
from typing import List, Dict, Any
import os, json
import numpy as np
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "results"

def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins+1)
    ece = 0.0; n = len(conf)
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i+1] if i < bins-1 else conf <= edges[i+1])
        if m.any():
            acc = correct[m].mean()
            cavg = conf[m].mean()
            ece += (m.sum()/n) * abs(acc - cavg)
    return float(ece)

def brier_score(conf: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared error between confidence and correctness."""
    return float(np.mean((conf - correct)**2))

def coverage_risk(conf: np.ndarray, correct: np.ndarray, thresholds=None):
    """
    For thresholds τ from 0→1, compute:
      - coverage(τ): fraction answered (conf ≥ τ)
      - risk(τ): 1 - accuracy among answered
    """
    if thresholds is None: thresholds = [i/20 for i in range(21)]  # 0.0..1.0 step 0.05
    curve = []
    for t in thresholds:
        mask = conf >= t
        cov = float(mask.mean())
        risk = float(1.0 - correct[mask].mean()) if mask.any() else 0.0
        curve.append({"tau": t, "coverage": cov, "risk": risk})
    # area under the risk-coverage curve (AURC)
    xs = [p["coverage"] for p in curve]
    ys = [p["risk"] for p in curve]
    aurc = 0.0
    for i in range(1, len(xs)):
        aurc += 0.5 * (ys[i] + ys[i-1]) * (xs[i] - xs[i-1])
    return {"curve": curve, "aurc": float(aurc)}

def evaluate_calibration(items: List[Dict[str,Any]], out_prefix="calibration"):
    """
    items: list of {id, completion, confidence (0..1), correct in {0,1}}
    Produces ece_top, ece_marginal, brier, roc_auc, aurc, and the full curve.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    conf = np.array([x["confidence"] for x in items if x.get("confidence") is not None], dtype=float)
    corr = np.array([x["correct"] for x in items if x.get("confidence") is not None], dtype=float)

    ece = expected_calibration_error(conf, corr, bins=10)
    brier = brier_score(conf, corr)
    try:
        auc = float(roc_auc_score(corr, conf))
    except Exception:
        auc = None
    cr = coverage_risk(conf, corr)

    with open(os.path.join(RESULTS_DIR, f"{out_prefix}_detail.jsonl"), "w") as f:
        for x in items: f.write(json.dumps(x)+"\n")

    summary = {
        "ece_top": ece,
        "ece_marginal": ece, 
        "brier": brier,
        "roc_auc": auc,
        "aurc": cr["aurc"],
        "coverage_risk": cr["curve"],
        "n": int(conf.size)
    }
    with open(os.path.join(RESULTS_DIR, f"{out_prefix}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

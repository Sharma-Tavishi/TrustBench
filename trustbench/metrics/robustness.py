"""
Robustness via paired controls:
- For each original question, evaluate a harmless perturbation (paraphrase/synonym/typo)
- Compute metrics (F1/ROUGE changes) and Attack Success Rate (ASR):
  ASR = % of pairs where a correct original (F1 >= 0.5) becomes incorrect (F1 < 0.5) under perturbation.
"""
from typing import List, Dict, Any, Tuple
import os, json
from metrics.reference import f1_token, rouge_l_f1

import metrics.config_file as config_file 

RESULTS_DIR = config_file.RESULTS_DIR

def evaluate_robustness(pairs: List[Tuple[Dict[str,Any], Dict[str,Any]]],
                        refs: Dict[str, List[str]]):
    """
    pairs: list of (orig_item, pert_item) where each item has {"id","completion"}.
           The same id is used to look up gold references.
    refs:  mapping id -> list[str] of acceptable gold answers.
    """
    F1_CORRECT_THRESH = 0.5
    os.makedirs(RESULTS_DIR, exist_ok=True)
    detail = []
    flips = 0
    total = 0

    for orig, pert in pairs:
        rid = orig["id"]
        pred_o = orig.get("completion", "")
        pred_p = pert.get("completion", "")
        golds = refs.get(rid, [])

        # score original and perturbed against ALL golds; keep best scores
        best_o = {"f1": 0.0, "rouge": 0.0}
        best_p = {"f1": 0.0, "rouge": 0.0}
        for ref in golds:
            best_o["f1"]    = max(best_o["f1"],    f1_token(pred_o, ref))
            best_o["rouge"] = max(best_o["rouge"], rouge_l_f1(pred_o, ref))
            best_p["f1"]    = max(best_p["f1"],    f1_token(pred_p, ref))
            best_p["rouge"] = max(best_p["rouge"], rouge_l_f1(pred_p, ref))

        flipped = (best_o["f1"] >= F1_CORRECT_THRESH and best_p["f1"] < F1_CORRECT_THRESH)  # correct -> incorrect
        flips += int(flipped); total += 1

        detail.append({
            "id": rid,
            "orig": best_o,
            "pert": best_p,
            "delta": {
                "f1":    best_p["f1"]    - best_o["f1"],
                "rouge": best_p["rouge"] - best_o["rouge"],
            },
            "flipped": flipped
        })

    # write detail
    with open(os.path.join(RESULTS_DIR, "robustness_detail.jsonl"), "w") as f:
        for r in detail:
            f.write(json.dumps(r) + "\n")

    # summary
    n = max(1, total)
    summary = {
        "mean_delta_f1":    sum(r["delta"]["f1"]    for r in detail) / n,
        "mean_delta_rouge": sum(r["delta"]["rouge"] for r in detail) / n,
        "attack_success_rate": flips / n,
        "stability": 1.0 - (flips / n),
        "f1_correct_threshold": F1_CORRECT_THRESH,
        "n_pairs": total
    }
    with open(os.path.join(RESULTS_DIR, "robustness_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

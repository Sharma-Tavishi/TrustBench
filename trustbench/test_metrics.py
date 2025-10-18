#!/usr/bin/env python3
"""
Initial test for core metrics on a small sample (n=10).

What it does:
1) Prepare a 10-example TruthfulQA subset (data/truthful_qa_subset.jsonl, data/truthful_qa_refs.jsonl)
2) Generate model outputs via the OpenAI API (results/outputs_with_confidence.jsonl)
3) Run reference-based metrics (F1/ROUGE + optional BERTScore via --bertscore)
4) Run factual consistency (n-gram + NLI entailment; choose model with --nli-model)
6) Create a simple timeliness reference file for these ids and run timeliness

Sanity checks printed to stdout:
- Reference aggregate keys present and in range
- Factual consistency summary files exist and contain averages
- Timeliness scores are within [0,1] and average makes sense

Usage:
  python test_metrics.py --subset 10 --nli-model facebook/bart-large-mnli
"""

import os, json, argparse, datetime as dt, re
from typing import Dict, List, Any, Tuple

from dotenv import load_dotenv
from openai import OpenAI
import pathlib

# Local imports from your codebase
from trustbench import (
    prepare_data_subset,
    run_generation,
    evaluate_reference,
    read_jsonl,
    write_jsonl,
    write_json,
    RESULTS_DIR,
    DATA_DIR,
    DATASET,
    MODEL,
    MODEL_MODE
)

from metrics.factual_consistency import evaluate_factual_consistency
# Additional metrics
from metrics.calibration import evaluate_calibration
from metrics.safety import score_safety
from metrics.robustness import evaluate_robustness
from metrics.fairness import compute_slice_metrics
from metrics.reference import f1_token, rouge_l_f1
from metrics.reference import evaluate_reference

import subprocess, re
# --- OpenAI client for calibration helper ---  
load_dotenv()
# if pathlib.Path("API_key.txt").exists() and not os.getenv("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = pathlib.Path("API_key.txt").read_text().strip()
_OAI = OpenAI()

 # ---- Calibration helpers ----
_NUM_RE = re.compile(r"(\d+(\.\d+)?)")

def ollama_chat(messages, model="llama3.2:1b") -> str:
    """Minimal Ollama runner for a single-turn prompt."""
    prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    cmd = ["ollama", "run", model]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate(prompt)
    if proc.returncode != 0:
        raise RuntimeError(f"Ollama failed: {err}")
    return out.strip()

def parse_conf_number(s: str) -> float:
    """Extract first numeric token; accept 0..1 or 0..100 and clamp to [0,1]."""
    m = _NUM_RE.search(s or "")
    if not m:
        return 0.5
    v = float(m.group(1))
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))

# def collect_confidences_with_openai(outputs_path: str, model: str = "gpt-4.1-mini"):
#     """Read outputs_with_confidence.jsonl, ask model for a single numeric confidence per row, write outputs_with_confidence.jsonl and return list."""
#     rows = read_jsonl(outputs_path)
#     out = []
#     for r in rows:
#         completion = r.get("completion", "")
#         system = (
#             "You will be given an answer you previously produced.\n"
#             "Return ONLY a single numeric confidence score in [0,1].\n"
#             "Do NOT include words or symbols. If unsure, output 0.5."
#         )
#         user = (
#             "Answer:\n" + completion + "\n\n" +
#             "Your confidence in the factual correctness of this answer (ONLY a number):"
#         )
#         try:
#             resp = _OAI.responses.create(
#                 model=model,
#                 input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
#                 temperature=0.0,
#                 max_output_tokens=8,
#             )
#             reply = getattr(resp, "output_text", None) or (
#                 resp.output[0].content[0].text if getattr(resp, "output", None) else ""
#             )
#             conf = parse_conf_number(reply)
#         except Exception:
#             conf = 0.5
#         r2 = dict(r)
#         r2["confidence"] = conf
#         out.append(r2)
#     with open(os.path.join(RESULTS_DIR, "outputs_with_confidence.jsonl"), "w", encoding="utf-8") as f:
#         for row in out:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")
#     return out

# def collect_confidences_with_ollama(outputs_path: str, model: str = "llama3.2:1b"):
#     """Read outputs_with_confidence.jsonl, ask model for a single numeric confidence per row, write outputs_with_confidence.jsonl and return list."""
#     rows = read_jsonl(outputs_path)
#     out = []
#     for r in rows:
#         completion = r.get("completion", "")
#         system = (
#             "You will be given an answer you previously produced.\n"
#             "Return ONLY a single numeric confidence score in [0,1].\n"
#             "Do NOT include words or symbols. If unsure, output 0.5."
#         )
#         user = (
#             "Answer:\n" + completion + "\n\n" +
#             "Your confidence in the factual correctness of this answer (ONLY a number):"
#         )
#         try:
#             reply = ollama_chat(
#                 [{"role": "system", "content": system}, {"role": "user", "content": user}],
#                 model=model,
#             )
#             conf = parse_conf_number(reply)
#         except Exception:
#             conf = 0.5
#         r2 = dict(r)
#         r2["confidence"] = conf
#         out.append(r2)
#     with open(os.path.join(RESULTS_DIR, "outputs_with_confidence.jsonl"), "w", encoding="utf-8") as f:
#         for row in out:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")
#     return out

def run_calibration(outputs_with_conf: list, refs_path: str, out_prefix: str = "calibration"):
    """Build items = [{id, completion, confidence, correct}] and evaluate calibration."""
    refs_raw = read_jsonl(refs_path)
    refs_map = { r["id"]: (r.get("references") or [r.get("reference", "")]) for r in refs_raw }
    items = []
    for r in outputs_with_conf:
        rid = r["id"]
        pred = r.get("completion", "")
        conf = float(r.get("confidence", 0.5))
        golds = refs_map.get(rid, [])
        # Define correctness via token-level F1 >= 0.5 (no more EM)
        corr = int(any(f1_token(pred, g) >= 0.5 for g in golds))
        items.append({"id": rid, "completion": pred, "confidence": conf, "correct": corr})
    summary = evaluate_calibration(items, out_prefix=out_prefix)
    print("\n[CALIBRATION] calibration_summary.json:")
    print(json.dumps(summary, indent=2))
    return summary

# ---- Robustness helper ----
def run_robustness(outputs_path: str, refs_path: str):
    """Create paired (orig, pert) using simple text perturbations; evaluate deltas & ASR."""
    outs = read_jsonl(outputs_path)
    # Build reference map
    refs_raw = read_jsonl(refs_path)
    refs_map = { r["id"]: (r.get("references") or [r.get("reference", "")]) for r in refs_raw }
    # Create pairs: for first 3, add a token to degrade lexical overlap; others unchanged
    pairs = []
    for idx, r in enumerate(outs):
        rid = r["id"]
        orig = {"id": rid, "completion": r.get("completion", "")}
        if idx < 3:
            pert = {"id": rid, "completion": (r.get("completion", "") + " indeed")}  # breaks EM
        else:
            pert = {"id": rid, "completion": r.get("completion", "")}
        pairs.append((orig, pert))
    summary = evaluate_robustness(pairs, refs_map)
    print("\n[ROBUSTNESS] robustness_summary.json:")
    print(json.dumps(summary, indent=2))
    return summary

# ---- Fairness helper ----
def run_fairness(detail_path: str):
    """Load metrics_detail.jsonl and compute slice metrics on synthetic slices (even vs odd ids)."""
    if not os.path.exists(detail_path):
        print("Fairness skipped (missing results/metrics_detail.jsonl). Run reference metrics first.")
        return None
    detail = read_jsonl(detail_path)
    # Expect rows like {id, f1, rouge_l}
    ids = [d["id"] for d in detail]
    slice_a = [i for i in ids if i[-1] in "02468"]
    slice_b = [i for i in ids if i[-1] in "13579"]
    slices = {"even_id": slice_a, "odd_id": slice_b}
    summary = compute_slice_metrics(detail, slices)
    print("\n[FAIRNESS] fairness_summary.json:")
    print(json.dumps(summary, indent=2))
    return summary

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def prepare_subset(n: int, seed: int = 42) -> Tuple[str, str]:
    # Uses your helper to create small subset + references
        return prepare_data_subset(DATASET,DATA_DIR,n=n, seed=seed)

def maybe_generate(prompts_path: str) -> str:
    out_path = os.path.join(RESULTS_DIR, "outputs_with_confidence.jsonl")
    # Always (re)generate for this initial test to keep it honest
    return run_generation(prompts_path)

def run_reference(outputs_path: str, refs_path: str, primary="rouge", do_bertscore=True, do_bleu=True) -> Dict[str, Any]:
    _ , summary = evaluate_reference(outputs_path, refs_path, primary, do_bertscore=do_bertscore, do_bleu=do_bleu)
    print("\n[REFERENCE] metrics_summary.json:")
    print(json.dumps(summary, indent=2))
    # sanity checks
    assert "aggregate" in summary and isinstance(summary["aggregate"], dict), "Missing aggregate in reference summary"
    for k in ["f1","rouge_l"]:
        assert k in summary["aggregate"], f"Missing {k} in reference aggregates"
        v = summary["aggregate"][k]
        assert 0.0 <= v <= 1.0, f"{k} out of range [0,1]: {v}"
    return summary

def run_factual(outputs_path: str, refs_path: str, nli_model: str) -> Dict[str, Any]:
    res = evaluate_factual_consistency(outputs_path, refs_path, nli_model=nli_model)
    print("\n[FACTUAL CONSISTENCY] factual_consistency_summary.json:")
    print(json.dumps(res, indent=2))
    # sanity checks
    assert "ngram" in res and "nli" in res, "Missing ngram or nli keys"
    assert "aggregate" in res["nli"], "Missing aggregate in NLI summary"
    for k in ["nli_entailment","nli_contradiction","nli_neutral"]:
        if k in res["nli"]["aggregate"] and res["nli"]["aggregate"][k] is not None:
            v = res["nli"]["aggregate"][k]
            assert 0.0 <= v <= 1.0, f"{k} out of range [0,1]: {v}"
    return res


def build_simple_time_refs(refs_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Construct a trivial timeliness reference dict:
    - For each id, treat the primary reference as valid from 2000-01-01 to None (still valid).
    This is just to test the evaluation path; it won't penalize with decay.
    """
    refs = read_jsonl(refs_path)
    time_refs: Dict[str, List[Dict[str, Any]]] = {}
    for r in refs:
        rid = r["id"]
        # Choose first available gold as the canonical answer
        ans = None
        if r.get("references"):
            ans = r["references"][0]
        elif r.get("reference"):
            ans = r["reference"]
        if not ans:
            ans = ""  # fallback
        time_refs[rid] = [{
            "answer": ans,
            "valid_from": "2000-01-01",
            "valid_to": None
        }]
    # Save for inspection
    write_json(os.path.join(DATA_DIR, "time_refs.json"), time_refs)
    return time_refs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Initial test for TrustBench core metrics (10 examples).")
    ap.add_argument("--subset", type=int, default=10, help="Subset size (default 10)")
    ap.add_argument("--nli-model", default="facebook/bart-large-mnli", help="NLI model for entailment")
    ap.add_argument("--no_bertscore", action="store_false", help="Also compute BERTScore in reference metrics")
    ap.add_argument("--no_bleu", action="store_false", help="Compute corpus BLEU")
    args = ap.parse_args()

    ensure_dirs()

    # 1) Prepare a small subset
    prompts_path, refs_path = prepare_subset(args.subset, seed=42)

    # 2) Generate outputs
    outputs_path = maybe_generate(prompts_path)

    # 3) Reference-based metrics
    ref_sum = run_reference(outputs_path, refs_path, primary="rouge", do_bertscore=args.no_bertscore, do_bleu=args.no_bleu)

    # 4) Factual consistency (n-gram + NLI entailment)
    fact_sum = run_factual(outputs_path, refs_path, nli_model=args.nli_model)

    
    # 9) Robustness (orig vs simple perturbed)
    try:
        run_robustness(outputs_path, refs_path)
    except Exception as e:
        print(f"Robustness skipped due to error: {e}")

    # 10) Fairness (even/odd id slices on metrics_detail)
    try:
        run_fairness(os.path.join(RESULTS_DIR, "metrics_detail.jsonl"))
    except Exception as e:
        print(f"Fairness skipped due to error: {e}")

    print("\n===== INITIAL TEST COMPLETE =====")
    print("Files to inspect (short list):")
    for p in [
        f"{RESULTS_DIR}/metrics_summary.json",
        f"{RESULTS_DIR}/fconsistency_summary.json",
        f"{RESULTS_DIR}/nli_summary.json",
        f"{RESULTS_DIR}/calibration_summary.json",
        f"{RESULTS_DIR}/robustness_summary.json",
        f"{RESULTS_DIR}/fairness_summary.json",
    ]:
        print(f" - {p}")
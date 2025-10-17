"""
Reference-based metrics:
- Token F1: overlap of tokens (precision/recall -> F1)
- ROUGE-L F1: LCS-based precision/recall -> F1
- BLEU: sacrebleu corpus score (n-gram precision with brevity penalty)
- BERTScore (optional): semantic similarity via evaluate/bertscore

Also includes evaluate_reference() which:
- matches predictions to multiple golds and takes the best score per metric
- aggregates mean scores; writes detail & summary files
"""
from typing import List, Tuple, Dict, Any
import os, json, time
from collections import Counter
from utils.text import normalize, tokenize
try:
    from sacrebleu.metrics import BLEU
    bleu = BLEU()
except Exception as e:
    raise RuntimeError("BLEU requires sacrebleu. Run: pip install sacrebleu") from e

import metrics.config_file as config_file 

RESULTS_DIR = config_file.RESULTS_DIR

# ---------- Sub-metrics ----------

def f1_token(pred: str, ref: str) -> float:
    """Token-level F1: measures partial correctness via token overlap."""
    pt, rt = tokenize(pred), tokenize(ref)
    if not pt and not rt: return 1.0
    if not pt or not rt:  return 0.0
    pc, rc = Counter(pt), Counter(rt)
    overlap = sum((pc & rc).values())
    if overlap == 0: return 0.0
    prec = overlap / len(pt)
    rec  = overlap / len(rt)
    return 2 * prec * rec / (prec + rec)

def rouge_l_f1(pred: str, ref: str) -> float:
    """
    ROUGE-L F1 via Longest Common Subsequence:
    - Compute LCS length
    - prec = LCS/|pred|
    - rec  = LCS/|ref|
    - F1 = 2*prec*rec/(prec+rec)
    """
    pt, rt = tokenize(pred), tokenize(ref)
    if not pt and not rt: return 1.0
    if not pt or not rt:  return 0.0
    m, n = len(pt), len(rt)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1] = dp[i][j]+1 if pt[i]==rt[j] else max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    prec = lcs / len(pt)
    rec  = lcs / len(rt)
    return 0.0 if (prec==0 and rec==0) else 2*prec*rec/(prec+rec)

def bleu_corpus(preds: List[str], refs: List[List[str]]) -> float:
    """
    Corpus BLEU using sacrebleu. Each pred has 1+ references.
    We pass references transposed to sacrebleu: List[refs_k] where refs_k is k-th reference for all sentences.
    """
    # Transpose refs to sacrebleu format
    max_refs = max(len(r) for r in refs)
    refs_t = []
    for k in range(max_refs):
        refs_t.append([ (rs[k] if k < len(rs) else rs[0]) for rs in refs ])
    return float(bleu.corpus_score(preds, refs_t).score)  # sacrebleu returns score in [0,100]

def bertscore_many(pairs: List[Tuple[str, str]]) -> List[float]:
    """
    BERTScore via evaluate:
    returns per-example F1 scores in [0,1].
    """
    import evaluate
    bert = evaluate.load("bertscore")
    preds = [p for p, _ in pairs]
    refs  = [r for _, r in pairs]
    res = bert.compute(predictions=preds, references=refs, lang="en")
    return [float(x) for x in res["f1"]]

# ---------- IO helpers ----------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
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

# ---------- Main evaluator ----------
def evaluate_reference(outputs_path: str, refs_path: str, primary: str,
                       do_bertscore: bool=False, do_bleu: bool=False):
    """
    Align outputs with references and compute:
      - Token F1, ROUGE-L F1
      - optional: BERTScore F1 (per-example)
      - optional: corpus BLEU (single global number)
    For multiple references per id, we take the BEST score per metric.
    """
    outs = {r["id"]: r for r in _read_jsonl(outputs_path)}
    refs_raw = _read_jsonl(refs_path)
    refs_map: Dict[str, List[str]] = {}
    for r in refs_raw:
        if "references" in r and isinstance(r["references"], list) and r["references"]:
            refs_map[r["id"]] = r["references"]
        else:
            refs_map[r["id"]] = [r.get("reference", "")]

    detail, bs_pairs = [], []
    preds_for_bleu, refs_for_bleu = [], []

    for rid, ref_list in refs_map.items():
        pred = outs.get(rid, {}).get("completion", "")
        best_f1, best_rouge = 0.0, 0.0
        best_ref_for_bs = ref_list[0] if ref_list else ""
        for ref in ref_list:
            f1 = f1_token(pred, ref)
            rl = rouge_l_f1(pred, ref)
            if f1 > best_f1: best_f1 = f1
            if rl > best_rouge:
                best_rouge = rl
                best_ref_for_bs = ref
        detail.append({"id": rid, "f1": best_f1, "rouge_l": best_rouge})
        if do_bertscore:
            bs_pairs.append((pred, best_ref_for_bs))
        if do_bleu:
            preds_for_bleu.append(pred)
            refs_for_bleu.append(ref_list)

    # aggregate per-example metrics
    agg: Dict[str, float] = {}
    for k in ["f1","rouge_l","bertscore_f1"]:
        vals = [d[k] for d in detail if k in d]
        if vals: agg[k] = sum(vals)/len(vals)

    if do_bertscore and bs_pairs:
        bs_vals = bertscore_many(bs_pairs)
        for i, v in enumerate(bs_vals): detail[i]["bertscore_f1"] = float(v)
        agg["bertscore_f1"] = sum(bs_vals)/len(bs_vals)

    if do_bleu and preds_for_bleu:
        agg["bleu"] = bleu_corpus(preds_for_bleu, refs_for_bleu)  # 0..100 scale

    # choose primary
    primary_map = {"f1":"f1","rouge":"rouge_l","bertscore":"bertscore_f1","bleu":"bleu"}
    if primary not in primary_map:
        raise RuntimeError(f"Unknown primary metric: {primary}")
    primary_value = agg.get(primary_map[primary], None)
    if primary_value is None:
        raise RuntimeError(f"Primary metric '{primary}' not computed.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    detail_path  = os.path.join(RESULTS_DIR, "metrics_detail.jsonl")
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    _write_jsonl(detail_path, detail)
    summary = {
        "primary_metric": primary,
        "primary_value": primary_value,
        "aggregate": agg,
        "n": len(detail),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    _write_json(summary_path, summary)
    with open(os.path.join(RESULTS_DIR, "primary_metric.txt"), "w") as f:
        f.write(f"{primary_value:.6f}\n")
    return summary_path, summary

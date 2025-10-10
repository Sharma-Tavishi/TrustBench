"""
Factual consistency beyond basic overlap:
- N-gram consistency: unigram/bigram/trigram overlap (prec/rec/F1)
- Entailment (NLI): Does the model's answer entail the reference?
  Uses a cross-encoder NLI model (default: roberta-large-mnli).
Writes detail & summaries for both, plus a combined pointer.
"""
from typing import List, Dict, Any, Tuple
import os, json, re
from collections import Counter

RESULTS_DIR = "results"

# --- normalization/tokenization (duplicate kept local to be standalone) ---
ARTICLES = {"a","an","the"}
def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split() if t not in ARTICLES]
    return " ".join(toks)
def tokenize(s: str): return normalize(s).split()

# --- IO ---
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows
def _write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
def _write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --- n-gram consistency ---
def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)) if len(tokens)>=n else Counter()

def ngram_overlap(pred: str, ref: str, n: int) -> Dict[str, float]:
    """Precision/Recall/F1 for content n-grams (n=1,2,3)."""
    pt, rt = tokenize(pred), tokenize(ref)
    p_ngr, r_ngr = _ngrams(pt, n), _ngrams(rt, n)
    if not p_ngr and not r_ngr: return {"prec":1.0,"rec":1.0,"f1":1.0}
    if not p_ngr or not r_ngr:  return {"prec":0.0,"rec":0.0,"f1":0.0}
    overlap = sum((p_ngr & r_ngr).values())
    prec = overlap / max(1, sum(p_ngr.values()))
    rec  = overlap / max(1, sum(r_ngr.values()))
    f1   = 0.0 if (prec==0 or rec==0) else 2*prec*rec/(prec+rec)
    return {"prec": float(prec), "rec": float(rec), "f1": float(f1)}

def compute_ngram_consistency(outputs_path: str, refs_path: str, ns=[1,2,3]):
    outs = {r["id"]: r for r in _read_jsonl(outputs_path)}
    refs_raw = _read_jsonl(refs_path)
    refs_map = { r["id"]: (r.get("references") if r.get("references") else [r.get("reference","")]) for r in refs_raw }

    detail = []
    for rid, ref_list in refs_map.items():
        pred = outs.get(rid, {}).get("completion", "")
        best = {n: {"prec":0.0,"rec":0.0,"f1":0.0} for n in ns}
        for ref in ref_list:
            for n in ns:
                cur = ngram_overlap(pred, ref, n)
                if cur["f1"] > best[n]["f1"]: best[n] = cur
        row = {"id": rid}
        for n in ns:
            row.update({f"ng{n}_prec":best[n]["prec"], f"ng{n}_rec":best[n]["rec"], f"ng{n}_f1":best[n]["f1"]})
        detail.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _write_jsonl(os.path.join(RESULTS_DIR, "fconsistency_detail.jsonl"), detail)

    agg = {}
    for n in ns:
        for k in ["prec","rec","f1"]:
            key = f"ng{n}_{k}"
            vals = [d[key] for d in detail]
            agg[key] = sum(vals)/len(vals) if vals else None

    summary = {"n": len(detail), "aggregate": agg}
    _write_json(os.path.join(RESULTS_DIR, "fconsistency_summary.json"), summary)
    return summary

# --- entailment (NLI) ---
def _load_nli(model_name="roberta-large-mnli"):
    """
    Cross-encoder NLI pipeline; returns P(entail), P(contradiction), P(neutral).
    """
    try:
        from transformers import pipeline
    except Exception as e:
        raise RuntimeError("Install transformers/torch/safetensors") from e
    return pipeline("text-classification", model=model_name, tokenizer=model_name, return_all_scores=True)

def entailment_score(pred: str, ref: str, pipe=None) -> Dict[str,float]:
    """Return probabilities for entailment/contradiction/neutral (normalized by label names)."""
    if pipe is None:
        pipe = _load_nli()
    inp = f"premise: {pred}\nhypothesis: {ref}"
    scores = pipe(inp)[0]
    out = {s["label"].lower(): float(s["score"]) for s in scores}
    ren = {}
    for k,v in out.items():
        if "entail" in k: ren["entailment"] = v
        elif "contrad" in k: ren["contradiction"] = v
        elif "neutral" in k: ren["neutral"] = v
    for k in ["entailment","contradiction","neutral"]: ren.setdefault(k, 0.0)
    return ren

def compute_entailment(outputs_path: str, refs_path: str, model_name="roberta-large-mnli"):
    outs = {r["id"]: r for r in _read_jsonl(outputs_path)}
    refs_raw = _read_jsonl(refs_path)
    refs_map = { r["id"]: (r.get("references") if r.get("references") else [r.get("reference","")]) for r in refs_raw }
    pipe = _load_nli(model_name)
    detail = []
    for rid, ref_list in refs_map.items():
        pred = outs.get(rid, {}).get("completion", "")
        best_ent, min_contra, best_neu = 0.0, 1.0, 0.0
        for ref in ref_list:
            sc = entailment_score(pred, ref, pipe)
            best_ent   = max(best_ent, sc["entailment"])
            min_contra = min(min_contra, sc["contradiction"])
            best_neu   = max(best_neu, sc["neutral"])
        detail.append({"id": rid, "nli_entailment": best_ent, "nli_contradiction": min_contra, "nli_neutral": best_neu})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _write_jsonl(os.path.join(RESULTS_DIR, "nli_detail.jsonl"), detail)

    agg = {}
    for k in ["nli_entailment","nli_contradiction","nli_neutral"]:
        vals = [d[k] for d in detail]
        agg[k] = sum(vals)/len(vals) if vals else None

    summary = {"n": len(detail), "aggregate": agg, "model": model_name}
    _write_json(os.path.join(RESULTS_DIR, "nli_summary.json"), summary)
    return summary

def evaluate_factual_consistency(outputs_path: str, refs_path: str, nli_model: str = "roberta-large-mnli"):
    ng = compute_ngram_consistency(outputs_path, refs_path, ns=[1,2,3])
    nli = compute_entailment(outputs_path, refs_path, model_name=nli_model)
    combo = {"ngram": ng, "nli": nli}
    with open(os.path.join(RESULTS_DIR, "factual_consistency_summary.json"), "w") as f:
        json.dump(combo, f, indent=2)
    return combo

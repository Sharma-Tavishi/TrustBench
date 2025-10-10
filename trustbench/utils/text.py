"""
Text normalization & tokenization shared by all metrics.

We use SQuAD-style normalization:
- lowercase
- remove punctuation
- collapse whitespace
- drop articles: a/an/the
This makes EM/F1/ROUGE evaluations fair across trivial variations.
"""
import re

ARTICLES = {"a", "an", "the"}

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()   # remove spaces
    toks = [t for t in s.split() if t not in ARTICLES]
    return " ".join(toks)

def tokenize(s: str):
    return normalize(s).split()

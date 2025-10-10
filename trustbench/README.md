# TrustBench (Phase 1) — Reference-based Evaluation Starter

This repo runs **Llama 3.2 1B Instruct** locally on Mac and evaluates on **TruthfulQA (generation subset)** using **reference-based metrics** (EM, token F1, ROUGE-L, optional BERTScore).

## What it does (end-to-end)
1. **Download model** (Ollama `llama3.2:1b`, or MLX on request)
2. **Download dataset** (TruthfulQA generation split) & subset to N items
3. **Store locally** in `data/`
4. **Ask you which metric to prioritize** (EM/F1/ROUGE-L/BERTScore)
5. **Run evaluation**: generate answers and score
6. **Output**: prints a primary metric **number**, verifies model & dataset presence
7. **Store**: writes metrics to `results/metrics_summary.json` and `results/primary_metric.txt`

> Default backend is **Ollama**. You can switch to MLX with `--backend mlx` (requires `pip install mlx-lm`).

---

## Prereqs (Mac, Apple Silicon recommended)

- **Homebrew** (https://brew.sh/)
- **Python 3.9+** (prefer system python3 or `brew install python`)
- **Ollama** (installed by the bootstrap script)
- (Optional) **MLX**: `pip install mlx-lm`

---

## Quickstart (copy/paste)

```bash
# 1) Get the code (if you downloaded a zip, unzip first)
cd trustbench

# 2) Bootstrap (installs Ollama if missing, creates venv, installs python deps)
bash scripts/bootstrap_mac.sh

# 3) Run the orchestrator (first run will download model & dataset)
source .venv/bin/activate
python trustbench.py --backend ollama --subset 150 --metric ask
```

You will be prompted: **"Which metric to prioritize? [em/f1/rouge/bertscore]"**  
Pick one (we recommend `rouge`).

**Outputs**
- `results/outputs.jsonl` — model generations
- `results/metrics_detail.jsonl` — per-item metrics
- `results/metrics_summary.json` — aggregate metrics
- `results/primary_metric.txt` — the single number requested
- Console also prints **model/dataset availability checks**

---

## Common commands

**Run with a fixed metric:**
```bash
python trustbench.py --metric rouge
```

**Use MLX backend:**
```bash
pip install mlx-lm
python trustbench.py --backend mlx --metric rouge
```

**Change subset size:**
```bash
python trustbench.py --subset 200
```

**Rerun only evaluation (skip generation):**
```bash
python trustbench.py --skip-generate --metric rouge
```

---

## Git: initialize and push to GitHub (step-by-step)

```bash
# From the 'trustbench' folder
git init
git add .
git commit -m "TrustBench v0: reference-based evaluation on TruthfulQA"

# Create a new GitHub repo (via web UI) named, say, trustbench
# Then set your remote (replace <YOUR-USERNAME>)
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/trustbench.git
git push -u origin main
```

> If you prefer SSH:
> ```bash
> git remote add origin git@github.com:<YOUR-USERNAME>/trustbench.git
> git push -u origin main
> ```

---

## Notes

- **Ollama check**: The script tries to `ollama pull llama3.2:1b` and confirm via `ollama list`.
- **Server**: If the HTTP call fails, start it in another terminal: `ollama serve`.
- **BERTScore** is optional and slower; only install/use if you pick it.
- **Reproducibility**: We set a fixed random seed for the subset selection and a generation seed where supported.

Happy testing! 🚀

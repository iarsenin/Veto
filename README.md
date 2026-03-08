# Veto — HCRG Research Project

> **For AI agents reading this file:** This README is the single source of truth for project state. Read it fully before taking any action. After each meaningful development (successful run, new results, new code added), update the `## Current Status` section to reflect the new state. Do not remove history — append new status entries with a date.

---

## What This Project Is

This is a research implementation of **HCRG (Head-Level Context-Dependent Repression Gates)**, a novel transformer architecture modification proposed in the paper:

> *"Learning to Veto: Mitigating Hallucinations via Head-Level Repression Gates in Language Models"*
> Full paper: [Google Doc](https://docs.google.com/document/d/1ZkC5BWcbW20FS0QZ1SR06tiTrxpSXCBZ5R1ZAP9GeJM/edit?tab=t.0)

The codebase is built on top of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

---

## The Core Idea

Standard transformers have a purely additive residual stream — every attention head contributes positively, and any inhibitory behaviour must emerge indirectly through learned weight cancellation. This is an inefficient inductive bias for reliable generation.

HCRG adds a small **context-dependent scalar gate per attention head per layer**. The gate is computed from the current token's pre-attention residual state:

```
g = sigmoid(W_gate @ x + b_gate)    # shape: (B, T, n_head)
y_gated[:,h,:] = g[:,h] * head_output[:,h,:]
```

Key design properties:
- Gates are **context-dependent**: a head suppressed in one prompt may be active in another
- Gates are **conditioned on the pre-attention residual** (not sequence pooling), preserving KV-cache compatibility
- Bias is **initialised to −5** so `sigmoid(−5) ≈ 0.007` — gates start nearly fully open, training begins identically to baseline
- **Parameter overhead < 0.1%** of total model params
- Training uses **standard next-token cross-entropy**, no auxiliary losses

The hypothesis is that gradient descent, given this explicit "shortcut", will learn to suppress heads that consistently cause hallucinations rather than routing complex cancellation gradients through millions of parameters.

---

## Paper's Testable Predictions

| # | Prediction | How to test |
|---|---|---|
| P1 | HCRG outperforms baseline on hallucination-sensitive tasks | TruthfulQA, HaluEval-Wild eval scripts (not yet built) |
| P2 | Better long-context entity consistency | LongBench contradiction rate (not yet built) |
| P3 | Minimal capability regression vs baseline | Val loss comparison — **current experiment grid** |
| P4 | Gates are sparse and context-dependent, not uniform dampeners | Gate activation probing notebook (not yet built) |
| P5 | Higher resistance to adversarial prompt framing | PS/MV framework (not yet built) |

---

## Repository Structure

```
.
├── model.py                  # Baseline GPT (original nanoGPT)
├── custom_model.py           # HCRG-augmented GPT (HCRGBlock, HCRGCausalSelfAttention)
├── train.py                  # Training loop — use --use_custom_arch=True for HCRG
├── sample.py                 # Text generation from a checkpoint
├── run_experiments.sh        # Full 12-run experiment grid (2 grids × 2 archs × 3 seeds)
├── analyze_results.py        # Aggregates metrics.jsonl → results.json
├── download_tinystories.py   # Downloads and tokenises TinyStories (Grid 1 dataset)
├── configurator.py           # CLI flag override utility
├── bench.py                  # Standalone throughput benchmark
├── scaling_laws.ipynb        # Notebook for scaling analysis
├── transformer_sizing.ipynb  # Notebook for model sizing
├── config/                   # Named training configs (eval_gpt2*.py, finetune_*)
└── data/
    ├── shakespeare_char/     # 1MB char-level dataset (smoke test only)
    ├── openwebtext/          # OWT data prep script
    ├── shakespeare/          # BPE Shakespeare data prep
    └── tinystories/          # Created by download_tinystories.py (Grid 1)
```

---

## Experiment Grid

The experiment compares **baseline GPT vs HCRG GPT** across two scales and three seeds (12 total runs):

### Grid 1 — Micro (~10M params)
- Dataset: TinyStories (`data/tinystories/`)
- Architecture: 4 layers, 4 heads, 256 embedding
- Tokens processed: 100M (800 iterations)
- Suitable for: local MPS run (~2–4 hrs on M1)

### Grid 2 — Standard (~124M params)
- Dataset: FineWeb-Edu (`data/fineweb/`)
- Architecture: 12 layers, 12 heads, 768 embedding (GPT-2 scale)
- Tokens processed: 1B (2100 iterations)
- Requires: cloud GPU (A100 ~10–11 hrs, ~$13–15 on Lambda Labs)

Output per run: `out/<grid>/<arch>/seed<N>/metrics.jsonl`
Final summary: `results.json` (written by `analyze_results.py`)

The `metrics.jsonl` format:
```
{"type": "meta",  "arch": "hcrg"|"baseline", "seed": N, ...}
{"type": "train", "iter": N, "loss": F, "grad_norm": F}
{"type": "eval",  "iter": N, "val_loss": F, "hidden_var": F}
```

---

## Hardware

**Local machine:** Apple M1 Mac mini, 8 GB RAM
- Use `--device=mps --compile=False --dtype=float32` for MPS
- Use `--device=cpu --compile=False --dtype=float32` for CPU

**Cloud:** Lambda Labs or Vast.ai, A100 80GB recommended
- Use `--device=cuda --compile=True` (default in `run_experiments.sh`)

---

## Dependencies

```bash
pip install torch torchvision numpy tiktoken datasets tqdm requests
```

On Apple Silicon, standard `pip install torch` includes MPS support (PyTorch ≥ 1.12).

---

## Running the Smoke Test (verified working)

Uses the character-level Shakespeare dataset (no tiktoken needed, prepares in seconds):

```bash
# Prep data
python data/shakespeare_char/prepare.py

# Baseline (30 iters, ~1 min on CPU)
python train.py \
  --dataset=shakespeare_char --out_dir=out/smoke/baseline/seed42 \
  --n_layer=2 --n_head=2 --n_embd=64 \
  --batch_size=4 --gradient_accumulation_steps=1 \
  --block_size=64 --max_iters=30 \
  --eval_interval=10 --log_interval=5 \
  --device=cpu --compile=False --dtype=float32 \
  --always_save_checkpoint=False --seed=42

# HCRG (same settings)
python train.py \
  --use_custom_arch=True \
  --dataset=shakespeare_char --out_dir=out/smoke/hcrg/seed42 \
  --n_layer=2 --n_head=2 --n_embd=64 \
  --batch_size=4 --gradient_accumulation_steps=1 \
  --block_size=64 --max_iters=30 \
  --eval_interval=10 --log_interval=5 \
  --device=cpu --compile=False --dtype=float32 \
  --always_save_checkpoint=False --seed=42
```

---

## Running on Cloud (Lambda Labs / Vast.ai)

```bash
# 1. SSH into instance, clone repo, install deps
git clone https://github.com/iarsenin/Veto.git && cd Veto
pip install tiktoken datasets tqdm

# 2. Prep Grid 1 dataset
python download_tinystories.py

# 3. Prep Grid 2 dataset (~30–45 min)
python -c "
from datasets import load_dataset
import tiktoken, numpy as np, os, pickle
os.makedirs('data/fineweb', exist_ok=True)
enc = tiktoken.get_encoding('gpt2')
ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                  split='train', streaming=True, trust_remote_code=True)
tokens=[]; target=1_100_000_000
for ex in ds:
    tokens.extend(enc.encode_ordinary(ex['text'])); tokens.append(enc.eot_token)
    if len(tokens)>=target: break
arr=np.array(tokens[:target],dtype=np.uint16)
n_val=int(len(arr)*0.005)
arr[:-n_val].tofile('data/fineweb/train.bin')
arr[-n_val:].tofile('data/fineweb/val.bin')
pickle.dump({'vocab_size':enc.n_vocab},open('data/fineweb/meta.pkl','wb'))
print('Done.')
"

# 4. Run the full grid inside tmux (survives SSH disconnect)
tmux new -s exp
bash run_experiments.sh
# Detach: Ctrl+B then D  |  Reattach: tmux attach -t exp

# 5. Copy results back to local machine
# (run on your Mac)
scp -r ubuntu@<ip>:~/Veto/out ./out-cloud
scp ubuntu@<ip>:~/Veto/results.json ./results-cloud.json
```

**Always terminate the instance immediately after copying results.**

---

## What to Build Next (after results are in)

1. **Gate sparsity probing** (P4) — load a trained HCRG checkpoint, collect gate activations across diverse prompts, plot distribution per head/layer. Tells us whether gates learned selective inhibition or collapsed to uniform dampening.

2. **TruthfulQA eval** (P1) — wire up EleutherAI's `lm-evaluation-harness` against the trained checkpoint to test hallucination rate directly.

3. **L1 sparsity regularizer** — the paper (§10) warns that without a sparsity penalty on gate outputs, gates may collapse. Worth adding as an optional `--gate_l1_coeff` flag to `train.py` and re-running.

---

## Current Status

### 2026-03-07 — Smoke test passed, awaiting cloud run

**Architecture:** Fully implemented and verified.
- `custom_model.py` implements HCRG exactly as specified in the paper (Appendix B)
- Gate init at −5 confirmed working (step-0 val loss matches baseline)
- Parameter overhead confirmed < 0.1%
- Both baseline and HCRG train stably on CPU (smoke test: 30 iters, Shakespeare char)

**Experiment grid:** Not yet run.
- `run_experiments.sh` is ready and correct
- Local smoke test passed on Apple M1 Mac mini (Python 3.10, PyTorch CPU)
- Cloud run pending: Lambda Labs had no A100 capacity at time of last attempt; Vast.ai is the recommended alternative

**Next action for agent:** Help user launch a cloud GPU instance (Lambda Labs or Vast.ai), run `run_experiments.sh`, and retrieve `results.json`. Once results are in, build the gate sparsity probing notebook.

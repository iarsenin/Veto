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
- Bias is **initialised to +5** so `sigmoid(+5) ≈ 0.993` — gates start nearly fully open, training begins identically to baseline
- **Parameter overhead < 0.1%** of total model params
- Training uses **standard next-token cross-entropy**, no auxiliary losses

The hypothesis is that gradient descent, given this explicit "shortcut", will learn to suppress heads that consistently cause hallucinations rather than routing complex cancellation gradients through millions of parameters.

---

## Paper's Testable Predictions

| # | Prediction | How to test | Status |
|---|---|---|---|
| P1 | HCRG outperforms baseline on hallucination-sensitive tasks | TruthfulQA, HaluEval-Wild eval scripts | Not yet built |
| P2 | Better long-context entity consistency | LongBench contradiction rate | Not yet built |
| P3 | Minimal capability regression vs baseline | Val loss comparison — experiment grid | **Micro: confirmed** (see results below). Standard: in progress |
| P4 | Gates are sparse and context-dependent, not uniform dampeners | Gate activation probing notebook | Notebook exists (`gate_probing.ipynb`), awaiting fixed checkpoints |
| P5 | Higher resistance to adversarial prompt framing | PS/MV framework | Not yet built |

---

## Repository Structure

```
.
├── model.py                  # Baseline GPT (original nanoGPT)
├── custom_model.py           # HCRG-augmented GPT (HCRGBlock, HCRGCausalSelfAttention)
├── train.py                  # Training loop — use --use_custom_arch=True for HCRG
├── sample.py                 # Text generation from a checkpoint
├── run_experiments.sh        # Full 12-run experiment grid (2 grids × 2 archs × 3 seeds)
├── analyze_results.py        # Aggregates metrics.jsonl → results.json (runs on server)
├── compare_runs.py           # Comprehensive local analysis: per-seed detail, cross-run comparison
├── download_tinystories.py   # Downloads and tokenises TinyStories (Grid 1 dataset)
├── gate_probing.ipynb        # Gate activation analysis notebook (P4)
├── configurator.py           # CLI flag override utility
├── bench.py                  # Standalone throughput benchmark
├── scaling_laws.ipynb        # Notebook for scaling analysis
├── transformer_sizing.ipynb  # Notebook for model sizing
├── config/                   # Named training configs (eval_gpt2*.py, finetune_*)
├── out-fixed/                # Results from Run 2 (fixed init, +5 bias) — current experiment
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
- Requires: cloud GPU (A100 ~10–11 hrs, ~$13–15 on RunPod/Lambda)

Output per run: `out/<grid>/<arch>/seed<N>/metrics.jsonl`
Final summary: `results.json` (written by `analyze_results.py`)

The `metrics.jsonl` format:
```
{"type": "meta",  "arch": "hcrg"|"baseline", "seed": N, ...}
{"type": "train", "iter": N, "loss": F, "grad_norm": F}
{"type": "eval",  "iter": N, "val_loss": F, "hidden_var": F}
```

**Note:** When `--compile=True`, torch.compile causes a warmup phase where the first ~100 iterations run uncompiled, then the model recompiles and re-runs those iters. This creates duplicate entries in `metrics.jsonl`. For analysis, use the **last** occurrence of each iter or deduplicate by iter number.

---

## Hardware

**Local machine:** MacBook Pro (or Apple M1 Mac mini, 8 GB RAM)
- Use `--device=mps --compile=False --dtype=float32` for MPS
- Use `--device=cpu --compile=False --dtype=float32` for CPU
- Python 3.10 required locally (torch installed under 3.10, not 3.12)

**Cloud:** RunPod A100-SXM4-80GB (current), also compatible with Lambda Labs / Vast.ai
- Use `--device=cuda --compile=True` (default in `run_experiments.sh`)

---

## Dependencies

```bash
pip install torch torchvision numpy tiktoken datasets tqdm requests
```

On Apple Silicon, standard `pip install torch` includes MPS support (PyTorch >= 1.12).

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

## Running on Cloud (RunPod)

```bash
# 1. SSH into instance, clone repo, install deps
git clone https://github.com/iarsenin/Veto.git && cd Veto
pip install tiktoken datasets tqdm

# 2. Apply the gate init fix if not yet committed to main:
#    In custom_model.py, ensure _GATE_BIAS_INIT = 5.0 (not -5.0)

# 3. Prep Grid 1 dataset (~5 min)
python download_tinystories.py

# 4. Prep Grid 2 dataset (~10–15 min on fast connection)
python -c "
from datasets import load_dataset
import tiktoken, numpy as np, os, pickle
os.makedirs('data/fineweb', exist_ok=True)
enc = tiktoken.get_encoding('gpt2')
ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                  split='train', streaming=True)
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

# 5. Run the full grid (use nohup to survive SSH disconnect)
nohup bash run_experiments.sh > /tmp/experiments.log 2>&1 &

# 6. Monitor progress
grep -E '(GRID=|finished)' /tmp/experiments.log

# 7. When done, copy results to /workspace (persists across pod restarts)
cp -r out /workspace/
cp results.json /workspace/

# 8. Copy results to local machine (run on your Mac)
scp -P <port> -i ~/.ssh/id_ed25519 -r root@<ip>:/root/Veto/out ./out-fixed
scp -P <port> -i ~/.ssh/id_ed25519 root@<ip>:/root/Veto/results.json ./results-fixed.json
```

**Always stop the pod immediately after copying results — it bills by the hour.**

---

## Results

### Run 1 — Bugged Init (2026-03-08) — discarded

The first experiment used `_GATE_BIAS_INIT = -5.0`, which caused gates to start nearly fully **closed** (`sigmoid(-5) ≈ 0.007`). HCRG underperformed baseline at both scales (micro: +0.017 val loss gap, standard: +0.115). Gate probing revealed the model never learned to open the gates. Raw data has been deleted; numbers are preserved here for reference only.

### Run 2 — Fixed Init (2026-03-09) — `out-fixed/`

After fixing `_GATE_BIAS_INIT` to `+5.0`, gates initialize nearly fully open (`sigmoid(+5) ≈ 0.993`), as intended.

**Micro results (complete):**

| Grid/Arch | Val Loss (mean ± std) | Grad Norm | Notes |
|---|---|---|---|
| micro/baseline | 4.024 ± 0.006 | 1.87 | |
| micro/hcrg | 4.025 ± 0.003 | 1.83 | **Gap: +0.001** (noise-level), lower variance |

Key improvements vs Run 1:
- HCRG-baseline gap: **+0.017 → +0.001** (16x smaller, within noise)
- Grad norm ratio (hcrg/baseline): **0.66 → 0.98** (gradient flow restored)
- HCRG seed variance: **0.007 → 0.003** (more stable training)
- Step-0 val losses match across architectures (~10.84), confirming fix

**Standard results: IN PROGRESS** — running on RunPod A100. ETA ~8-10 hours from start of standard grid (~2026-03-09 22:30 UTC).

---

## What to Build Next

1. **Complete standard scale analysis** — when Run 2 standard results arrive, compare HCRG vs baseline at 124M params / 1B tokens. This is the key test: at this scale, gates have enough training signal to potentially learn useful selective suppression.

2. **Gate sparsity probing on fixed checkpoints** (P4) — re-run `gate_probing.ipynb` on the new `out-fixed/` HCRG checkpoints. With correct initialization, we should see gates diverging from ~1.0 to show selective per-head suppression patterns (or not — if gates stay near 1.0, the architecture may need the L1 sparsity regularizer to learn suppression).

3. **L1 sparsity regularizer** — the paper (§10) warns that without a sparsity penalty, gates may not learn to close. Add `--gate_l1_coeff` flag to `train.py` and re-run if gates show no sparsity in the fixed run.

4. **TruthfulQA eval** (P1) — wire up EleutherAI's `lm-evaluation-harness` against a trained standard-scale HCRG checkpoint.

5. **Longer training / larger scale** — if standard results are promising, consider training for more tokens or at larger model size to see if gate effects amplify with scale.

---

## Current Status

### 2026-03-07 — Initial implementation

Architecture implemented and smoke-tested on CPU (Shakespeare char, 30 iters). All code verified working.

### 2026-03-08 — Run 1 (bugged init)

Ran all 12 experiments on RunPod A100. Results in `out-cloud/`. Discovered via `gate_probing.ipynb` that `_GATE_BIAS_INIT` was `-5.0` (gates nearly closed at init), causing HCRG to underperform baseline. Fixed to `+5.0` in `custom_model.py`.

### 2026-03-09 — Run 2 (fixed init, standard grid in progress)

Re-ran experiments with corrected gate initialization. Micro grid complete — HCRG matches baseline (P3 confirmed at micro scale). Standard grid launched on RunPod A100, expected to finish ~06:00–08:00 UTC March 10.

**Standard grid is currently running on RunPod. START HERE:**

1. **Check if standard runs finished:**
   ```bash
   ssh root@64.247.196.119 -p 18535 -i ~/.ssh/id_ed25519 \
     "grep -E '(GRID=|finished|Done)' /tmp/experiments.log"
   ```
   - `Done. See results.json` at the end → all 6 standard runs complete
   - If SSH refused → pod may have been stopped; check RunPod dashboard

2. **Download results** (micro already in `out-fixed/micro/`):
   ```bash
   for arch in baseline hcrg; do
     for seed in 42 100 1337; do
       mkdir -p out-fixed/standard/${arch}/seed${seed}
       scp -P 18535 -i ~/.ssh/id_ed25519 \
         root@64.247.196.119:/root/Veto/out/standard/${arch}/seed${seed}/metrics.jsonl \
         out-fixed/standard/${arch}/seed${seed}/
     done
   done
   ```

3. **Download checkpoints** for gate probing:
   ```bash
   scp -P 18535 -i ~/.ssh/id_ed25519 -r \
     root@64.247.196.119:/root/Veto/out ./out-fixed-full
   ```

4. **Stop the RunPod pod immediately** — it costs ~$1.64/hr.

5. **Run analysis:**
   ```bash
   python3.10 compare_runs.py --run out-fixed
   ```

6. **Gate probing** on fixed HCRG checkpoints via `gate_probing.ipynb`.

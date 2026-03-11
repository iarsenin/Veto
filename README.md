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
| P3 | Minimal capability regression vs baseline | Val loss comparison — experiment grid | **Confirmed and reproduced.** HCRG matches baseline at micro; **outperforms at standard** (−0.016 to −0.018 val loss across two independent hardware runs) |
| P4 | Gates are sparse and context-dependent, not uniform dampeners | Gate activation probing (`full_analysis.py`, `gate_probing.ipynb`) | **Confirmed.** Standard-scale gates show substantial suppression (mean=0.66), context dependence (within-prompt std=0.14), and clear layer structure (L0≈1.0, L11≈0.27). See Gate Probing section below. |
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
├── compare_runs.py           # Quick local analysis: per-seed detail, cross-run comparison
├── full_analysis.py          # Complete analysis: gate probing (P4), convergence, cross-run, cross-seed
├── download_tinystories.py   # Downloads and tokenises TinyStories (Grid 1 dataset)
├── gate_probing.ipynb        # Gate activation analysis notebook (P4)
├── configurator.py           # CLI flag override utility
├── bench.py                  # Standalone throughput benchmark
├── scaling_laws.ipynb        # Notebook for scaling analysis
├── transformer_sizing.ipynb  # Notebook for model sizing
├── config/                   # Named training configs (eval_gpt2*.py, finetune_*)
├── out-fixed/                # Run 2 metrics (A100, no checkpoints — lost on pod termination)
├── out-run3/out/             # Run 3 full results (RTX 6000 Ada) — checkpoints + metrics
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

**Cloud:** RunPod — tested on A100-SXM4-80GB (Run 2) and RTX 6000 Ada 48GB (Run 3). Also compatible with Lambda Labs / Vast.ai
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

# 2. Prep Grid 1 dataset (~5 min)
python download_tinystories.py

# 3. Prep Grid 2 dataset (~10–15 min on fast connection)
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

# 4. Run the full grid (use nohup to survive SSH disconnect)
nohup bash run_experiments.sh > /tmp/experiments.log 2>&1 &

# 5. Monitor progress
grep -E '(GRID=|finished)' /tmp/experiments.log

# 6. When done, copy results to /workspace (persists across pod restarts)
cp -r out /workspace/
cp results.json /workspace/

# 7. Copy results to local machine (run on your Mac)
scp -P <port> -i ~/.ssh/id_ed25519 -r root@<ip>:/root/Veto/out ./out-fixed
scp -P <port> -i ~/.ssh/id_ed25519 root@<ip>:/root/Veto/results.json ./results-fixed.json
```

**Always stop the pod immediately after copying results — it bills by the hour.**

---

## Results

### Run 1 — Bugged Init (2026-03-08) — discarded

The first experiment used `_GATE_BIAS_INIT = -5.0`, which caused gates to start nearly fully **closed** (`sigmoid(-5) ≈ 0.007`). HCRG underperformed baseline at both scales (micro: +0.017 val loss gap, standard: +0.115). Gate probing revealed the model never learned to open the gates. Raw data has been deleted; numbers are preserved here for reference only.

### Run 2 — Fixed Init (2026-03-09/10, A100) — `out-fixed/`

All 12 runs complete. After fixing `_GATE_BIAS_INIT` to `+5.0`, gates initialize nearly fully open (`sigmoid(+5) ≈ 0.993`), as intended. Metrics only (checkpoints lost on pod termination).

### Run 3 — Reproducibility (2026-03-11, RTX 6000 Ada) — `out-run3/out/`

Full re-run of all 12 experiments on different hardware. All checkpoints and metrics saved.

**Summary (Run 3, primary results):**

| Grid/Arch | Val Loss (mean ± std) | Grad Norm | Notes |
|---|---|---|---|
| micro/baseline | 4.025 ± 0.006 | 1.88 | |
| micro/hcrg | 4.023 ± 0.004 | 1.90 | Gap: −0.002 (noise-level) |
| **standard/baseline** | **3.701 ± 0.016** | **1.23** | |
| **standard/hcrg** | **3.685 ± 0.004** | **1.22** | **Gap: −0.016 (HCRG wins)** |

**Cross-hardware reproducibility (Run 2 A100 vs Run 3 RTX 6000 Ada):**

| | Run 3 (RTX 6000 Ada) | Run 2 (A100) |
|---|---|---|
| standard HCRG-baseline gap | **−0.016** | **−0.018** |
| micro HCRG-baseline gap | −0.002 | +0.001 |
| Absolute val loss difference | < 0.005 per configuration | |

**Per-seed detail (standard scale, Run 3):**

| Seed | Baseline val loss | HCRG val loss | Delta |
|---|---|---|---|
| 42 | 3.687 | 3.680 | −0.007 |
| 100 | 3.697 | 3.686 | −0.011 |
| 1337 | 3.718 | 3.688 | −0.030 |
| **Mean** | **3.701** | **3.685** | **−0.016** |

**Convergence trajectory (standard, Run 3, mean across 3 seeds):**

| Iter | Baseline | HCRG | Delta | Note |
|---|---|---|---|---|
| 0 | 10.979 ± 0.009 | 10.974 ± 0.013 | −0.005 | Tied — gates start open |
| 500 | 5.658 ± 0.008 | 5.636 ± 0.004 | −0.022 | HCRG ahead early |
| 1000 | 4.460 ± 0.011 | 4.470 ± 0.006 | +0.010 | Brief HCRG dip (gates adjusting) |
| 1500 | 3.928 ± 0.012 | 3.920 ± 0.002 | −0.008 | HCRG recovers |
| 2000 | 3.701 ± 0.016 | 3.685 ± 0.004 | **−0.016** | **HCRG pulls ahead** |

**Analysis:**

- **P3 exceeded expectations and reproduced across hardware.** HCRG improves over baseline at standard scale by −0.016 to −0.018 val loss, consistently across all 3 seeds and two different GPUs (A100, RTX 6000 Ada).
- **The gain emerges late in training.** At iter 1000, HCRG is briefly behind (+0.010), likely as gates transition from their open initialization to learned suppression patterns. By iter 2000, the gates have settled and HCRG overtakes baseline.
- **Lower seed variance.** HCRG std is 0.004 vs baseline's 0.016, suggesting the gating mechanism stabilizes training.
- **Gradient norms are similar** (ratio 0.989), unlike the bugged run where closed gates starved gradient flow (ratio 0.66).
- **At micro scale, gates don't have enough training signal** (only 800 iters) to learn useful patterns — they stay near 1.0 and results are identical to baseline. The standard scale (2100 iters, 1B tokens) gives the gates time to differentiate.

---

## Gate Probing Results (P4)

Full analysis performed via `full_analysis.py` on Run 3 checkpoints (`out-run3/out/`). Interactive visualization available in `gate_probing.ipynb`.

### Standard Scale (12L/12H/768E, 1B tokens)

**Overall gate statistics:**

| Metric | Value |
|---|---|
| Mean gate value | 0.656 |
| Std | 0.305 |
| < 0.5 (actively suppressing) | 30.8% |
| < 0.9 | 69.7% |
| > 0.99 (fully open) | 6.1% |
| Heads with mean gate < 0.9 | 120 / 144 (83%) |

**Layer-wise structure — gates exhibit a clear depth gradient:**

| Layer | Mean gate | <0.5 % | <0.9 % | Interpretation |
|---|---|---|---|---|
| L0 | 0.988 | 0.0% | 1.0% | Near-fully open — early features pass through |
| L1–L4 | 0.79–0.87 | 2–5% | 48–75% | Modest suppression begins |
| L5–L8 | 0.55–0.75 | 17–46% | 62–85% | Active gating — heads selectively suppressed |
| L9–L11 | 0.27–0.51 | 51–84% | 87–100% | Heavy suppression — most heads partially silenced |

**Context dependence:**

| Metric | Value | Interpretation |
|---|---|---|
| Within-prompt std | 0.144 | High — same head gets different gate values at different positions |
| Cross-prompt std | 0.077 | Moderate — gate patterns differ across input types |
| Most variable head | L9/H3 (within-std=0.303) | This head's gate swings from ~0.3 to ~0.9 depending on token context |

**Most suppressed heads (near-zero gates):**

L11/H9 (mean=0.077), L10/H9 (0.119), L11/H10 (0.164), L10/H0 (0.173), L11/H1 (0.178). These heads are almost entirely silenced — the model learned they are not useful for next-token prediction.

**Cross-seed consistency:**

| Seed pair | Correlation (r) of per-head mean gates |
|---|---|
| 42 vs 100 | 0.767 |
| 42 vs 1337 | 0.770 |
| 100 vs 1337 | 0.741 |

The layer-wise suppression gradient is consistent across all 3 seeds (L0≈0.98 open → L11≈0.26 closed), confirming this is a learned structural property, not a random seed artifact.

**Gate bias vs weight analysis:**

Biases barely moved from init (mean=4.90, init=5.0, delta=−0.10). All gating behavior comes from the **learned weights** (W_gate @ x), not the biases. The weight norms increase from L0 (14.1, high because L0 barely uses gating) through L1–L11 (3.4–5.9, growing with depth). This confirms the gates are genuinely computing context-dependent functions.

### Micro Scale (4L/4H/256E, 100M tokens)

Gates remained near-fully open: mean=0.993, 0% below 0.9. With only 800 training iterations, the gates never received enough gradient signal to learn suppression patterns. This explains why micro-scale HCRG matches baseline exactly.

### P4 Verdict

**P4 is confirmed.** At standard scale, HCRG gates are:
1. **Sparse** — 83% of heads have mean gate < 0.9; deep layers are 50–84% below 0.5
2. **Context-dependent** — within-prompt std of 0.14 means gates actively respond to token content
3. **Structured** — clear depth gradient from open (L0) to heavily gated (L11)
4. **Reproducible** — consistent patterns across 3 random seeds (r ≈ 0.76)

The gates are NOT uniform dampeners. They learned selective, position-dependent head suppression.

---

## What to Build Next

1. **TruthfulQA eval** (P1) — wire up EleutherAI's `lm-evaluation-harness` against the standard-scale HCRG checkpoints in `out-run3/out/standard/hcrg/`. The val loss improvement and gate probing results strongly suggest HCRG should reduce hallucinations; TruthfulQA would provide direct evidence.

2. **Longer training** — the HCRG advantage is still growing at iter 2000 (gap widens from −0.008 at iter 1500 to −0.016 at iter 2000). Training for 5000–10000 iterations could reveal whether the gain continues to compound or plateaus.

3. **L1 sparsity regularizer** — the paper (§10) suggests a sparsity penalty on gate outputs. The gates already learned substantial sparsity without it (mean gate 0.66, 83% of heads < 0.9). Adding `--gate_l1_coeff` to `train.py` could push deeper suppression and potentially amplify the advantage.

4. **Larger scale** — the gain appeared at 124M / 1B tokens but not at 10M / 100M tokens. Testing at 350M+ params or 10B+ tokens would show whether the effect amplifies with scale, as the paper predicts.

5. **Head ablation study** — the gate probing identified specific heads that are almost fully suppressed (L11/H9 at 0.077, L10/H9 at 0.119). Pruning these heads from the baseline and comparing performance would validate whether the gates are identifying genuinely redundant computation.

---

## Current Status

### 2026-03-07 — Initial implementation

Architecture implemented and smoke-tested on CPU (Shakespeare char, 30 iters). All code verified working.

### 2026-03-08 — Run 1 (bugged init)

Ran all 12 experiments on RunPod A100. Discovered `_GATE_BIAS_INIT` was `-5.0` (gates nearly closed). Fixed to `+5.0` in `custom_model.py`.

### 2026-03-10 — Run 2 complete: HCRG outperforms baseline at standard scale

All 12 runs finished on RunPod A100. Metrics downloaded to `out-fixed/`. Checkpoints lost on pod termination.

### 2026-03-11 — Run 3 complete + Gate probing confirms P4

**Run 3 (reproducibility):** Re-ran all 12 experiments on RTX 6000 Ada. All checkpoints + metrics saved locally in `out-run3/out/`. Results reproduce within ±0.005 of Run 2 (HCRG advantage: −0.016 vs −0.018 on standard scale).

**Gate probing (P4):** `full_analysis.py` confirms gates learned substantial, context-dependent suppression at standard scale:
- Mean gate value = 0.66 (far from the 0.993 initialization)
- 120/144 heads have mean gate < 0.9
- Clear depth gradient: L0 open (0.99) → L11 heavily gated (0.27)
- Within-prompt std = 0.14 (context-dependent, not static)
- Consistent across 3 seeds (r ≈ 0.77)
- Micro-scale gates stayed at 0.993 (insufficient training signal)

**Status: P3 confirmed and reproduced. P4 confirmed. P1/P2/P5 remain untested.**

**Next action for agent:**

1. Review the Results and Gate Probing sections above.
2. Checkpoints available locally in `out-run3/out/` for further analysis.
3. Priority: TruthfulQA eval (P1) using standard-scale HCRG checkpoints.
4. See "What to Build Next" for full roadmap.

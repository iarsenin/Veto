# Veto — HCRG Research Project

> **For AI agents:** This README is the single source of truth for project state. Read it fully before taking any action. The complete empirical results are in `results.md`. After any meaningful development, append a new entry to `## Current Status` with a date.

---

## What This Project Is

A research implementation of **HCRG (Head-Level Context-Dependent Repression Gates)**, a transformer architecture modification proposed in:

> *"Learning to Veto: Mitigating Hallucinations via Head-Level Repression Gates in Language Models"*
> Full paper: [Google Doc](https://docs.google.com/document/d/1ZkC5BWcbW20FS0QZ1SR06tiTrxpSXCBZ5R1ZAP9GeJM/edit?tab=t.0)

Built on top of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

---

## The Core Idea

Standard transformers have a purely additive residual stream — every attention head contributes positively, and inhibitory behaviour must emerge indirectly through weight cancellation. HCRG adds an explicit **per-head, context-dependent scalar gate**:

```
g = sigmoid(W_gate @ LayerNorm(x) + b_gate)   # shape: (B, T, n_head)
head_output[:,h,:] *= g[:,h,:]
```

Key properties:
- Gates are **per-token** — the same head can be open on one token and suppressed on another in the same sequence
- Conditioning is on the **pre-attention residual stream** — preserves KV-cache compatibility
- Initialized with bias = +5, so σ(+5) ≈ 0.993 — gates start fully open; training is identical to baseline at step 0
- **Parameter overhead 0.08%** — 221K additional params on a 124M model
- **No auxiliary losses** — standard next-token cross-entropy only

The hypothesis: given an explicit suppression mechanism, gradient descent will learn to gate heads that cause harmful attention patterns, rather than routing those corrections through the main weights.

---

## Testable Predictions

| # | Prediction | Test | Status |
|---|---|---|---|
| P1 | Outperforms baseline on hallucination tasks | TruthfulQA, HaluEval | Not yet tested — needs larger scale |
| P2 | Better long-context entity consistency | LongBench contradiction rate | Not yet tested |
| P3 | Matches or exceeds baseline on capability | Val loss comparison | **Confirmed.** −0.016 (1B tokens, 3 seeds), −0.010 (2.46B tokens, cosine LR). Never reverses after 0.25B tokens. |
| P4 | Gates are sparse and context-dependent | Gate probing (1000 seqs × 3 seeds) | **Confirmed.** Mean gate 0.72, within-seq std 0.15, depth gradient L0→L11, consistent across seeds. |
| P5 | Resistance to adversarial framing | PS/MV framework | Not yet tested |

---

## Repository Structure

```
.
├── model.py                  # Baseline GPT-2 (original nanoGPT)
├── custom_model.py           # HCRG architecture (HCRGBlock, HCRGCausalSelfAttention)
├── train.py                  # Training loop — use --use_custom_arch=True for HCRG
├── run_experiments.sh        # Full 12-run experiment grid (2 scales × 2 archs × 3 seeds)
├── run_final.sh              # Final convergence run script (cosine LR)
├── analyze_results.py        # Aggregates metrics.jsonl → results.json
├── compare_runs.py           # Per-seed detail and cross-run comparison
├── full_analysis.py          # Gate probing, convergence, cross-seed analysis
├── probe_gates.py            # Large-scale gate probing (1000 seqs, forward hooks)
├── ablation_prune.py         # Static pruning ablation
├── gate_probing.ipynb        # Interactive gate analysis notebook
├── download_tinystories.py   # Downloads and tokenises TinyStories dataset
├── configurator.py           # CLI flag override utility
├── results.md                # ← Complete empirical results and analysis
├── probing_stats_seed*.json  # Gate probing data (1000 seqs × 3 seeds)
├── ablation_results.json     # Static pruning ablation results
└── data/
    ├── shakespeare_char/     # 1MB char-level dataset (smoke test)
    └── tinystories/          # Created by download_tinystories.py
```

---

## Experiment Grid

Two scales, two architectures, three seeds (12 runs total):

| | Micro | Standard |
|---|---|---|
| Params | ~10M | ~124M |
| Architecture | 4L/4H/256E | 12L/12H/768E (GPT-2) |
| Dataset | TinyStories | FineWeb-Edu |
| Tokens | 100M (800 iters) | 1B (2100 iters) |
| Hardware | Local MPS/CPU | Cloud GPU |

Metrics are written to `out/<grid>/<arch>/seed<N>/metrics.jsonl`. Format:
```
{"type": "train", "iter": N, "loss": F, "grad_norm": F}
{"type": "eval",  "iter": N, "val_loss": F, "hidden_var": F}
```

When `--compile=True`, `torch.compile` warmup creates duplicate iter numbers in `metrics.jsonl`. Always use the **last occurrence** of each iter for analysis.

---

## Hardware and Dependencies

**Local (Mac):** Python 3.10 required (torch is installed under 3.10).
```bash
pip install torch torchvision numpy tiktoken datasets tqdm
```
Run flags: `--device=mps --compile=False --dtype=float32`

**Cloud (RunPod/Lambda):** Tested on A100-SXM4-80GB and RTX 6000 Ada 48GB.
Run flags: `--device=cuda --compile=True --dtype=bfloat16`

---

## Running a Smoke Test

```bash
python data/shakespeare_char/prepare.py

# Baseline
python train.py \
  --dataset=shakespeare_char --out_dir=out/smoke/baseline/seed42 \
  --n_layer=2 --n_head=2 --n_embd=64 \
  --batch_size=4 --gradient_accumulation_steps=1 --block_size=64 \
  --max_iters=30 --eval_interval=10 --log_interval=5 \
  --device=cpu --compile=False --dtype=float32 \
  --always_save_checkpoint=False --seed=42

# HCRG (add one flag)
python train.py --use_custom_arch=True  [same flags as above]
```

---

## Running the Full Grid on Cloud

```bash
# 1. Clone and install deps
git clone https://github.com/iarsenin/Veto.git && cd Veto
pip install tiktoken datasets tqdm

# 2. Prepare datasets (run one-time)
python download_tinystories.py    # Grid 1 (~5 min)
python -c "                       # Grid 2 (~15 min)
from datasets import load_dataset
import tiktoken, numpy as np, os, pickle
os.makedirs('data/fineweb', exist_ok=True)
enc = tiktoken.get_encoding('gpt2')
ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
tokens = []
for ex in ds:
    tokens.extend(enc.encode_ordinary(ex['text'])); tokens.append(enc.eot_token)
    if len(tokens) >= 1_100_000_000: break
arr = np.array(tokens[:1_100_000_000], dtype=np.uint16)
n_val = int(len(arr) * 0.005)
arr[:-n_val].tofile('data/fineweb/train.bin')
arr[-n_val:].tofile('data/fineweb/val.bin')
pickle.dump({'vocab_size': enc.n_vocab}, open('data/fineweb/meta.pkl', 'wb'))
"

# 3. Run experiment grid
nohup bash run_experiments.sh > /tmp/experiments.log 2>&1 &

# 4. Back up before stopping pod (container disk is wiped on termination)
cp -r out /workspace/ && cp results.json /workspace/

# 5. Download to local (run on your Mac)
scp -P <port> -i ~/.ssh/id_ed25519 -r root@<ip>:/root/Veto/out ./out-new
```

---

## Results Summary

Full results, analysis, and data reference: **`results.md`**

Quick summary:

| Experiment | HCRG delta (standard scale) |
|---|---|
| Run 2 (fixed init, A100, 3 seeds) | **−0.018** |
| Run 3 (reproducibility, RTX 6000, 3 seeds) | **−0.016** |
| Phase 2 (5000 iters, flat LR) | −0.005 (oscillating — flat LR artifact) |
| **Final convergence (5000 iters, cosine LR)** | **−0.010 (stable)** |

HCRG consistently outperforms baseline at standard scale (124M params). No advantage at micro scale (10M). Gates confirmed sparse and context-dependent. Dynamic gating strictly outperforms static head pruning.

---

## Current Status

### 2026-03-07 — Implementation

Architecture implemented and smoke-tested on CPU (Shakespeare char, 30 iters). All code verified working.

### 2026-03-08 — Run 1: bug found

First experiment used `_GATE_BIAS_INIT = -5.0` (gates nearly closed at init). HCRG underperformed. Fixed to `+5.0` in `custom_model.py`.

### 2026-03-10 — Run 2: HCRG outperforms at standard scale

All 12 runs on A100. HCRG advantage −0.018 standard, −0.001 micro. Checkpoints lost on pod termination.

### 2026-03-11 — Run 3: reproducibility confirmed, P4 confirmed

Re-ran all 12 runs on RTX 6000 Ada. HCRG advantage −0.016 (reproduces Run 2 within ±0.002). All checkpoints saved in `out-run3/out/`. Gate probing (P4): mean gate 0.66, 120/144 heads < 0.9, depth gradient, within-seq std 0.14.

### 2026-03-12 — Phase 2: large-scale probing and ablation

Large-scale gate probing (1000 seqs × 3 seeds): context dependence confirmed at scale, depth gradient and head rankings stable across seeds. Static pruning ablation: dynamic gating strictly superior — best static prune achieves 0 improvement; HCRG achieves −0.010. Extended training to 5000 iters (flat LR) showed oscillating gap; identified flat LR as confound.

### 2026-03-13 — Final convergence: definitive benchmark complete

Re-ran from scratch with cosine LR (`lr_decay_iters=5000`, `warmup_iters=500`, `min_lr=6e-5`). HCRG overtakes baseline at 0.25B tokens, never reverses, final gap **−0.010** (seeds 42 and 1337 agree within 0.001).

**Status: Comprehensive empirical case for HCRG is complete. P3 and P4 confirmed across 5 independent experiments and 2 hardware configurations. P1/P2/P5 require larger scale or task-specific benchmarks. See `results.md` for full analysis.**

### Next actions

1. Run TruthfulQA eval using `lm-evaluation-harness` on `out-run3/out/standard/hcrg/` checkpoints (note: 124M is a weak baseline for this benchmark — consider 350M+ for meaningful scores).
2. Add a third seed (seed 100) to the final convergence run.
3. Run `full_analysis.py` against `out-final/` checkpoints to get updated gate probing on the 5000-iter HCRG models.

# HCRG Phase 2: Execution Plan and Analysis

## Assessment of the Proposal

The paper writer's three tasks are well-targeted. Each directly addresses a specific weakness in the current evidence:

| Task | Addresses | Risk |
|---|---|---|
| 1. Asymptotic training | "Gap still widening at iter 2100" | Low — straightforward extension |
| 2. Large-scale probing | "Only 4 hand-picked prompts" | Zero — inference on existing checkpoints |
| 3. Static pruning ablation | "Maybe HCRG is just auto-pruning?" | Medium — result could go either way |

**Overall verdict: Approved with refinements below.**

---

## Task-by-Task Analysis

### Task 1: Asymptotic Training (5000 iterations)

**The proposal is sound but we can cut compute cost by ~40% by resuming from existing checkpoints** rather than retraining from scratch.

Key insight: the learning rate schedule in `run_experiments.sh` uses the `train.py` defaults (`warmup_iters=2000`, `lr_decay_iters=600000`). Since `lr_decay_iters` was never changed, the LR at iter 2100 is:

```
decay_ratio = (2100 - 2000) / (600000 - 2000) = 0.00017
LR ≈ 6e-4 (essentially peak)
```

At iter 5000 it's still `LR ≈ 6e-4`. The LR is flat throughout 0–5000 iters. This means:

1. **Resuming from checkpoint is valid** — no LR discontinuity
2. We only need to train from iter ~2000 to 5000 (3000 new iters per run, not 5000)
3. Existing eval points (0, 500, 1000, 1500, 2000) are already in `metrics.jsonl` from Run 3

**Refinements:**
- **Resume from Run 3 checkpoints** (`out-run3/out/standard/{arch}/seed{N}/ckpt.pt`). Use `--init_from=resume` and `--max_iters=5000`.
- **eval_interval=250** (not 500) — finer granularity to better identify the plateau point. This gives us eval points at 2250, 2500, 2750, 3000, ..., 5000.
- **always_save_checkpoint=True** — ensures we get the final checkpoint even if val loss starts rising.
- **2 seeds minimum** (42, 1337) as proposed. 3 seeds (add 100) is better for error bars but costs 50% more. Recommend starting with 2, adding 100 if budget allows.

**Caveat to note in the paper:** the flat LR schedule (no cosine annealing over the training horizon) means the model isn't getting the benefit of LR warmdown. Results at 5000 iters may be conservative compared to a properly scheduled run. This is consistent with Phase 1 results.

### Task 2: Large-Scale Gate Probing (1000 sequences)

**This can be done locally on CPU — no GPU pod needed.** The 124M model does inference quickly on CPU. 1000 sequences × 1024 tokens ≈ 1M tokens of inference.

**One dependency:** we need the FineWeb-Edu `val.bin` file, which is on the terminated pod. We have two options:
- Prepare it locally (download ~5.5M tokens from HuggingFace, tokenize) — ~10 min
- Copy it from the Task 1 pod during the training run

**Refinements:**
- **Probe all 3 seeds** (42, 100, 1337), not just seed 42. Cross-seed statistics with 1000 sequences each will be far more robust than our current 4-prompt analysis. Marginal compute cost (3× inference on CPU, still fast).
- **Output format:** the proposal's `probing_stats.json` schema is correct. Add percentile distributions (5th, 25th, 50th, 75th, 95th) for gate values per layer.

### Task 3: Static Pruning Ablation

**Good experiment, but needs multiple pruning thresholds, not just top 10%.**

**Refinements:**
- **Test thresholds at 5%, 10%, 15%, 20%, 30% of heads pruned.** This gives a curve, not a single data point. If pruning 10% helps but 20% hurts, we learn something about the precision of HCRG's head selection.
- **Three-way comparison at each threshold:**
  1. Unpruned baseline (constant reference: val_loss ≈ 3.70)
  2. Baseline with static pruning (heads zeroed based on HCRG gate ranking)
  3. HCRG (val_loss ≈ 3.68)
- **Also test "inverse pruning"** — zero out the heads that HCRG keeps MOST open. This is a sanity check: if suppressing the "important" heads hurts more than suppressing the "unimportant" ones, it confirms HCRG's ranking is meaningful.
- **Can run on CPU locally** using existing checkpoints from `out-run3/out/`. No GPU needed.

---

## Execution Plan

### Phase 2a: Local work (no GPU, do first)

Prepare the FineWeb-Edu val.bin locally and run Tasks 2 and 3 while waiting for the pod.

**Step 1:** Prepare FineWeb-Edu val.bin locally
- Download a small slice from HuggingFace, tokenize with tiktoken
- Only need ~5.5M tokens (the validation split)
- Time: ~10 min, Memory: < 1GB

**Step 2:** Run `probe_gates.py` (Task 2) on CPU
- Load HCRG checkpoints for all 3 seeds from `out-run3/out/standard/hcrg/`
- Run inference on 1000 random sequences from val.bin
- Export `probing_stats.json`
- Time: ~10-20 min per seed on M1 MacBook

**Step 3:** Run `ablation_prune.py` (Task 3) on CPU
- Load baseline checkpoint from `out-run3/out/standard/baseline/seed42/`
- Use HCRG gate rankings from Step 2 to select heads to prune
- Evaluate at each pruning threshold (5%, 10%, 15%, 20%, 30%)
- Also evaluate inverse pruning
- Time: ~5-10 min per threshold on CPU

### Phase 2b: GPU work (pod needed for Task 1 only)

**Step 4:** Provision pod, upload checkpoints, prepare data

Upload the 4 checkpoints needed for resumption:
```
out-run3/out/standard/baseline/seed42/ckpt.pt     (1.4 GB)
out-run3/out/standard/baseline/seed1337/ckpt.pt   (1.4 GB)
out-run3/out/standard/hcrg/seed42/ckpt.pt         (1.4 GB)
out-run3/out/standard/hcrg/seed1337/ckpt.pt       (1.4 GB)
```
Total upload: ~5.6 GB

Also need FineWeb-Edu dataset on pod (train.bin + val.bin, ~2.1 GB). Re-download on pod since uploading 2.1 GB of data is slower than re-preparing.

**Step 5:** Run 4 extended training runs (resumed from iter ~2000 to 5000)
- Baseline seed42, Baseline seed1337, HCRG seed42, HCRG seed1337
- Each run: ~3000 new iterations
- Eval every 250 iters → 12 new eval points per run

**Step 6:** Download all results
- New metrics.jsonl files (append to existing)
- New checkpoints at iter 5000
- **Download BEFORE stopping pod** (lesson learned from Phase 1)

**Step 7:** Analysis
- Combine existing eval points (0-2000) with new points (2250-5000)
- Plot HCRG-baseline delta over time, identify plateau
- Export convergence CSV/JSON

---

## GPU Pod Requirements and Cost Analysis

Task 1 is the only task requiring a GPU pod. Here is what to expect:

### Compute requirements

- 4 training runs × ~3000 iters each (resumed) = 12,000 total iterations
- Each iteration: batch_size=12, grad_accum=40, block_size=1024 (491,520 tokens)
- Model: 124M params, bfloat16
- VRAM needed: ~14-16 GB (model + gradients + optimizer + activations for batch=12)

### If retraining from scratch instead of resuming

- 4 runs × 5000 iters = 20,000 total iterations
- ~67% more compute, ~67% more cost
- Advantage: cleaner (no dependence on Phase 1 checkpoints)
- Recommended only if budget is not a constraint

### GPU comparison table

Timing is based on Phase 1 measurements (RTX 6000 Ada: 5.26 s/iter measured, A100: 2.90 s/iter measured) and estimates for others.

#### Resuming from checkpoint (12,000 iters)

| GPU | VRAM | Fits? | Est. sec/iter | Total time | Est. $/hr | Est. total cost |
|---|---|---|---|---|---|---|
| **RTX 4090** | 24 GB | Yes | ~4.0 s | ~13.3 hrs | $0.35-0.44 | **$5-6** |
| **RTX A6000** | 48 GB | Yes | ~5.0 s | ~16.7 hrs | $0.50-0.76 | **$8-13** |
| **RTX 6000 Ada** | 48 GB | Yes | 5.26 s (measured) | ~17.5 hrs | $0.75 | **$13** |
| **A100 40GB** | 40 GB | Yes | ~3.1 s | ~10.3 hrs | $1.10-1.20 | **$11-12** |
| **A100 80GB** | 80 GB | Yes | 2.90 s (measured) | ~9.7 hrs | $1.50-1.75 | **$15-17** |
| **H100 80GB** | 80 GB | Yes | ~2.0 s | ~6.7 hrs | $2.50-3.50 | **$17-23** |
| RTX 3090 | 24 GB | Tight | ~6.0 s | ~20.0 hrs | $0.25-0.35 | **$5-7** |

#### From scratch (20,000 iters)

Multiply times and costs above by ~1.67.

### Recommendations

**Best value:** RTX 4090 (~$5-6, 13 hrs) or RTX 3090 (~$5-7, 20 hrs). Both have enough VRAM (24 GB) for our 124M model with batch=12.

**Best speed:** A100 40GB (~$11-12, 10 hrs). Good balance of speed and cost.

**Avoid:** H100 — overkill for a 124M model, most expensive option.

**VRAM note:** All GPUs with ≥ 24 GB VRAM should fit. The 124M model with batch_size=12, block_size=1024, bfloat16 uses ~14-16 GB peak including torch.compile overhead. GPUs with only 16 GB (e.g., RTX 4080, T4) may not fit and should be avoided.

### Adding seed 100 (optional, +50% cost)

| Scenario | Runs | Resume iters | From-scratch iters |
|---|---|---|---|
| 2 seeds (42, 1337) | 4 | 12,000 | 20,000 |
| 3 seeds (42, 100, 1337) | 6 | 18,000 | 30,000 |

---

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| HCRG advantage plateaus or reverses after iter 2100 | Weakens the paper narrative (but is still a valid finding) | Report honestly — a plateau is expected and informative |
| Static pruning matches HCRG performance | Reduces "dynamic gating" claim to "automatic head importance discovery" | Still valuable; emphasize context-dependence from Task 2 |
| Checkpoint resume causes training instability | Corrupted results | Monitor first 100 iters of resumed run; if loss spikes, fall back to from-scratch |
| Pod terminates before download | Data loss (again) | Auto-backup to /workspace after each run; download incrementally |

---

## Deliverables

1. `probe_gates.py` — large-scale probing script (Task 2)
2. `ablation_prune.py` — static pruning ablation script (Task 3)
3. `run_phase2.sh` — extended training wrapper with auto-backup (Task 1)
4. `probing_stats.json` — gate statistics from 1000 sequences × 3 seeds
5. `ablation_results.json` — pruning ablation at multiple thresholds
6. `phase2_results.md` — analysis write-up with convergence plots data
7. Updated `README.md` with Phase 2 findings

#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
#
# Executes the full experimental grid:
#   Grid 1 – Micro  (≈10M params)  on TinyStories  – 100M tokens processed
#   Grid 2 – Standard (124M params) on FineWeb-Edu  – 1B  tokens processed
#
# Each grid × each arch × each seed → one training run.
# Total: 2 grids × 2 archs × 3 seeds = 12 runs.
#
# Prerequisites
# -------------
#   pip install torch tiktoken datasets numpy
#   python download_tinystories.py          # populates data/tinystories/
#   python data/fineweb/prepare.py          # populates data/fineweb/ (see below)
#
# FineWeb-Edu data prep (nanoGPT style):
#   See https://github.com/karpathy/nanoGPT – run the FineWeb prepare script
#   or use:
#     python -c "
#     from datasets import load_dataset
#     import tiktoken, numpy as np, os, pickle
#     os.makedirs('data/fineweb', exist_ok=True)
#     enc = tiktoken.get_encoding('gpt2')
#     ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
#                       split='train', streaming=True, trust_remote_code=True)
#     tokens=[]; target=1_100_000_000
#     for ex in ds:
#         tokens.extend(enc.encode_ordinary(ex['text'])); tokens.append(enc.eot_token)
#         if len(tokens)>=target: break
#     arr=np.array(tokens[:target],dtype=np.uint16)
#     n_val=int(len(arr)*0.005)
#     arr[:-n_val].tofile('data/fineweb/train.bin')
#     arr[-n_val:].tofile('data/fineweb/val.bin')
#     pickle.dump({'vocab_size':enc.n_vocab},open('data/fineweb/meta.pkl','wb'))
#     "
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
SEEDS=(42 100 1337)

# ── helpers ──────────────────────────────────────────────────────────────────

run_one() {
    local LABEL="$1"
    local ARCH="$2"        # baseline | hcrg
    local SEED="$3"
    local DATASET="$4"
    local N_LAYER="$5"
    local N_HEAD="$6"
    local N_EMBD="$7"
    local BATCH_SIZE="$8"
    local GRAD_ACCUM="$9"
    local MAX_ITERS="${10}"
    local BLOCK_SIZE="${11}"

    local USE_CUSTOM="False"
    if [[ "$ARCH" == "hcrg" ]]; then
        USE_CUSTOM="True"
    fi

    local OUT="out/${LABEL}/${ARCH}/seed${SEED}"
    mkdir -p "$OUT"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  GRID=${LABEL}  ARCH=${ARCH}  SEED=${SEED}"
    echo "  out_dir=${OUT}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    $PYTHON train.py \
        --use_custom_arch="$USE_CUSTOM" \
        --seed="$SEED" \
        --dataset="$DATASET" \
        --out_dir="$OUT" \
        --n_layer="$N_LAYER" \
        --n_head="$N_HEAD" \
        --n_embd="$N_EMBD" \
        --batch_size="$BATCH_SIZE" \
        --gradient_accumulation_steps="$GRAD_ACCUM" \
        --max_iters="$MAX_ITERS" \
        --block_size="$BLOCK_SIZE" \
        --eval_interval=500 \
        --log_interval=10 \
        --always_save_checkpoint=False \
        --compile=True \
        --device=cuda \
        2>&1 | tee "${OUT}/train.log"

    echo "  ✓ finished  →  ${OUT}/metrics.jsonl"
}

# =============================================================================
# Grid 1 – Micro scale (~10M params)
# Dataset : TinyStories (100M tokens total)
#
# tokens_per_iter = grad_accum × world_size × batch × block_size
#                 = 8 × 1 × 16 × 1024 = 131 072
# max_iters for 100M tokens = ceil(100_000_000 / 131_072) = 763
# We round up to 800 to ensure ≥ 100M tokens processed.
# =============================================================================
MICRO_DATASET="tinystories"
MICRO_N_LAYER=4
MICRO_N_HEAD=4
MICRO_N_EMBD=256
MICRO_BATCH=16
MICRO_GRAD_ACCUM=8
MICRO_BLOCK=1024
# tokens_per_iter = 8 * 1 * 16 * 1024 = 131072
# 100_000_000 / 131072 ≈ 763 → use 800
MICRO_MAX_ITERS=800

echo "============================================================"
echo " GRID 1  –  Micro scale (~10M)  –  TinyStories  –  100M tk"
echo "============================================================"

# Download TinyStories if not yet done
if [[ ! -f "data/tinystories/train.bin" ]]; then
    echo "Downloading and tokenising TinyStories …"
    $PYTHON download_tinystories.py
fi

for SEED in "${SEEDS[@]}"; do
    for ARCH in baseline hcrg; do
        run_one "micro" "$ARCH" "$SEED" \
            "$MICRO_DATASET" "$MICRO_N_LAYER" "$MICRO_N_HEAD" "$MICRO_N_EMBD" \
            "$MICRO_BATCH" "$MICRO_GRAD_ACCUM" "$MICRO_MAX_ITERS" "$MICRO_BLOCK"
    done
done

# =============================================================================
# Grid 2 – Standard scale (124M params)
# Dataset : FineWeb-Edu (10B sample)
#
# tokens_per_iter = 40 × 1 × 12 × 1024 = 491 520
# max_iters for 1B tokens = ceil(1_000_000_000 / 491_520) = 2035
# We round up to 2100.
# =============================================================================
STD_DATASET="fineweb"
STD_N_LAYER=12
STD_N_HEAD=12
STD_N_EMBD=768
STD_BATCH=12
STD_GRAD_ACCUM=40
STD_BLOCK=1024
# tokens_per_iter = 40 * 1 * 12 * 1024 = 491520
# 1_000_000_000 / 491520 ≈ 2035 → use 2100
STD_MAX_ITERS=2100

echo "============================================================"
echo " GRID 2  –  Standard scale (124M)  –  FineWeb-Edu  –  1B tk"
echo "============================================================"

if [[ ! -f "data/fineweb/train.bin" ]]; then
    echo ""
    echo "  ERROR: data/fineweb/train.bin not found."
    echo "  Please run the FineWeb-Edu preparation step described in the header"
    echo "  of this script before running Grid 2."
    echo ""
    exit 1
fi

for SEED in "${SEEDS[@]}"; do
    for ARCH in baseline hcrg; do
        run_one "standard" "$ARCH" "$SEED" \
            "$STD_DATASET" "$STD_N_LAYER" "$STD_N_HEAD" "$STD_N_EMBD" \
            "$STD_BATCH" "$STD_GRAD_ACCUM" "$STD_MAX_ITERS" "$STD_BLOCK"
    done
done

# =============================================================================
# Aggregate results
# =============================================================================
echo ""
echo "All runs complete.  Generating results.json …"
$PYTHON analyze_results.py

echo ""
echo "Done.  See results.json"

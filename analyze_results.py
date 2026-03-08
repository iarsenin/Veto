"""
analyze_results.py

Reads all metrics.jsonl files produced by train.py and outputs results.json
with the following structure:

{
  "<grid>/<arch>": {
    "final_val_loss_mean":  float,   // mean of final val loss across 3 seeds
    "final_val_loss_std":   float,   // std  of final val loss across 3 seeds
    "mean_grad_norm":       float,   // mean grad norm over last 1000 train steps
    "spike_count":          int      // # steps where train loss > rolling_mean + 10*rolling_std
  },
  ...
}

Usage:
    python analyze_results.py [--out results.json]
"""

import argparse
import json
import math
import os
import statistics
from collections import defaultdict


GRIDS = ['micro', 'standard']
ARCHS = ['baseline', 'hcrg']
SEEDS = [42, 100, 1337]

GRAD_NORM_TAIL = 1000    # last N train steps for mean_grad_norm
SPIKE_WINDOW = 100       # rolling window length for spike detection
SPIKE_THRESHOLD = 10.0   # # of std-devs above rolling mean → spike


def load_metrics(path: str) -> tuple[list[dict], list[dict]]:
    """Return (train_records, eval_records) from a metrics.jsonl file."""
    train_records: list[dict] = []
    eval_records: list[dict] = []
    if not os.path.exists(path):
        return train_records, eval_records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec_type = rec.get('type')
            if rec_type == 'train':
                train_records.append(rec)
            elif rec_type == 'eval':
                eval_records.append(rec)
    return train_records, eval_records


def compute_spike_count(train_records: list[dict]) -> int:
    """
    Count steps where loss > rolling_mean(prev 100) + 10 * rolling_std(prev 100).
    Rolling statistics use the previous SPIKE_WINDOW steps (not the current one).
    Steps with fewer than 2 prior samples are skipped.
    """
    losses = [r['loss'] for r in train_records if 'loss' in r]
    spike_count = 0
    for i in range(SPIKE_WINDOW, len(losses)):
        window = losses[i - SPIKE_WINDOW: i]
        mu = statistics.mean(window)
        sigma = statistics.stdev(window)
        if sigma > 0 and losses[i] > mu + SPIKE_THRESHOLD * sigma:
            spike_count += 1
    return spike_count


def mean_grad_norm_tail(train_records: list[dict]) -> float:
    """Mean gradient norm over the last GRAD_NORM_TAIL recorded train steps."""
    norms = [r['grad_norm'] for r in train_records if 'grad_norm' in r]
    if not norms:
        return float('nan')
    tail = norms[-GRAD_NORM_TAIL:]
    return statistics.mean(tail)


def final_val_loss(eval_records: list[dict]) -> float:
    """Val loss from the last eval record."""
    records_with_val = [r for r in eval_records if 'val_loss' in r]
    if not records_with_val:
        return float('nan')
    return records_with_val[-1]['val_loss']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results.json', help='Output JSON path')
    parser.add_argument(
        '--base_dir', default='out',
        help='Base directory produced by run_experiments.sh',
    )
    args = parser.parse_args()

    results: dict = {}
    missing_runs: list[str] = []

    for grid in GRIDS:
        for arch in ARCHS:
            key = f"{grid}/{arch}"
            val_losses: list[float] = []
            grad_norms: list[float] = []
            spike_counts: list[int] = []

            for seed in SEEDS:
                run_dir = os.path.join(args.base_dir, grid, arch, f"seed{seed}")
                metrics_file = os.path.join(run_dir, 'metrics.jsonl')

                if not os.path.exists(metrics_file):
                    missing_runs.append(f"{key}/seed{seed}")
                    continue

                train_recs, eval_recs = load_metrics(metrics_file)

                vl = final_val_loss(eval_recs)
                gn = mean_grad_norm_tail(train_recs)
                sc = compute_spike_count(train_recs)

                if not math.isnan(vl):
                    val_losses.append(vl)
                grad_norms.append(gn)
                spike_counts.append(sc)

            def _safe_mean(xs):
                xs = [x for x in xs if not math.isnan(x)]
                return statistics.mean(xs) if xs else float('nan')

            def _safe_std(xs):
                xs = [x for x in xs if not math.isnan(x)]
                if len(xs) < 2:
                    return 0.0
                return statistics.stdev(xs)

            results[key] = {
                "final_val_loss_mean": _safe_mean(val_losses),
                "final_val_loss_std": _safe_std(val_losses),
                "mean_grad_norm": _safe_mean(grad_norms),
                "spike_count": sum(spike_counts),
                "_seeds_found": len(val_losses),
            }

    # Write output
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {args.out}")

    if missing_runs:
        print(f"\nWARNING: {len(missing_runs)} run(s) had no metrics file:")
        for r in missing_runs:
            print(f"  • {r}")

    # Pretty-print summary table
    print("\n{'='*70}")
    header = f"{'Grid/Arch':<25} {'val_loss_mean':>14} {'val_loss_std':>13} {'grad_norm':>11} {'spikes':>8}"
    print(header)
    print('-' * 75)
    for key, v in sorted(results.items()):
        row = (
            f"{key:<25} "
            f"{v['final_val_loss_mean']:>14.4f} "
            f"{v['final_val_loss_std']:>13.4f} "
            f"{v['mean_grad_norm']:>11.4f} "
            f"{v['spike_count']:>8d}"
        )
        print(row)


if __name__ == '__main__':
    main()

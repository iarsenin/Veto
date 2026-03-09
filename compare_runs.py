"""
compare_runs.py

Comprehensive analysis of experiment results across one or two run directories.
Handles deduplication of torch.compile warmup entries, per-seed breakdowns,
convergence trajectories, and cross-run comparison (e.g. bugged vs fixed init).

Usage:
    # Analyse a single run
    python compare_runs.py --run out-fixed

    # Only analyse micro grid (skip missing standard)
    python compare_runs.py --run out-fixed --grids micro

    # Compare two runs side-by-side (if you have a second results dir)
    python compare_runs.py --run out-fixed --baseline-run out-other
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


def load_metrics(path: str) -> tuple[list[dict], list[dict]]:
    """Load and deduplicate metrics.jsonl, returning (train, eval) record lists."""
    train_raw: list[dict] = []
    eval_raw: list[dict] = []
    if not os.path.exists(path):
        return train_raw, eval_raw
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('type') == 'train':
                train_raw.append(rec)
            elif rec.get('type') == 'eval':
                eval_raw.append(rec)

    # Deduplicate: keep last occurrence of each iter (torch.compile warmup
    # re-logs early iters; the second pass has the compiled model's values)
    def _dedup(records):
        by_iter = {}
        for r in records:
            by_iter[r.get('iter', id(r))] = r
        return sorted(by_iter.values(), key=lambda r: r.get('iter', 0))

    return _dedup(train_raw), _dedup(eval_raw)


def safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(xs) if xs else float('nan')


def safe_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def analyse_run(base_dir: str, grids: list[str], label: str = '') -> dict:
    """Analyse all available runs under base_dir. Returns structured results dict."""
    results = {}
    header = f" {label} " if label else ""
    print(f"\n{'=' * 70}")
    print(f"{'ANALYSIS':^70}")
    if header:
        print(f"{header:^70}")
    print(f"  base_dir = {base_dir}")
    print(f"{'=' * 70}")

    for grid in grids:
        for arch in ARCHS:
            key = f"{grid}/{arch}"
            seed_data = {}

            for seed in SEEDS:
                metrics_file = os.path.join(base_dir, grid, arch, f"seed{seed}", 'metrics.jsonl')
                if not os.path.exists(metrics_file):
                    continue

                train_recs, eval_recs = load_metrics(metrics_file)
                if not train_recs and not eval_recs:
                    continue

                last_train_iter = train_recs[-1]['iter'] if train_recs else 0
                final_vl = eval_recs[-1]['val_loss'] if eval_recs else float('nan')
                step0_vl = next((e['val_loss'] for e in eval_recs if e.get('iter') == 0), float('nan'))

                norms = [r['grad_norm'] for r in train_recs if 'grad_norm' in r]
                mean_gn = statistics.mean(norms[-1000:]) if norms else float('nan')

                seed_data[seed] = {
                    'final_val_loss': final_vl,
                    'step0_val_loss': step0_vl,
                    'last_iter': last_train_iter,
                    'mean_grad_norm': mean_gn,
                    'n_train': len(train_recs),
                    'n_eval': len(eval_recs),
                    'eval_trajectory': [(e.get('iter', '?'), e['val_loss']) for e in eval_recs],
                }

            if not seed_data:
                continue

            val_losses = [d['final_val_loss'] for d in seed_data.values()]
            grad_norms = [d['mean_grad_norm'] for d in seed_data.values()]
            step0s = [d['step0_val_loss'] for d in seed_data.values()]

            results[key] = {
                'val_loss_mean': safe_mean(val_losses),
                'val_loss_std': safe_std(val_losses),
                'grad_norm_mean': safe_mean(grad_norms),
                'step0_val_loss_mean': safe_mean(step0s),
                'seeds': seed_data,
                'n_seeds': len(seed_data),
            }

            # Print per-seed detail
            print(f"\n  {key} ({len(seed_data)} seeds)")
            print(f"  {'seed':<8} {'final_vl':>10} {'step0_vl':>10} {'grad_norm':>10} {'iters':>7} {'eval_pts':>9}")
            print(f"  {'-' * 56}")
            for seed in sorted(seed_data):
                d = seed_data[seed]
                print(f"  {seed:<8} {d['final_val_loss']:>10.4f} {d['step0_val_loss']:>10.4f} "
                      f"{d['mean_grad_norm']:>10.4f} {d['last_iter']:>7} {d['n_eval']:>9}")

            print(f"  {'MEAN':<8} {safe_mean(val_losses):>10.4f}{'':>10} "
                  f"{safe_mean(grad_norms):>10.4f}")
            print(f"  {'STD':<8} {safe_std(val_losses):>10.4f}")

    # Summary table
    if results:
        print(f"\n  {'─' * 60}")
        print(f"  {'SUMMARY':^60}")
        print(f"  {'─' * 60}")
        print(f"  {'Grid/Arch':<22} {'val_loss':>12} {'± std':>8} {'grad_norm':>11} {'seeds':>6}")
        print(f"  {'-' * 60}")
        for key in sorted(results):
            r = results[key]
            print(f"  {key:<22} {r['val_loss_mean']:>12.4f} {r['val_loss_std']:>8.4f} "
                  f"{r['grad_norm_mean']:>11.4f} {r['n_seeds']:>6}")

        # HCRG vs baseline delta for each grid
        print(f"\n  {'─' * 60}")
        print(f"  {'HCRG vs BASELINE DELTA':^60}")
        print(f"  {'─' * 60}")
        for grid in grids:
            bkey = f"{grid}/baseline"
            hkey = f"{grid}/hcrg"
            if bkey in results and hkey in results:
                vl_delta = results[hkey]['val_loss_mean'] - results[bkey]['val_loss_mean']
                gn_ratio = results[hkey]['grad_norm_mean'] / results[bkey]['grad_norm_mean']
                s0_delta = results[hkey]['step0_val_loss_mean'] - results[bkey]['step0_val_loss_mean']
                print(f"  {grid}:")
                print(f"    val_loss delta:   {vl_delta:+.4f}  ({'HCRG worse' if vl_delta > 0.01 else 'HCRG better' if vl_delta < -0.01 else 'within noise'})")
                print(f"    grad_norm ratio:  {gn_ratio:.3f}   (1.0 = identical)")
                print(f"    step-0 vl delta:  {s0_delta:+.4f}  ({'MISMATCH' if abs(s0_delta) > 0.1 else 'OK — models start identically'})")

    return results


def compare_runs(run_a: dict, run_b: dict, label_a: str, label_b: str, grids: list[str]):
    """Print side-by-side comparison of two analysed runs."""
    print(f"\n{'=' * 70}")
    print(f"{'CROSS-RUN COMPARISON':^70}")
    print(f"  {label_a}  vs  {label_b}")
    print(f"{'=' * 70}")

    print(f"\n  {'Grid/Arch':<22} {'val_loss (' + label_a + ')':>18} {'val_loss (' + label_b + ')':>18} {'improvement':>13}")
    print(f"  {'-' * 73}")

    for grid in grids:
        for arch in ARCHS:
            key = f"{grid}/{arch}"
            if key in run_a and key in run_b:
                va = run_a[key]['val_loss_mean']
                vb = run_b[key]['val_loss_mean']
                delta = va - vb
                print(f"  {key:<22} {va:>18.4f} {vb:>18.4f} {delta:>+13.4f}")

    print(f"\n  {'Grid/Arch':<22} {'grad_norm (' + label_a + ')':>18} {'grad_norm (' + label_b + ')':>18} {'ratio':>13}")
    print(f"  {'-' * 73}")

    for grid in grids:
        for arch in ARCHS:
            key = f"{grid}/{arch}"
            if key in run_a and key in run_b:
                ga = run_a[key]['grad_norm_mean']
                gb = run_b[key]['grad_norm_mean']
                ratio = ga / gb if gb != 0 else float('inf')
                print(f"  {key:<22} {ga:>18.4f} {gb:>18.4f} {ratio:>13.3f}")

    # HCRG gap comparison
    print(f"\n  {'─' * 60}")
    print(f"  {'HCRG-BASELINE GAP COMPARISON':^60}")
    print(f"  {'─' * 60}")
    for grid in grids:
        bkey = f"{grid}/baseline"
        hkey = f"{grid}/hcrg"
        if all(k in run_a for k in [bkey, hkey]) and all(k in run_b for k in [bkey, hkey]):
            gap_a = run_a[hkey]['val_loss_mean'] - run_a[bkey]['val_loss_mean']
            gap_b = run_b[hkey]['val_loss_mean'] - run_b[bkey]['val_loss_mean']
            print(f"  {grid}:")
            print(f"    {label_a}: {gap_a:+.4f}")
            print(f"    {label_b}: {gap_b:+.4f}")
            if abs(gap_b) > 0:
                pct_remaining = abs(gap_a) / abs(gap_b) * 100
                print(f"    gap reduced by {abs(gap_b) - abs(gap_a):.4f}  ({100 - pct_remaining:.0f}% reduction)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyse and compare experiment results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--run', required=True, help='Primary results directory (e.g. out-fixed)')
    parser.add_argument('--baseline-run', default=None, help='Optional second run to compare against (e.g. out-cloud)')
    parser.add_argument('--grids', nargs='+', default=GRIDS, choices=GRIDS, help='Which grids to analyse')
    args = parser.parse_args()

    run_results = analyse_run(args.run, args.grids, label=args.run)

    if args.baseline_run:
        baseline_results = analyse_run(args.baseline_run, args.grids, label=args.baseline_run)
        compare_runs(run_results, baseline_results, label_a=args.run, label_b=args.baseline_run, grids=args.grids)

    print()


if __name__ == '__main__':
    main()

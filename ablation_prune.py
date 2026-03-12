"""
ablation_prune.py — Static pruning ablation (Phase 2, Task 3)

Tests whether HCRG's learned gating is superior to simply hard-pruning the
heads it identifies as least important.

Loads a trained Baseline checkpoint, identifies the most-suppressed heads from
HCRG gate probing stats, zeroes out those heads in the baseline, and evaluates.

Comparison points:
  1. Unpruned baseline (reference)
  2. Baseline with static pruning (heads zeroed by HCRG ranking)
  3. Baseline with inverse pruning (most-open heads zeroed — sanity check)
  4. HCRG model (from probing stats)

Usage:
    python3.10 ablation_prune.py \
        --baseline-ckpt out-run3/out/standard/baseline/seed42/ckpt.pt \
        --probing-stats probing_stats_seed42.json \
        --val-bin data/fineweb/val.bin \
        --output ablation_results.json
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath('.'))
from model import GPTConfig, GPT


def load_baseline_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_args = ckpt['model_args']
    conf = GPTConfig(**model_args)
    model = GPT(conf)
    state_dict = ckpt['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, ckpt.get('config', {})


def prune_heads(model, heads_to_prune, n_head, n_embd):
    """Zero out c_proj columns for specified heads (list of (layer, head) tuples)."""
    head_size = n_embd // n_head
    model = copy.deepcopy(model)
    for layer_idx, head_idx in heads_to_prune:
        block = model.transformer.h[layer_idx]
        col_start = head_idx * head_size
        col_end = col_start + head_size
        with torch.no_grad():
            block.attn.c_proj.weight[:, col_start:col_end] = 0.0
    return model


@torch.no_grad()
def evaluate(model, val_bin_path, batch_size, block_size, n_batches, device):
    data = np.memmap(val_bin_path, dtype=np.uint16, mode='r')
    losses = []
    for i in range(n_batches):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[j:j+block_size].astype(np.int64)) for j in ix])
        y = torch.stack([torch.from_numpy(data[j+1:j+1+block_size].astype(np.int64)) for j in ix])
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description='Static pruning ablation')
    parser.add_argument('--baseline-ckpt', required=True)
    parser.add_argument('--probing-stats', required=True)
    parser.add_argument('--val-bin', required=True)
    parser.add_argument('--output', default='ablation_results.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--n-batches', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--hcrg-val-loss', type=float, default=None,
                        help='HCRG val loss for comparison (from results.json)')
    args = parser.parse_args()

    print(f'Loading probing stats: {args.probing_stats}')
    with open(args.probing_stats) as f:
        stats = json.load(f)

    n_layer = stats['config']['n_layer']
    n_head = stats['config']['n_head']
    n_embd = n_layer  # placeholder, will get from model
    total_heads = n_layer * n_head

    # Rank heads by mean gate value (ascending = most suppressed first)
    head_ranking = []
    for head_key, head_data in stats['per_head'].items():
        parts = head_key.split('_')
        layer = int(parts[0][1:])
        head = int(parts[1][1:])
        head_ranking.append((layer, head, head_data['mean_gate']))
    head_ranking.sort(key=lambda x: x[2])

    print(f'\nHead ranking (most suppressed first):')
    for i, (l, h, g) in enumerate(head_ranking[:10]):
        print(f'  #{i+1}: L{l}/H{h} mean_gate={g:.4f}')

    print(f'\nLoading baseline model: {args.baseline_ckpt}')
    model, cfg = load_baseline_checkpoint(args.baseline_ckpt, args.device)
    n_embd = cfg.get('n_embd', 768)

    # Evaluate unpruned baseline
    print(f'\nEvaluating unpruned baseline ({args.n_batches} batches)...')
    t0 = time.time()
    baseline_loss = evaluate(model, args.val_bin, args.batch_size, args.block_size,
                             args.n_batches, args.device)
    print(f'  Unpruned baseline: val_loss = {baseline_loss:.4f} ({time.time()-t0:.1f}s)')

    # Test multiple pruning thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
    results = {
        'config': {
            'baseline_ckpt': args.baseline_ckpt,
            'probing_stats': args.probing_stats,
            'n_batches': args.n_batches,
            'batch_size': args.batch_size,
            'total_heads': total_heads,
        },
        'unpruned_baseline': baseline_loss,
        'hcrg_val_loss': args.hcrg_val_loss,
        'pruning_results': [],
        'inverse_pruning_results': [],
    }

    for pct in thresholds:
        n_prune = max(1, int(total_heads * pct))
        heads = [(l, h) for l, h, _ in head_ranking[:n_prune]]
        head_names = [f'L{l}/H{h}' for l, h in heads]

        print(f'\nPruning {n_prune}/{total_heads} heads ({pct:.0%}): {head_names[:5]}...')
        pruned_model = prune_heads(model, heads, n_head, n_embd)
        pruned_model.to(args.device)

        t0 = time.time()
        pruned_loss = evaluate(pruned_model, args.val_bin, args.batch_size, args.block_size,
                               args.n_batches, args.device)
        delta = pruned_loss - baseline_loss
        print(f'  Pruned baseline: val_loss = {pruned_loss:.4f} (delta: {delta:+.4f}, {time.time()-t0:.1f}s)')

        results['pruning_results'].append({
            'threshold': pct,
            'n_heads_pruned': n_prune,
            'heads': head_names,
            'val_loss': pruned_loss,
            'delta_vs_unpruned': delta,
        })
        del pruned_model

    # Inverse pruning: prune the most OPEN heads (sanity check)
    for pct in [0.10, 0.20]:
        n_prune = max(1, int(total_heads * pct))
        heads = [(l, h) for l, h, _ in head_ranking[-n_prune:]]
        head_names = [f'L{l}/H{h}' for l, h in heads]

        print(f'\nInverse pruning {n_prune}/{total_heads} heads ({pct:.0%}, most OPEN): {head_names[:5]}...')
        pruned_model = prune_heads(model, heads, n_head, n_embd)
        pruned_model.to(args.device)

        t0 = time.time()
        pruned_loss = evaluate(pruned_model, args.val_bin, args.batch_size, args.block_size,
                               args.n_batches, args.device)
        delta = pruned_loss - baseline_loss
        print(f'  Inverse pruned: val_loss = {pruned_loss:.4f} (delta: {delta:+.4f}, {time.time()-t0:.1f}s)')

        results['inverse_pruning_results'].append({
            'threshold': pct,
            'n_heads_pruned': n_prune,
            'heads': head_names,
            'val_loss': pruned_loss,
            'delta_vs_unpruned': delta,
        })
        del pruned_model

    # Summary
    print(f'\n{"=" * 70}')
    print(f'ABLATION SUMMARY')
    print(f'{"=" * 70}')
    print(f'  Unpruned baseline:  {baseline_loss:.4f}')
    if args.hcrg_val_loss:
        print(f'  HCRG (reference):   {args.hcrg_val_loss:.4f} (delta: {args.hcrg_val_loss - baseline_loss:+.4f})')
    print(f'\n  Static pruning (suppress most-gated heads):')
    print(f'  {"Threshold":<12} {"Heads":<8} {"Val Loss":<12} {"Δ vs Baseline":<16} {"vs HCRG"}')
    print(f'  {"-" * 60}')
    for r in results['pruning_results']:
        vs_hcrg = ''
        if args.hcrg_val_loss:
            vs_hcrg = 'better' if r['val_loss'] < args.hcrg_val_loss else 'worse'
        print(f'  {r["threshold"]:<12.0%} {r["n_heads_pruned"]:<8} {r["val_loss"]:<12.4f} '
              f'{r["delta_vs_unpruned"]:<+16.4f} {vs_hcrg}')

    print(f'\n  Inverse pruning (suppress most-open heads — sanity check):')
    for r in results['inverse_pruning_results']:
        print(f'  {r["threshold"]:<12.0%} {r["n_heads_pruned"]:<8} {r["val_loss"]:<12.4f} '
              f'{r["delta_vs_unpruned"]:<+16.4f}')

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults written to {args.output}')


if __name__ == '__main__':
    main()

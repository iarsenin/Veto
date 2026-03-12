"""
probe_gates.py — Large-scale gate probing (Phase 2, Task 2)

Loads trained HCRG checkpoints and runs inference on 1000 random sequences
from the FineWeb-Edu validation split to produce statistically robust gate
activation statistics.

Usage:
    python3.10 probe_gates.py --checkpoint out-run3/out/standard/hcrg/seed42/ckpt.pt \
                              --val-bin data/fineweb/val.bin \
                              --n-sequences 1000 --seq-len 1024 \
                              --output probing_stats_seed42.json

    # All 3 seeds:
    for s in 42 100 1337; do
        python3.10 probe_gates.py \
            --checkpoint out-run3/out/standard/hcrg/seed${s}/ckpt.pt \
            --val-bin data/fineweb/val.bin \
            --output probing_stats_seed${s}.json
    done
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath('.'))
from custom_model import GPTConfig, GPT


def load_checkpoint(path, device='cpu'):
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


def main():
    parser = argparse.ArgumentParser(description='Large-scale HCRG gate probing')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--val-bin', required=True)
    parser.add_argument('--n-sequences', type=int, default=1000)
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--output', default='probing_stats.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    print(f'Loading checkpoint: {args.checkpoint}')
    model, cfg = load_checkpoint(args.checkpoint, args.device)
    n_layer = cfg.get('n_layer', len(model.transformer.h))
    n_head = cfg.get('n_head', model.transformer.h[0].attn.n_head)
    print(f'Model: {n_layer}L, {n_head}H')

    print(f'Loading validation data: {args.val_bin}')
    data = np.memmap(args.val_bin, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    max_start = total_tokens - args.seq_len
    print(f'  {total_tokens:,} tokens, max_start={max_start:,}')

    rng = np.random.RandomState(42)
    starts = rng.randint(0, max_start, size=args.n_sequences)

    # Storage: per-sequence, per-layer, per-head statistics
    # For memory efficiency, accumulate running stats instead of storing all activations
    # Per (layer, head): track mean gate per sequence, and per-position variance per sequence
    seq_means = np.zeros((args.n_sequences, n_layer, n_head), dtype=np.float32)
    seq_within_stds = np.zeros((args.n_sequences, n_layer, n_head), dtype=np.float32)

    # Gate percentiles — accumulate all per-layer values
    all_gate_values_per_layer = [[] for _ in range(n_layer)]

    n_batches = (args.n_sequences + args.batch_size - 1) // args.batch_size
    t0 = time.time()

    for batch_idx in range(n_batches):
        s = batch_idx * args.batch_size
        e = min(s + args.batch_size, args.n_sequences)
        batch_starts = starts[s:e]

        seqs = np.stack([data[st:st + args.seq_len].astype(np.int64) for st in batch_starts])
        x = torch.tensor(seqs, dtype=torch.long, device=args.device)

        gate_acts = {}
        hooks = []
        for i, block in enumerate(model.transformer.h):
            def make_hook(layer_idx):
                def hook_fn(module, inp, out):
                    gate_acts[layer_idx] = torch.sigmoid(out).detach().cpu().numpy()
                return hook_fn
            h = block.gate_proj.register_forward_hook(make_hook(i))
            hooks.append(h)

        with torch.no_grad():
            model(x)

        for h in hooks:
            h.remove()

        for i in range(e - s):
            seq_idx = s + i
            for layer in range(n_layer):
                g = gate_acts[layer][i]  # (seq_len, n_head)
                seq_means[seq_idx, layer] = g.mean(axis=0)
                seq_within_stds[seq_idx, layer] = g.std(axis=0)

                if batch_idx % 50 == 0 and i == 0:
                    all_gate_values_per_layer[layer].append(g.flatten())

        if (batch_idx + 1) % 25 == 0 or batch_idx == 0:
            elapsed = time.time() - t0
            rate = (batch_idx + 1) / elapsed
            eta = (n_batches - batch_idx - 1) / rate if rate > 0 else 0
            print(f'  batch {batch_idx + 1}/{n_batches}, '
                  f'{elapsed:.0f}s elapsed, ETA {eta:.0f}s', flush=True)

    elapsed = time.time() - t0
    print(f'Inference complete: {args.n_sequences} sequences in {elapsed:.1f}s')

    # Compute aggregate statistics
    results = {
        'config': {
            'checkpoint': args.checkpoint,
            'n_sequences': args.n_sequences,
            'seq_len': args.seq_len,
            'n_layer': n_layer,
            'n_head': n_head,
        },
        'per_layer': {},
        'per_head': {},
        'summary': {},
    }

    global_mean_gates = seq_means.mean(axis=0)  # (n_layer, n_head)
    cross_seq_std = seq_means.std(axis=0)  # (n_layer, n_head)
    mean_within_std = seq_within_stds.mean(axis=0)  # (n_layer, n_head)

    for layer in range(n_layer):
        layer_key = f'L{layer}'
        layer_means = seq_means[:, layer, :]  # (n_seq, n_head)
        layer_within = seq_within_stds[:, layer, :]

        results['per_layer'][layer_key] = {
            'mean_gate': float(global_mean_gates[layer].mean()),
            'std_across_heads': float(global_mean_gates[layer].std()),
            'mean_within_seq_std': float(mean_within_std[layer].mean()),
            'mean_cross_seq_std': float(cross_seq_std[layer].mean()),
        }

        for head in range(n_head):
            head_key = f'L{layer}_H{head}'
            results['per_head'][head_key] = {
                'mean_gate': float(global_mean_gates[layer, head]),
                'within_seq_std': float(mean_within_std[layer, head]),
                'cross_seq_std': float(cross_seq_std[layer, head]),
            }

    # Gate value percentiles from sampled sequences
    for layer in range(n_layer):
        if all_gate_values_per_layer[layer]:
            vals = np.concatenate(all_gate_values_per_layer[layer])
            pcts = np.percentile(vals, [5, 25, 50, 75, 95])
            results['per_layer'][f'L{layer}']['percentiles'] = {
                'p5': float(pcts[0]), 'p25': float(pcts[1]), 'p50': float(pcts[2]),
                'p75': float(pcts[3]), 'p95': float(pcts[4]),
            }

    # Summary
    all_mean_gates = global_mean_gates.flatten()
    all_within = mean_within_std.flatten()
    all_cross = cross_seq_std.flatten()

    results['summary'] = {
        'overall_mean_gate': float(all_mean_gates.mean()),
        'overall_std': float(all_mean_gates.std()),
        'mean_within_seq_std': float(all_within.mean()),
        'mean_cross_seq_std': float(all_cross.mean()),
        'fraction_below_0.5': float((all_mean_gates < 0.5).mean()),
        'fraction_below_0.9': float((all_mean_gates < 0.9).mean()),
        'fraction_above_0.99': float((all_mean_gates > 0.99).mean()),
        'heads_with_mean_below_0.9': int((all_mean_gates < 0.9).sum()),
        'total_heads': int(len(all_mean_gates)),
    }

    # Top suppressed and most variable heads
    flat_idx_suppressed = np.argsort(all_mean_gates)
    top_suppressed = []
    for idx in flat_idx_suppressed[:15]:
        l, h = divmod(idx, n_head)
        top_suppressed.append({
            'head': f'L{l}_H{h}',
            'mean_gate': float(global_mean_gates[l, h]),
            'within_std': float(mean_within_std[l, h]),
            'cross_std': float(cross_seq_std[l, h]),
        })
    results['top_suppressed'] = top_suppressed

    flat_idx_variable = np.argsort(all_within)[::-1]
    top_variable = []
    for idx in flat_idx_variable[:15]:
        l, h = divmod(idx, n_head)
        top_variable.append({
            'head': f'L{l}_H{h}',
            'mean_gate': float(global_mean_gates[l, h]),
            'within_std': float(mean_within_std[l, h]),
            'cross_std': float(cross_seq_std[l, h]),
        })
    results['top_variable'] = top_variable

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results written to {args.output}')

    # Print summary
    print(f'\n{"=" * 60}')
    print(f'GATE PROBING SUMMARY ({args.n_sequences} sequences × {args.seq_len} tokens)')
    print(f'{"=" * 60}')
    print(f'  Overall mean gate:        {results["summary"]["overall_mean_gate"]:.4f}')
    print(f'  Mean within-seq std:      {results["summary"]["mean_within_seq_std"]:.4f}')
    print(f'  Mean cross-seq std:       {results["summary"]["mean_cross_seq_std"]:.4f}')
    print(f'  Heads with mean < 0.9:    {results["summary"]["heads_with_mean_below_0.9"]}/{results["summary"]["total_heads"]}')
    print(f'  Fraction mean < 0.5:      {results["summary"]["fraction_below_0.5"]:.1%}')
    print(f'\n  Per-layer mean gate:')
    for layer in range(n_layer):
        lk = f'L{layer}'
        r = results['per_layer'][lk]
        print(f'    {lk}: {r["mean_gate"]:.4f}  within_std={r["mean_within_seq_std"]:.4f}  cross_std={r["mean_cross_seq_std"]:.4f}')
    print(f'\n  Top-5 most suppressed:')
    for i, h in enumerate(results['top_suppressed'][:5]):
        print(f'    #{i+1}: {h["head"]}  mean={h["mean_gate"]:.4f}  within_std={h["within_std"]:.4f}')
    print(f'\n  Top-5 most variable:')
    for i, h in enumerate(results['top_variable'][:5]):
        print(f'    #{i+1}: {h["head"]}  mean={h["mean_gate"]:.4f}  within_std={h["within_std"]:.4f}')


if __name__ == '__main__':
    main()

"""
full_analysis.py

Comprehensive analysis of HCRG experiments:
  1. Gate sparsity probing (P4) — loads HCRG checkpoints, runs forward passes,
     analyzes gate activation patterns across prompts
  2. Convergence trajectory — eval loss at each checkpoint across all seeds
  3. Cross-run reproducibility — compares Run 2 (A100) vs Run 3 (RTX 6000 Ada)
  4. Per-seed gate comparison — checks if gate patterns are consistent across seeds

Usage:
    python3.10 full_analysis.py --run out-run3/out [--run2 out-fixed]
"""

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath('.'))
from custom_model import GPTConfig, GPT

try:
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
except ImportError:
    enc = None
    print("WARNING: tiktoken not available, using dummy tokenizer")

SEEDS = [42, 100, 1337]

PROMPTS = {
    'story': (
        "Once upon a time, there was a little girl named Lily. She loved to play "
        "in the garden with her dog Max. One sunny day, they found a beautiful "
        "butterfly sitting on a flower. Lily wanted to catch it, but Max barked "
        "and the butterfly flew away. Lily was sad, but then she saw a rainbow "
        "in the sky and smiled. She knew that tomorrow would be another adventure."
    ),
    'factual': (
        "The mitochondria is the powerhouse of the cell. It produces adenosine "
        "triphosphate through oxidative phosphorylation. The process involves "
        "the electron transport chain located in the inner mitochondrial membrane. "
        "Glucose is first broken down through glycolysis in the cytoplasm, producing "
        "pyruvate which enters the mitochondrial matrix for the citric acid cycle."
    ),
    'code': (
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    "
        "a, b = 0, 1\n    for _ in range(2, n + 1):\n        "
        "a, b = b, a + b\n    return b\n\n"
        "print([fibonacci(i) for i in range(20)])"
    ),
    'repetitive': 'the cat sat on the mat. ' * 20,
}


def encode_text(text, max_len=512):
    if enc is not None:
        return enc.encode(text)[:max_len]
    return list(range(min(max_len, len(text.split()))))


def load_hcrg_checkpoint(path, device='cpu'):
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


def collect_gates(model, token_ids, device='cpu'):
    gate_activations = {}
    hooks = []

    for i, block in enumerate(model.transformer.h):
        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                gate_activations[layer_idx] = torch.sigmoid(out).detach().cpu()
            return hook_fn
        h = block.gate_proj.register_forward_hook(make_hook(i))
        hooks.append(h)

    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        model(idx)

    for h in hooks:
        h.remove()

    return {k: v.squeeze(0) for k, v in gate_activations.items()}


def load_metrics(path):
    train_raw, eval_raw = [], []
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

    def _dedup(records):
        by_iter = {}
        for r in records:
            by_iter[r.get('iter', id(r))] = r
        return sorted(by_iter.values(), key=lambda r: r.get('iter', 0))

    return _dedup(train_raw), _dedup(eval_raw)


# ═══════════════════════════════════════════════════════════════════════
#  1. GATE PROBING (P4)
# ═══════════════════════════════════════════════════════════════════════

def gate_weight_analysis(model, n_layer, label=''):
    print(f'\n  {label} — Learned gate_proj parameters:')
    print(f'  {"Layer":<8} {"Bias (raw)":<50} {"Bias (sigmoid)":<50} {"W norm":<10}')
    print(f'  {"-" * 120}')
    all_biases = []
    for i, block in enumerate(model.transformer.h):
        gp = block.gate_proj
        bias = gp.bias.data.cpu().numpy()
        sig_bias = 1.0 / (1.0 + np.exp(-bias))
        w_norm = gp.weight.data.cpu().norm().item()
        bias_str = ', '.join(f'{b:+.3f}' for b in bias)
        sig_str = ', '.join(f'{s:.3f}' for s in sig_bias)
        print(f'  L{i:<7} [{bias_str}]  [{sig_str}]  {w_norm:.4f}')
        all_biases.extend(bias.tolist())

    all_biases = np.array(all_biases)
    print(f'\n  Bias drift from init (+5.0):')
    print(f'    Mean bias: {all_biases.mean():.3f} (init: +5.000, delta: {all_biases.mean() - 5:.3f})')
    print(f'    Std of biases: {all_biases.std():.3f}')
    print(f'    Range: [{all_biases.min():.3f}, {all_biases.max():.3f}]')
    return all_biases


def probe_gates_single_model(model, n_layer, n_head, label=''):
    print(f'\n{"=" * 70}')
    print(f'  GATE PROBING — {label}')
    print(f'{"=" * 70}')

    all_gates = {}
    for pname, text in PROMPTS.items():
        tokens = encode_text(text)
        all_gates[pname] = collect_gates(model, tokens)
        print(f'  {pname}: {len(tokens)} tokens')

    # Overall statistics
    all_vals = []
    for prompt_gates in all_gates.values():
        for layer_gates in prompt_gates.values():
            all_vals.append(layer_gates.numpy().flatten())
    all_vals = np.concatenate(all_vals)

    print(f'\n  Gate value statistics (across all prompts, layers, heads, positions):')
    print(f'    Mean:                       {all_vals.mean():.4f}')
    print(f'    Std:                        {all_vals.std():.4f}')
    print(f'    Min:                        {all_vals.min():.4f}')
    print(f'    Max:                        {all_vals.max():.4f}')
    print(f'    Median:                     {np.median(all_vals):.4f}')
    print(f'    < 0.5 (actively suppress):  {(all_vals < 0.5).mean() * 100:.2f}%')
    print(f'    < 0.9:                      {(all_vals < 0.9).mean() * 100:.2f}%')
    print(f'    < 0.95:                     {(all_vals < 0.95).mean() * 100:.2f}%')
    print(f'    > 0.99 (fully open):        {(all_vals > 0.99).mean() * 100:.2f}%')

    # Per-layer analysis
    print(f'\n  Per-layer mean gate value:')
    print(f'  {"Layer":<8} {"Mean":<10} {"Std":<10} {"Min":<10} {"Max":<10} {"<0.5 %":<10} {"<0.9 %":<10}')
    print(f'  {"-" * 60}')
    for layer in range(n_layer):
        layer_vals = []
        for prompt_gates in all_gates.values():
            layer_vals.append(prompt_gates[layer].numpy().flatten())
        layer_vals = np.concatenate(layer_vals)
        print(f'  L{layer:<7} {layer_vals.mean():<10.4f} {layer_vals.std():<10.4f} '
              f'{layer_vals.min():<10.4f} {layer_vals.max():<10.4f} '
              f'{(layer_vals < 0.5).mean() * 100:<10.2f} {(layer_vals < 0.9).mean() * 100:<10.2f}')

    # Per-head analysis (mean across prompts and positions)
    print(f'\n  Per-head mean gate value (averaged across all prompts/positions):')
    head_means = np.zeros((n_layer, n_head))
    head_stds = np.zeros((n_layer, n_head))
    for layer in range(n_layer):
        for head in range(n_head):
            vals = []
            for prompt_gates in all_gates.values():
                vals.append(prompt_gates[layer][:, head].numpy())
            vals = np.concatenate(vals)
            head_means[layer, head] = vals.mean()
            head_stds[layer, head] = vals.std()

    header = '  ' + 'Layer'.ljust(8) + ''.join(f'H{h:<7}' for h in range(min(n_head, 12)))
    print(header)
    print(f'  {"-" * len(header)}')
    for layer in range(n_layer):
        row = f'  L{layer:<7}'
        for head in range(min(n_head, 12)):
            row += f'{head_means[layer, head]:<8.4f}'
        print(row)

    # Context dependence: cross-prompt variation
    print(f'\n  Context dependence (cross-prompt variation of per-head mean gate):')
    per_prompt_means = []
    for pname, prompt_gates in all_gates.items():
        means = np.zeros((n_layer, n_head))
        for layer in range(n_layer):
            means[layer] = prompt_gates[layer].numpy().mean(axis=0)
        per_prompt_means.append(means)
    stacked = np.stack(per_prompt_means)
    cross_std = stacked.std(axis=0)
    print(f'    Mean cross-prompt std:  {cross_std.mean():.4f}')
    print(f'    Max cross-prompt std:   {cross_std.max():.4f} at L{np.unravel_index(cross_std.argmax(), cross_std.shape)[0]}/H{np.unravel_index(cross_std.argmax(), cross_std.shape)[1]}')

    # Within-prompt variation (context dependence per position)
    within_stds = []
    for pname, prompt_gates in all_gates.items():
        for layer in range(n_layer):
            within_stds.append(prompt_gates[layer].numpy().std(axis=0).mean())
    print(f'    Mean within-prompt std: {np.mean(within_stds):.4f}')

    # Most suppressed heads
    flat_idx = np.argsort(head_means.flatten())
    print(f'\n  Top-10 most suppressed heads (lowest mean gate):')
    for rank, idx in enumerate(flat_idx[:10]):
        l, h = divmod(idx, n_head)
        print(f'    #{rank + 1}: L{l}/H{h}  mean={head_means[l, h]:.4f}  std={head_stds[l, h]:.4f}')

    # Most variable heads (highest within-prompt std)
    var_per_head = np.zeros((n_layer, n_head))
    for layer in range(n_layer):
        for head in range(n_head):
            vals = []
            for prompt_gates in all_gates.values():
                vals.append(prompt_gates[layer][:, head].numpy().std())
            var_per_head[layer, head] = np.mean(vals)

    flat_var_idx = np.argsort(var_per_head.flatten())[::-1]
    print(f'\n  Top-10 most context-variable heads (highest within-prompt std):')
    for rank, idx in enumerate(flat_var_idx[:10]):
        l, h = divmod(idx, n_head)
        print(f'    #{rank + 1}: L{l}/H{h}  mean_gate={head_means[l, h]:.4f}  within_std={var_per_head[l, h]:.4f}')

    # Verdict
    print(f'\n  {"─" * 60}')
    print(f'  VERDICT for {label}:')
    verdicts = []
    if all_vals.mean() > 0.99:
        verdicts.append('Gates near-fully open — minimal learning')
    elif all_vals.mean() > 0.95:
        verdicts.append('Gates mostly open with modest suppression')
    elif all_vals.mean() > 0.8:
        verdicts.append('Gates show moderate suppression activity')
    else:
        verdicts.append('Gates show SUBSTANTIAL suppression activity')

    if np.mean(within_stds) > 0.05:
        verdicts.append('Context-DEPENDENT (significant within-prompt variation)')
    elif np.mean(within_stds) > 0.02:
        verdicts.append('Mildly context-dependent')
    else:
        verdicts.append('Low within-prompt variation — near-static gates')

    if cross_std.mean() > 0.02:
        verdicts.append('Input-DEPENDENT (different prompts → different patterns)')
    elif cross_std.mean() > 0.005:
        verdicts.append('Mildly input-dependent')
    else:
        verdicts.append('Low cross-prompt variation — gates may be fixed values')

    suppressed_count = (head_means < 0.9).sum()
    total_heads = n_layer * n_head
    if suppressed_count > 0:
        verdicts.append(f'{suppressed_count}/{total_heads} heads have mean gate < 0.9 (sparse suppression)')
    else:
        verdicts.append('No heads with mean gate < 0.9')

    for v in verdicts:
        print(f'    • {v}')

    return all_gates, head_means, head_stds, all_vals


# ═══════════════════════════════════════════════════════════════════════
#  2. CONVERGENCE TRAJECTORIES
# ═══════════════════════════════════════════════════════════════════════

def convergence_analysis(base_dir, grid='standard'):
    print(f'\n{"=" * 70}')
    print(f'  CONVERGENCE TRAJECTORIES — {grid}')
    print(f'  base_dir = {base_dir}')
    print(f'{"=" * 70}')

    all_evals = defaultdict(lambda: defaultdict(list))

    for arch in ['baseline', 'hcrg']:
        for seed in SEEDS:
            mf = os.path.join(base_dir, grid, arch, f'seed{seed}', 'metrics.jsonl')
            _, eval_recs = load_metrics(mf)
            for e in eval_recs:
                it = e.get('iter', 0)
                all_evals[arch][it].append(e['val_loss'])

    iters_b = sorted(all_evals['baseline'].keys())
    iters_h = sorted(all_evals['hcrg'].keys())
    common_iters = sorted(set(iters_b) & set(iters_h))

    if not common_iters:
        print('  No common eval iterations found.')
        return

    print(f'\n  {"Iter":<8} {"Baseline (mean)":<18} {"HCRG (mean)":<18} {"Delta":<12} {"Note"}')
    print(f'  {"-" * 70}')

    for it in common_iters:
        b_mean = statistics.mean(all_evals['baseline'][it])
        h_mean = statistics.mean(all_evals['hcrg'][it])
        delta = h_mean - b_mean
        b_std = statistics.stdev(all_evals['baseline'][it]) if len(all_evals['baseline'][it]) > 1 else 0
        h_std = statistics.stdev(all_evals['hcrg'][it]) if len(all_evals['hcrg'][it]) > 1 else 0
        note = ''
        if abs(delta) < 0.005:
            note = 'tied'
        elif delta < 0:
            note = 'HCRG leads'
        else:
            note = 'baseline leads'

        print(f'  {it:<8} {b_mean:<10.4f}±{b_std:<6.4f} {h_mean:<10.4f}±{h_std:<6.4f} {delta:<+12.4f} {note}')

    # Final delta
    final_b = statistics.mean(all_evals['baseline'][common_iters[-1]])
    final_h = statistics.mean(all_evals['hcrg'][common_iters[-1]])
    print(f'\n  Final delta at iter {common_iters[-1]}: {final_h - final_b:+.4f}')

    # Per-seed final values
    print(f'\n  Per-seed final val loss (iter {common_iters[-1]}):')
    print(f'  {"Seed":<8} {"Baseline":<12} {"HCRG":<12} {"Delta":<12}')
    print(f'  {"-" * 44}')
    for seed in SEEDS:
        mf_b = os.path.join(base_dir, grid, 'baseline', f'seed{seed}', 'metrics.jsonl')
        mf_h = os.path.join(base_dir, grid, 'hcrg', f'seed{seed}', 'metrics.jsonl')
        _, eval_b = load_metrics(mf_b)
        _, eval_h = load_metrics(mf_h)
        if eval_b and eval_h:
            vl_b = eval_b[-1]['val_loss']
            vl_h = eval_h[-1]['val_loss']
            print(f'  {seed:<8} {vl_b:<12.4f} {vl_h:<12.4f} {vl_h - vl_b:<+12.4f}')


# ═══════════════════════════════════════════════════════════════════════
#  3. CROSS-RUN REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════

def cross_run_analysis(dir_a, dir_b, label_a='Run A', label_b='Run B'):
    print(f'\n{"=" * 70}')
    print(f'  CROSS-RUN REPRODUCIBILITY')
    print(f'  {label_a}: {dir_a}')
    print(f'  {label_b}: {dir_b}')
    print(f'{"=" * 70}')

    for grid in ['micro', 'standard']:
        print(f'\n  --- {grid} ---')
        print(f'  {"Arch":<12} {"Seed":<8} {label_a + " vl":<14} {label_b + " vl":<14} {"Δ":<10}')
        print(f'  {"-" * 60}')

        for arch in ['baseline', 'hcrg']:
            deltas = []
            for seed in SEEDS:
                mf_a = os.path.join(dir_a, grid, arch, f'seed{seed}', 'metrics.jsonl')
                mf_b = os.path.join(dir_b, grid, arch, f'seed{seed}', 'metrics.jsonl')
                _, eval_a = load_metrics(mf_a)
                _, eval_b = load_metrics(mf_b)
                if eval_a and eval_b:
                    vl_a = eval_a[-1]['val_loss']
                    vl_b = eval_b[-1]['val_loss']
                    delta = vl_a - vl_b
                    deltas.append(delta)
                    print(f'  {arch:<12} {seed:<8} {vl_a:<14.4f} {vl_b:<14.4f} {delta:<+10.4f}')
            if deltas:
                print(f'  {arch:<12} {"MEAN":<8} {"":14} {"":14} {statistics.mean(deltas):<+10.4f}')

        # HCRG delta comparison
        for label, d in [(label_a, dir_a), (label_b, dir_b)]:
            b_vals, h_vals = [], []
            for seed in SEEDS:
                _, eval_b = load_metrics(os.path.join(d, grid, 'baseline', f'seed{seed}', 'metrics.jsonl'))
                _, eval_h = load_metrics(os.path.join(d, grid, 'hcrg', f'seed{seed}', 'metrics.jsonl'))
                if eval_b:
                    b_vals.append(eval_b[-1]['val_loss'])
                if eval_h:
                    h_vals.append(eval_h[-1]['val_loss'])
            if b_vals and h_vals:
                gap = statistics.mean(h_vals) - statistics.mean(b_vals)
                print(f'  HCRG-baseline gap ({label}): {gap:+.4f}')


# ═══════════════════════════════════════════════════════════════════════
#  4. CROSS-SEED GATE CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════

def cross_seed_gate_analysis(base_dir, grid='standard'):
    print(f'\n{"=" * 70}')
    print(f'  CROSS-SEED GATE CONSISTENCY — {grid}/hcrg')
    print(f'{"=" * 70}')

    models = {}
    for seed in SEEDS:
        ckpt_path = os.path.join(base_dir, grid, 'hcrg', f'seed{seed}', 'ckpt.pt')
        if os.path.exists(ckpt_path):
            print(f'  Loading seed {seed} from {ckpt_path}...')
            model, _ = load_hcrg_checkpoint(ckpt_path)
            models[seed] = model

    if len(models) < 2:
        print('  Need at least 2 seeds for comparison.')
        return

    prompt_text = PROMPTS['factual']
    tokens = encode_text(prompt_text)

    seed_head_means = {}
    for seed, model in models.items():
        gates = collect_gates(model, tokens)
        n_layer = len(gates)
        n_head = gates[0].shape[1]
        means = np.zeros((n_layer, n_head))
        for layer in range(n_layer):
            means[layer] = gates[layer].numpy().mean(axis=0)
        seed_head_means[seed] = means

    seeds_list = sorted(seed_head_means.keys())
    print(f'\n  Correlation of per-head mean gate values between seeds (on "factual" prompt):')
    for i, s1 in enumerate(seeds_list):
        for s2 in seeds_list[i + 1:]:
            flat1 = seed_head_means[s1].flatten()
            flat2 = seed_head_means[s2].flatten()
            corr = np.corrcoef(flat1, flat2)[0, 1]
            print(f'    seed {s1} vs seed {s2}: r = {corr:.4f}')

    # Which heads are consistently suppressed across seeds?
    all_means = np.stack([seed_head_means[s] for s in seeds_list])
    mean_across_seeds = all_means.mean(axis=0)
    std_across_seeds = all_means.std(axis=0)

    print(f'\n  Heads with mean gate < 0.95 consistently across seeds:')
    n_layer, n_head = mean_across_seeds.shape
    found = False
    for l in range(n_layer):
        for h in range(n_head):
            if mean_across_seeds[l, h] < 0.95:
                seed_vals = [f'{seed_head_means[s][l, h]:.4f}' for s in seeds_list]
                print(f'    L{l}/H{h}: mean={mean_across_seeds[l, h]:.4f} '
                      f'std={std_across_seeds[l, h]:.4f} '
                      f'per-seed=[{", ".join(seed_vals)}]')
                found = True
    if not found:
        print('    (none — all heads have mean gate >= 0.95)')

    # Per-layer summary
    print(f'\n  Per-layer mean gate (averaged across heads and seeds):')
    for l in range(n_layer):
        print(f'    L{l}: {mean_across_seeds[l].mean():.4f} (std across heads: {mean_across_seeds[l].std():.4f})')


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Full HCRG experiment analysis')
    parser.add_argument('--run', required=True, help='Primary results directory')
    parser.add_argument('--run2', default=None, help='Second run for cross-run comparison')
    args = parser.parse_args()

    print('=' * 70)
    print('  FULL HCRG EXPERIMENT ANALYSIS')
    print(f'  Primary run: {args.run}')
    if args.run2:
        print(f'  Comparison run: {args.run2}')
    print('=' * 70)

    # ── 1. Gate probing on standard HCRG (seed 42) ──
    std_ckpt = os.path.join(args.run, 'standard', 'hcrg', 'seed42', 'ckpt.pt')
    if os.path.exists(std_ckpt):
        print('\nLoading standard HCRG model (seed 42)...')
        std_model, std_cfg = load_hcrg_checkpoint(std_ckpt)
        n_layer = std_cfg.get('n_layer', 12)
        n_head = std_cfg.get('n_head', 12)

        probe_gates_single_model(std_model, n_layer, n_head,
                                 label=f'Standard HCRG (seed 42)')
        gate_weight_analysis(std_model, n_layer, label='Standard HCRG seed42')
        del std_model
    else:
        print(f'\n  Standard HCRG checkpoint not found at {std_ckpt}')

    # ── 1b. Gate probing on micro HCRG (seed 42) for comparison ──
    micro_ckpt = os.path.join(args.run, 'micro', 'hcrg', 'seed42', 'ckpt.pt')
    if os.path.exists(micro_ckpt):
        print('\nLoading micro HCRG model (seed 42)...')
        micro_model, micro_cfg = load_hcrg_checkpoint(micro_ckpt)
        n_layer_m = micro_cfg.get('n_layer', 4)
        n_head_m = micro_cfg.get('n_head', 4)

        probe_gates_single_model(micro_model, n_layer_m, n_head_m,
                                 label=f'Micro HCRG (seed 42)')
        gate_weight_analysis(micro_model, n_layer_m, label='Micro HCRG seed42')
        del micro_model
    else:
        print(f'\n  Micro HCRG checkpoint not found at {micro_ckpt}')

    # ── 2. Convergence trajectories ──
    convergence_analysis(args.run, grid='micro')
    convergence_analysis(args.run, grid='standard')

    # ── 3. Cross-run reproducibility ──
    if args.run2 and os.path.exists(args.run2):
        cross_run_analysis(args.run, args.run2,
                           label_a='Run3 (RTX6000)', label_b='Run2 (A100)')

    # ── 4. Cross-seed gate consistency (standard) ──
    cross_seed_gate_analysis(args.run, grid='standard')

    print(f'\n{"=" * 70}')
    print(f'  ANALYSIS COMPLETE')
    print(f'{"=" * 70}\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
scripts/generate_ablation_tables.py — Generate LaTeX ablation tables.

Reads logs/noise_ablation/summary.log and produces two tables:
  - Noise ablation    (rows = σ values,         cols = MNIST / FashionMNIST × AC F1 / ASR)
  - Pretrain ablation (rows = pretrain epochs,   cols = same)

Each cell shows mean±std across seeds. Bold marks the best AC F1 per dataset column.

Usage:
    python scripts/generate_ablation_tables.py              # print both tables (k=2)
    python scripts/generate_ablation_tables.py --k 4
    python scripts/generate_ablation_tables.py --out results/tables/
"""

import argparse
import os
import re
import sys
import numpy as np

SUMMARY_LOG     = 'logs/noise_ablation/summary.log'
NOISE_LEVELS    = ['0.00', '0.01', '0.05', '0.10', '0.20']
PRETRAIN_EPOCHS = [1, 5, 10]
DEFAULT_K       = 2


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_log(path):
    """
    Returns:
      metrics[prefix] = list of {k: {ac_acc, ac_f1, raw_acc, raw_f1}}  (one per seed)
      asr_map[prefix] = list of float (ASR %, one per seed)
    prefix examples: 'noise_fmnist_0.05_r0.15', 'pretrain_mnist_5ep_r0.15'
    """
    metrics = {}
    asr_map = {}

    if not os.path.exists(path):
        print(f'Warning: {path} not found', file=sys.stderr)
        return metrics, asr_map

    with open(path) as f:
        text = f.read()

    for block_label, block_body in re.findall(
        r'── ([^\n]+) ──\n(.*?)(?=── |\Z)', text, re.DOTALL
    ):
        block_label = block_label.strip()
        m = re.match(
            r'((noise|pretrain)_(?:fmnist|mnist)_[^_]+(?:ep)?_r[0-9.]+)_seed(\d+)',
            block_label
        )
        if not m:
            continue
        prefix = m.group(1)

        asr_match = re.search(r'asr \(avg\):\s*([\d.]+)%', block_body)
        if asr_match:
            asr_map.setdefault(prefix, []).append(float(asr_match.group(1)))

        per_k = {}
        for row in re.finditer(
            r'k=(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%', block_body
        ):
            k = int(row.group(1))
            per_k[k] = {
                'ac_acc':  float(row.group(2)),
                'ac_f1':   float(row.group(3)),
                'raw_acc': float(row.group(4)),
                'raw_f1':  float(row.group(5)),
            }
        if per_k:
            metrics.setdefault(prefix, []).append(per_k)

    return metrics, asr_map


def metric_stats(prefix, metrics, k, key):
    vals = [e[k][key] for e in metrics.get(prefix, []) if k in e]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def asr_stats(prefix, asr_map):
    vals = asr_map.get(prefix, [])
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(mean, std, bold=False):
    s = rf'{mean:.2f}{{\tiny$\pm${std:.2f}}}'
    return r'\textbf{' + s + '}' if bold else s


def fmt_asr(prefix, asr_map):
    v = asr_stats(prefix, asr_map)
    if v is None:
        return '--'
    return rf'{v[0]:.2f}{{\tiny$\pm${v[1]:.2f}}}'


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def make_table(label, caption, cond_label, row_display, row_prefixes,
               metrics, asr_map, k):
    """
    One combined MNIST + FashionMNIST table.
    Columns: condition | MNIST AC F1 | MNIST ASR | FashionMNIST AC F1 | FashionMNIST ASR
    """
    ds_labels = ['MNIST', 'FashionMNIST']

    # Find best AC F1 per dataset column (for bolding)
    best_ac = {}
    for ds_label, pfx_list in zip(ds_labels, row_prefixes):
        best = -1.0
        for pfx in pfx_list:
            v = metric_stats(pfx, metrics, k, 'ac_f1')
            if v and v[0] > best:
                best = v[0]
        best_ac[ds_label] = best

    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{|c|cc|cc|}',
        r'\hline',
        (r'& \multicolumn{2}{c|}{\textbf{MNIST}} '
         r'& \multicolumn{2}{c|}{\textbf{FashionMNIST}} \\'),
        (cond_label
         + r' & \textbf{AC F1} & \textbf{ASR}'
         + r' & \textbf{AC F1} & \textbf{ASR} \\'),
        r'\hline',
    ]

    for display, pfx_mnist, pfx_fmnist in zip(row_display, row_prefixes[0], row_prefixes[1]):
        cells = [display]
        for ds_label, pfx in zip(ds_labels, [pfx_mnist, pfx_fmnist]):
            ac = metric_stats(pfx, metrics, k, 'ac_f1')
            ac_str = fmt(*ac, bold=(ac is not None and abs(ac[0] - best_ac[ds_label]) < 1e-9)) if ac else '--'
            cells.append(ac_str)
            cells.append(fmt_asr(pfx, asr_map))
        lines.append(' & '.join(cells) + r' \\')

    lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX ablation tables from noise_ablation summary log'
    )
    parser.add_argument('--out', default=None,
                        help='Directory to write .tex files (default: stdout)')
    parser.add_argument('--log', default=SUMMARY_LOG,
                        help=f'Path to summary log (default: {SUMMARY_LOG})')
    parser.add_argument('--k', type=int, default=DEFAULT_K, choices=[2, 4, 6, 10],
                        help=f'n_components value to report (default: {DEFAULT_K})')
    args = parser.parse_args()

    metrics, asr_map = parse_log(args.log)

    tables = [
        dict(
            label   = 'tab:noise',
            caption = (
                r'Effect of gradient noise $\sigma$ on AC detection ($k='
                + str(args.k)
                + r'$) and attack success rate at $p{=}15\%$ on MNIST and FashionMNIST.'
            ),
            cond_label  = r'$\sigma$',
            row_display = [rf'${n}$' for n in NOISE_LEVELS],
            row_prefixes = (
                [f'noise_mnist_{n}_r0.15'  for n in NOISE_LEVELS],
                [f'noise_fmnist_{n}_r0.15' for n in NOISE_LEVELS],
            ),
        ),
        dict(
            label   = 'tab:pretrain',
            caption = (
                r'Effect of reconstruction model pretraining on AC detection ($k='
                + str(args.k)
                + r'$) and attack success rate at $p{=}15\%$ on MNIST and FashionMNIST.'
            ),
            cond_label  = r'pretrain',
            row_display = [r'$0$ ep'] + [rf'${e}$ ep' for e in PRETRAIN_EPOCHS],
            row_prefixes = (
                [f'noise_mnist_0.00_r0.15']   + [f'pretrain_mnist_{e}ep_r0.15'  for e in PRETRAIN_EPOCHS],
                [f'noise_fmnist_0.00_r0.15']  + [f'pretrain_fmnist_{e}ep_r0.15' for e in PRETRAIN_EPOCHS],
            ),
        ),
    ]

    for t in tables:
        table = make_table(
            label        = t['label'],
            caption      = t['caption'],
            cond_label   = t['cond_label'],
            row_display  = t['row_display'],
            row_prefixes = t['row_prefixes'],
            metrics      = metrics,
            asr_map      = asr_map,
            k            = args.k,
        )
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            out_path = os.path.join(args.out, t['label'].replace('tab:', '') + '.tex')
            with open(out_path, 'w') as f:
                f.write(table + '\n')
            print(f'Saved → {out_path}')
        else:
            print(f"\n% ── {t['label']} {'─' * 50}")
            print(table)


if __name__ == '__main__':
    main()

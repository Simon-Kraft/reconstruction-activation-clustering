#!/usr/bin/env python3
"""
scripts/generate_silhouette_ablation_tables.py — Silhouette ablation tables.

Reads results/{exp_id}/n_components_2/silhouette_results.json for every
noise / pretrain ablation experiment and produces two LaTeX tables:
  - Noise ablation    (rows = σ values,       cols = MNIST / FashionMNIST)
  - Pretrain ablation (rows = pretrain epochs, cols = MNIST / FashionMNIST)

Matches the layout of generate_ablation_tables.py.

Usage:
    python scripts/generate_silhouette_ablation_tables.py
    python scripts/generate_silhouette_ablation_tables.py --out results/tables/
    python scripts/generate_silhouette_ablation_tables.py --k 4
"""

import argparse
import json
import os
import sys
import numpy as np

RESULTS_DIR     = 'results'
SEEDS           = [41, 42, 43]
NOISE_LEVELS    = ['0.0', '0.01', '0.05', '0.1', '0.2']
PRETRAIN_EPOCHS = [0, 1, 5, 10]
DEFAULT_K       = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def exp_id(dataset, noise, pretrain, seed):
    return (
        f"{dataset}_rotating"
        f"_r0.15"
        f"_sub0.25"
        f"_recongeiping"
        f"_replace0"
        f"_noise{noise}"
        f"_pre{pretrain}"
        f"_seed{seed}"
    )


def load_silhouette(dataset, noise, pretrain, seed, k):
    path = os.path.join(
        RESULTS_DIR,
        exp_id(dataset, noise, pretrain, seed),
        f'n_components_{k}',
        'silhouette_results.json',
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)['mean_silhouette']


def seed_stats(dataset, noise, pretrain, k):
    vals = [
        v for s in SEEDS
        if (v := load_silhouette(dataset, noise, pretrain, s, k)) is not None
    ]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(mean, std, bold=False):
    s = rf'{mean:.3f}{{\tiny$\pm${std:.3f}}}'
    return r'\textbf{' + s + '}' if bold else s


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def make_table(label, caption, cond_label, row_display, row_params, k):
    """
    row_params: list of (noise_str, pretrain_int) tuples, one per row.
    Columns: condition | MNIST silhouette | FashionMNIST silhouette
    """
    datasets = ['MNIST', 'FashionMNIST']

    # Best silhouette per dataset column
    best = {}
    for ds in datasets:
        b = -1.0
        for noise, pretrain in row_params:
            v = seed_stats(ds, noise, pretrain, k)
            if v and v[0] > b:
                b = v[0]
        best[ds] = b

    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{|c|c|c|}',
        r'\hline',
        r'& \textbf{MNIST} & \textbf{FashionMNIST} \\',
        cond_label + r' & \textbf{Silhouette} & \textbf{Silhouette} \\',
        r'\hline',
    ]

    for display, (noise, pretrain) in zip(row_display, row_params):
        cells = [display]
        for ds in datasets:
            v = seed_stats(ds, noise, pretrain, k)
            cells.append(fmt(*v, bold=abs(v[0] - best[ds]) < 1e-9) if v else '--')
        lines.append(' & '.join(cells) + r' \\')

    lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX silhouette ablation tables'
    )
    parser.add_argument('--out', default=None,
                        help='Directory to write .tex files (default: stdout)')
    parser.add_argument('--k', type=int, default=DEFAULT_K, choices=[2, 4, 6, 10],
                        help=f'n_components value (default: {DEFAULT_K})')
    args = parser.parse_args()

    k = args.k

    tables = [
        dict(
            label      = 'tab:silhouette_noise',
            caption    = (
                r'Mean AC silhouette score under gradient noise ($k=' + str(k) + r'$, '
                r'$p{=}15\%$, pretrain$=0$). Higher = better cluster separation.'
            ),
            cond_label = r'$\sigma$',
            row_display = [rf'${n}$' for n in NOISE_LEVELS],
            row_params  = [(n, 0) for n in NOISE_LEVELS],
        ),
        dict(
            label      = 'tab:silhouette_pretrain',
            caption    = (
                r'Mean AC silhouette score vs.\ pretraining epochs ($k=' + str(k) + r'$, '
                r'$p{=}15\%$, $\sigma{=}0$). Higher = better cluster separation.'
            ),
            cond_label = r'pretrain',
            row_display = [r'$0$ ep'] + [rf'${e}$ ep' for e in PRETRAIN_EPOCHS[1:]],
            row_params  = [('0.0', e) for e in PRETRAIN_EPOCHS],
        ),
    ]

    for t in tables:
        table = make_table(
            label       = t['label'],
            caption     = t['caption'],
            cond_label  = t['cond_label'],
            row_display = t['row_display'],
            row_params  = t['row_params'],
            k           = k,
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

#!/usr/bin/env python3
"""
scripts/generate_silhouette_tables.py — LaTeX silhouette-score tables.

Reads results/{exp_id}/n_components_2/silhouette_results.json for every
MNIST / FashionMNIST × geiping / badnets × seed combination, averages the
mean silhouette score across seeds, and produces one LaTeX table (k=2).

Usage:
    python scripts/generate_silhouette_tables.py
    python scripts/generate_silhouette_tables.py --out results/tables/
    python scripts/generate_silhouette_tables.py --k 4
"""

import argparse
import json
import os
import sys
import numpy as np

RESULTS_DIR = 'results'
DATASETS    = ['MNIST', 'FashionMNIST']
METHODS     = [('geiping', 'Ours'), ('badnets', 'Baseline')]
RATES       = ['0.1', '0.15', '0.33']
RATE_TEX    = [r'$p{=}10\%$', r'$p{=}15\%$', r'$p{=}33\%$']
SEEDS       = [41, 42, 43]
DEFAULT_K   = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def exp_id(dataset, method, rate, seed):
    return (
        f"{dataset}_rotating"
        f"_r{rate}"
        f"_sub0.25"
        f"_recon{method}"
        f"_replace0"
        f"_noise0.0"
        f"_pre0"
        f"_seed{seed}"
    )


def load_silhouette(dataset, method, rate, seed, k):
    path = os.path.join(
        RESULTS_DIR,
        exp_id(dataset, method, rate, seed),
        f'n_components_{k}',
        'silhouette_results.json',
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)['mean_silhouette']


def seed_stats(dataset, method, rate, k):
    vals = [
        v for s in SEEDS
        if (v := load_silhouette(dataset, method, rate, s, k)) is not None
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


def cell(dataset, method, rate, k, best):
    v = seed_stats(dataset, method, rate, k)
    if v is None:
        return '--'
    return fmt(*v, bold=abs(v[0] - best[(dataset, rate)]) < 1e-9)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def make_table(k):
    # Determine best (highest) mean silhouette per (dataset, rate) across methods
    best = {}
    for ds in DATASETS:
        for rate in RATES:
            b = -1.0
            for me, _ in METHODS:
                v = seed_stats(ds, me, rate, k)
                if v and v[0] > b:
                    b = v[0]
            best[(ds, rate)] = b

    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        (r'\caption{Mean AC silhouette score ($k=' + str(k) + r'$, averaged across seeds). '
         r'Higher = better cluster separation. '
         r'Ours = Geiping et al.~\cite{geiping2020inverting}. '
         r'Baseline = BadNets (clean originals).}'),
        r'\label{tab:silhouette_k' + str(k) + r'}',
        r'\begin{tabular}{|c|c|ccc|ccc|}',
        r'\hline',
        (r'\multirow{2}{*}{\textbf{Metric}} & \multirow{2}{*}{\textbf{Method}} & '
         r'\multicolumn{3}{c|}{\textbf{MNIST}} & \multicolumn{3}{c|}{\textbf{FashionMNIST}} \\'),
        r'& & ' + ' & '.join(RATE_TEX) + r' & ' + ' & '.join(RATE_TEX) + r' \\',
        r'\hline',
    ]

    for me_idx, (me, me_label) in enumerate(METHODS):
        cells = ' & '.join(
            cell(ds, me, rate, k, best)
            for ds in DATASETS
            for rate in RATES
        )
        if me_idx == 0:
            lines.append(
                r'\multirow{2}{*}{\textit{Silhouette}}'
                f' & {me_label} & {cells} \\\\'
            )
        else:
            lines.append(f'    & {me_label} & {cells} \\\\')
    lines.append(r'\hline')
    lines += [r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX silhouette tables for MNIST/FashionMNIST experiments'
    )
    parser.add_argument('--out', default=None,
                        help='Directory to write .tex files (default: stdout)')
    parser.add_argument('--k', type=int, default=DEFAULT_K, choices=[2, 4, 6, 10],
                        help=f'n_components value (default: {DEFAULT_K})')
    args = parser.parse_args()

    table = make_table(args.k)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, f'silhouette_k{args.k}.tex')
        with open(out_path, 'w') as f:
            f.write(table + '\n')
        print(f'Saved → {out_path}')
    else:
        print(table)


if __name__ == '__main__':
    main()

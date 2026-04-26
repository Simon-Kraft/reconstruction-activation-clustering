#!/usr/bin/env python3
"""
scripts/generate_latex_tables.py — Generate LaTeX result tables from summary logs.

Reads logs/mnist/summary.log and logs/fashionmnist/summary.log, averages
metrics across seeds, and produces one LaTeX table per n_components value.
Bold marks the better of Ours vs Baseline for each cell.

Usage:
    python scripts/generate_latex_tables.py              # print all tables
    python scripts/generate_latex_tables.py --out results/tables/
    python scripts/generate_latex_tables.py --k 10      # single k value
"""

import argparse
import os
import re
import sys
import numpy as np

SUMMARY_LOGS = {
    'MNIST':        'logs/mnist/summary.log',
    'FashionMNIST': 'logs/fashionmnist/summary.log',
}

RATES   = ['0.10', '0.15', '0.33']
N_COMPS = [2, 4, 6, 10]

METHODS = [('geiping', 'Ours'), ('badnets', 'Baseline')]

# (key in data dict, LaTeX label, separator after this metric row pair)
METRICS = [
    ('ac_acc',  r'\textit{AC Acc.}',  r'\cline{1-8}'),
    ('ac_f1',   r'\textit{AC F1}',    r'\hline'),
    ('raw_acc', r'\textit{Raw Acc.}', r'\cline{1-8}'),
    ('raw_f1',  r'\textit{Raw F1}',   r'\hline'),
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_log(path):
    """
    Returns dict[(method, rate, seed, k)] -> {ac_acc, ac_f1, raw_acc, raw_f1}.
    """
    result = {}
    if not os.path.exists(path):
        print(f'Warning: {path} not found', file=sys.stderr)
        return result

    with open(path) as f:
        text = f.read()

    for block_label, block_body in re.findall(
        r'── ([^\n]+) ──\n(.*?)(?=── |\Z)', text, re.DOTALL
    ):
        m = re.match(
            r'(?:mn|fm)_(geiping|badnets)_r([0-9.]+)_seed([0-9]+)',
            block_label.strip()
        )
        if not m:
            continue
        method, rate, seed = m.group(1), m.group(2), int(m.group(3))

        for row in re.finditer(
            r'k=(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%',
            block_body
        ):
            k = int(row.group(1))
            result[(method, rate, seed, k)] = {
                'ac_acc':  float(row.group(2)),
                'ac_f1':   float(row.group(3)),
                'raw_acc': float(row.group(4)),
                'raw_f1':  float(row.group(5)),
            }
    return result


def stats_across_seeds(data, method, rate, k):
    entries = [v for (me, r, s, kk), v in data.items()
               if me == method and r == rate and kk == k]
    if not entries:
        return None
    return {metric: (float(np.mean([e[metric] for e in entries])),
                     float(np.std([e[metric] for e in entries])))
            for metric in entries[0]}


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def bf(val_str, do_bold):
    return r'\textbf{' + val_str + '}' if do_bold else val_str


def make_table(all_data, k):
    datasets = ['MNIST', 'FashionMNIST']
    rate_tex = [r'$p{=}10\%$', r'$p{=}15\%$', r'$p{=}33\%$']

    # Pre-aggregate: vals[dataset][method][rate] -> {metric: (mean, std)} or None
    vals = {
        ds: {
            me: {r: stats_across_seeds(all_data[ds], me, r, k) for r in RATES}
            for me, _ in METHODS
        }
        for ds in datasets
    }

    def cell(ds, me, rate, metric_key):
        v_this  = vals[ds][me][rate]
        me_other = 'badnets' if me == 'geiping' else 'geiping'
        v_other = vals[ds][me_other][rate]
        if v_this is None:
            return '--'
        mean, std = v_this[metric_key]
        other_mean = v_other[metric_key][0] if v_other is not None else -1
        val_str = rf'{mean:.2f}{{\tiny$\pm${std:.2f}}}'
        return bf(val_str, mean >= other_mean)

    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        (r'\caption{AC and Raw Clustering detection performance ($k=' + str(k) + r'$). '
         r'Ours = Reconstructed + Poisoned (Geiping et al.~\cite{geiping2020inverting}). '
         r'Baseline = Clean originals (Chen et al.~\cite{chen2018activationclustering}).}'),
        r'\label{tab:results_k' + str(k) + r'}',
        r'\begin{tabular}{|c|c|ccc|ccc|}',
        r'\hline',
        (r'\multirow{2}{*}{\textbf{Metric}} & \multirow{2}{*}{\textbf{Method}} & '
         r'\multicolumn{3}{c|}{\textbf{MNIST}} & \multicolumn{3}{c|}{\textbf{FashionMNIST}} \\'),
        r'& & ' + ' & '.join(rate_tex) + r' & ' + ' & '.join(rate_tex) + r' \\',
        r'\hline',
    ]

    for metric_key, metric_label, separator in METRICS:
        for me_idx, (me, me_label) in enumerate(METHODS):
            cells = ' & '.join(
                cell(ds, me, rate, metric_key)
                for ds in datasets
                for rate in RATES
            )
            if me_idx == 0:
                lines.append(
                    r'\multirow{2}{*}{' + metric_label + r'}'
                    f' & {me_label} & {cells} \\\\'
                )
            else:
                lines.append(f'    & {me_label} & {cells} \\\\')
        lines.append(separator)

    lines += [r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX tables from MNIST/FashionMNIST summary logs'
    )
    parser.add_argument('--out', default=None,
                        help='Directory to write .tex files (default: stdout)')
    parser.add_argument('--k', type=int, default=None, choices=N_COMPS,
                        help='Generate table for a single k value only')
    args = parser.parse_args()

    all_data = {ds: parse_log(path) for ds, path in SUMMARY_LOGS.items()}

    ks = [args.k] if args.k else N_COMPS
    for k in ks:
        table = make_table(all_data, k)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            out_path = os.path.join(args.out, f'results_k{k}.tex')
            with open(out_path, 'w') as f:
                f.write(table + '\n')
            print(f'Saved → {out_path}')
        else:
            print(f'\n% ── k={k} {"─" * 50}')
            print(table)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
scripts/select_n_comp.py — Report which n_components (ICA dimensionality)
gives the highest silhouette score.

Motivation: reviewers flagged that n_components=2 was picked by looking at
ground-truth detection F1, which is a form of tuning bias (AC is meant to
be unsupervised — a real defender has no poison labels to tune against).
Silhouette score needs no ground-truth labels at all, so selecting
n_components by silhouette instead is methodologically sound for an
unsupervised method, and the paper can report the resulting F1 as a
downstream *consequence* of the selection rather than the criterion itself.

Auto-discovers every (method, rate, seed, n_comp) combination that has a
silhouette_results.json under outputs/<dataset>/ — no hardcoded list of
rates/seeds/n_comp values, so it stays correct no matter what you've swept.

Usage:
    python scripts/select_n_comp.py --dataset MNIST
    python scripts/select_n_comp.py --dataset MNIST --out outputs/MNIST/n_comp_selection.json
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def discover(dataset):
    """
    Scans outputs/<dataset>/*/results/n_components_*/silhouette_results.json.

    Returns:
        per_n_comp:        {n_comp: [mean_silhouette, ...]}  (all methods/rates/seeds pooled)
        per_method_n_comp:  {method: {n_comp: [mean_silhouette, ...]}}
    """
    pattern = os.path.join('outputs', dataset, '*', 'results', 'n_components_*', 'silhouette_results.json')
    per_n_comp = defaultdict(list)
    per_method_n_comp = defaultdict(lambda: defaultdict(list))

    for path in glob.glob(pattern):
        m = re.search(r'n_components_(\d+)', path)
        if not m:
            continue
        n_comp = int(m.group(1))

        # outputs/<dataset>/<exp_id>/results/n_components_<n>/silhouette_results.json
        exp_id = path.split(os.sep)[2]
        method_match = re.match(r'([a-zA-Z]+)_rotating', exp_id)
        method = method_match.group(1) if method_match else 'unknown'

        with open(path) as f:
            sil = json.load(f)['mean_silhouette']

        per_n_comp[n_comp].append(sil)
        per_method_n_comp[method][n_comp].append(sil)

    return per_n_comp, per_method_n_comp


def main():
    p = argparse.ArgumentParser(
        description='Report the n_components value with the highest silhouette score'
    )
    p.add_argument('--dataset', type=str, default='MNIST')
    p.add_argument('--out', type=str, default=None,
                    help='Path to save the JSON report '
                         '(default: outputs/<dataset>/n_comp_selection.json)')
    args = p.parse_args()

    per_n_comp, per_method_n_comp = discover(args.dataset)
    if not per_n_comp:
        print(f"No silhouette_results.json files found under outputs/{args.dataset}/ "
              f"— has the experiment suite for this dataset been run yet?")
        return

    stats = {
        n: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n_runs': len(v)}
        for n, v in per_n_comp.items()
    }
    best_n_comp = max(stats, key=lambda n: stats[n]['mean'])

    print(f"Silhouette score by n_components — {args.dataset} "
          f"(all reconstruction methods/rates/seeds pooled)")
    print(f"  {'n_comp':<8} {'mean_silhouette':<20} {'n_runs':<8}")
    for n in sorted(stats):
        marker = '  <-- highest' if n == best_n_comp else ''
        print(f"  {n:<8} {stats[n]['mean']:.4f} ± {stats[n]['std']:.4f}      "
              f"{stats[n]['n_runs']:<8}{marker}")

    print(f"\nBest n_components by silhouette score (unsupervised criterion): {best_n_comp}")

    best_n_comp_per_method = {}
    for method, n_map in per_method_n_comp.items():
        means = {n: float(np.mean(v)) for n, v in n_map.items()}
        best_n_comp_per_method[method] = max(means, key=means.get)
        print(f"  {method:<10} best n_comp = {best_n_comp_per_method[method]}")

    report = {
        'dataset': args.dataset,
        'criterion': 'max mean silhouette score across all runs found '
                     '(unsupervised — no ground-truth poison labels used)',
        'best_n_components': best_n_comp,
        'best_n_components_per_method': best_n_comp_per_method,
        'per_n_components': stats,
    }

    out_path = args.out or os.path.join('outputs', args.dataset, 'n_comp_selection.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == '__main__':
    main()

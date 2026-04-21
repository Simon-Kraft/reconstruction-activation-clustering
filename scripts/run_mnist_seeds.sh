#!/bin/bash
# scripts/run_mnist_p10_seeds.sh
#
# Runs MNIST p=10% across 5 seeds for both Geiping and BadNets baseline.
# Uses n_components=2 ICA — change AC_N_COMPONENTS in config.py before running.
#
# Usage:
#   bash scripts/run_mnist_p10_seeds.sh
#
# Results land in results/ and are aggregated at the end.

set -e

DATASET="MNIST"
RATE="0.10"
SUBSAMPLE="0.25"
SEEDS=(41 42 43 44 45)

echo "========================================================"
echo "  MNIST p=10% — 5 seeds — Geiping vs BadNets"
echo "  subsample=${SUBSAMPLE}  ICA n_components=2"
echo "========================================================"

# ── Geiping (use_reconstruction=1) ───────────────────────────────────
echo ""
echo "── Geiping reconstruction ──────────────────────────────"
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "  → seed=${SEED}"
    python pipeline.py \
        --dataset        "$DATASET" \
        --poison_rate    "$RATE" \
        --subsample_rate "$SUBSAMPLE" \
        --use_reconstruction 1 \
        --seed           "$SEED" \
        --no_plots
done

# ── BadNets baseline (use_reconstruction=0) ──────────────────────────
echo ""
echo "── BadNets baseline ────────────────────────────────────"
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "  → seed=${SEED}"
    python pipeline.py \
        --dataset        "$DATASET" \
        --poison_rate    "$RATE" \
        --subsample_rate "$SUBSAMPLE" \
        --use_reconstruction 0 \
        --seed           "$SEED" \
        --no_plots
done

# ── Aggregate results ─────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Aggregating results across seeds"
echo "========================================================"

python - <<'PYEOF'
import json
import numpy as np
import os

DATASET    = "MNIST"
RATE       = "0.1"
SUBSAMPLE  = "0.25"
SEEDS      = [41, 42, 43, 44, 45]

def load_f1(recon, seed):
    exp_id = (
        f"{DATASET}_rotating"
        f"_r{RATE}"
        f"_sub{SUBSAMPLE}"
        f"_recon{recon}"
        f"_replace0"
        f"_noise0.0"
        f"_pre0"
        f"_seed{seed}"
    )
    ac_path  = f"results/{exp_id}/ac_detection_results.json"
    raw_path = f"results/{exp_id}/raw_detection_results.json"

    if not os.path.exists(ac_path):
        print(f"  Missing: {ac_path}")
        return None, None

    with open(ac_path)  as f: ac_data  = json.load(f)
    with open(raw_path) as f: raw_data = json.load(f)

    return ac_data['overall_f1'] * 100, raw_data['overall_f1'] * 100

for recon, label in [(1, "Geiping (Ours)"), (0, "BadNets (Baseline)")]:
    ac_f1s, raw_f1s = [], []
    for seed in SEEDS:
        ac_f1, raw_f1 = load_f1(recon, seed)
        if ac_f1 is not None:
            ac_f1s.append(ac_f1)
            raw_f1s.append(raw_f1)

    if ac_f1s:
        print(f"\n{label}:")
        print(f"  Seeds:   {SEEDS[:len(ac_f1s)]}")
        print(f"  AC  F1:  {np.mean(ac_f1s):.2f} ± {np.std(ac_f1s):.2f}%")
        print(f"  Raw F1:  {np.mean(raw_f1s):.2f} ± {np.std(raw_f1s):.2f}%")
        print(f"  AC  F1 per seed:  {[f'{v:.1f}' for v in ac_f1s]}")
        print(f"  Raw F1 per seed:  {[f'{v:.1f}' for v in raw_f1s]}")
    else:
        print(f"\n{label}: no results found")

PYEOF

echo ""
echo "Done."
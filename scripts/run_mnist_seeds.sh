#!/bin/bash
# scripts/run_mnist_p10_seeds.sh
#
# Runs MNIST p=10% across 5 seeds for both Geiping and BadNets baseline.
# Matches the logging and structure of run_fashionmnist_experiments.sh.
#
# Usage:
#   chmod +x scripts/run_mnist_p10_seeds.sh
#   ./scripts/run_mnist_p10_seeds.sh

set -e

LOGS_DIR="logs/mnist_p10_seeds"
mkdir -p "$LOGS_DIR"

SUMMARY_LOG="$LOGS_DIR/summary.log"
TOTAL_START=$(date +%s)

DATASET="MNIST"
RATE="0.10"
SUBSAMPLE="0.25"
SEEDS=(41 42 43 44 45)

echo "============================================================" | tee "$SUMMARY_LOG"
echo "  MNIST p=10% — 5 seeds — Geiping vs BadNets" | tee -a "$SUMMARY_LOG"
echo "  subsample=${SUBSAMPLE}" | tee -a "$SUMMARY_LOG"
echo "  Started: $(date)" | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"

run_experiment() {
    local label="$1"
    local logfile="$LOGS_DIR/${label}.log"
    shift
    local cmd="python pipeline.py --dataset $DATASET $@"

    echo "" | tee -a "$SUMMARY_LOG"
    echo "── $label ──" | tee -a "$SUMMARY_LOG"
    echo "   cmd: $cmd" | tee -a "$SUMMARY_LOG"

    START=$(date +%s)
    echo "============================================================" > "$logfile"
    echo "  Experiment: $label" >> "$logfile"
    echo "  Command:    $cmd" >> "$logfile"
    echo "  Started:    $(date)" >> "$logfile"
    echo "============================================================" >> "$logfile"

    eval "$cmd" 2>&1 | tee -a "$logfile"

    END=$(date +%s)
    ELAPSED=$(( END - START ))
    MINUTES=$(( ELAPSED / 60 ))
    SECONDS=$(( ELAPSED % 60 ))

    AC_F1=$(grep  "AC overall F1"  "$logfile" | tail -1 | awk '{print $NF}')
    RAW_F1=$(grep "Raw overall F1" "$logfile" | tail -1 | awk '{print $NF}')
    CA=$(grep     "Clean accuracy" "$logfile" | tail -1 | awk '{print $NF}')
    ASR=$(grep    "ASR (avg)"      "$logfile" | tail -1 | awk '{print $NF}')

    echo "   time:      ${MINUTES}m ${SECONDS}s" | tee -a "$SUMMARY_LOG"
    echo "   clean_acc: $CA"  | tee -a "$SUMMARY_LOG"
    echo "   asr (avg): $ASR" | tee -a "$SUMMARY_LOG"
    echo "   AC F1:     $AC_F1"  | tee -a "$SUMMARY_LOG"
    echo "   Raw F1:    $RAW_F1" | tee -a "$SUMMARY_LOG"
}

# ===========================================================================
# GROUP 1 — Geiping reconstruction
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 1: Geiping reconstruction (use_reconstruction=1)"   | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for SEED in "${SEEDS[@]}"; do
    run_experiment "mnist_p10_geiping_seed${SEED}" \
        --poison_rate    "$RATE" \
        --subsample_rate "$SUBSAMPLE" \
        --use_reconstruction 1 \
        --seed           "$SEED" \
        --no_plots
done

# ===========================================================================
# GROUP 2 — BadNets baseline
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 2: BadNets baseline (use_reconstruction=0)"         | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for SEED in "${SEEDS[@]}"; do
    run_experiment "mnist_p10_badnets_seed${SEED}" \
        --poison_rate    "$RATE" \
        --subsample_rate "$SUBSAMPLE" \
        --use_reconstruction 0 \
        --seed           "$SEED" \
        --no_plots
done

# ===========================================================================
# Aggregate results across seeds
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  Aggregating results across seeds"                          | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

python - <<PYEOF | tee -a "$SUMMARY_LOG"
import json
import numpy as np
import os

DATASET   = "MNIST"
RATE      = "0.1"
SUBSAMPLE = "0.25"
SEEDS     = [41, 42, 43, 44, 45]

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
        print(f"\n{label}  (n={len(ac_f1s)} seeds):")
        print(f"  AC  F1:  {np.mean(ac_f1s):.2f} +/- {np.std(ac_f1s):.2f}%")
        print(f"  Raw F1:  {np.mean(raw_f1s):.2f} +/- {np.std(raw_f1s):.2f}%")
        print(f"  AC  F1 per seed: {[f'{v:.1f}' for v in ac_f1s]}")
        print(f"  Raw F1 per seed: {[f'{v:.1f}' for v in raw_f1s]}")
    else:
        print(f"\n{label}: no results found")
PYEOF

# ===========================================================================
# Done
# ===========================================================================
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_MINUTES=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECONDS=$(( TOTAL_ELAPSED % 60 ))

echo "" | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"
echo "  All experiments complete" | tee -a "$SUMMARY_LOG"
echo "  Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s" | tee -a "$SUMMARY_LOG"
echo "  Finished:   $(date)" | tee -a "$SUMMARY_LOG"
echo "  Summary:    $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"
echo "  Logs:       $LOGS_DIR/" | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"
#!/bin/bash
# scripts/run_mnist.sh — MNIST experiment suite.
#
# Runs Geiping and BadNets across 3 seeds and 3 poison rates.
# Evaluates AC clustering at multiple n_components values per run.
#
# Usage:
#   chmod +x scripts/run_mnist.sh
#   ./scripts/run_mnist.sh

set -e

LOGS_DIR="logs/mnist"
mkdir -p "$LOGS_DIR"

SUMMARY_LOG="$LOGS_DIR/summary.log"
TOTAL_START=$(date +%s)

SEEDS=(41 42 43)
RATES=(0.10 0.15 0.33)
AC_N_COMPONENTS="2,4,6,10"
SUBSAMPLE="0.25"

echo "============================================================" | tee "$SUMMARY_LOG"
echo "  MNIST Experiment Suite" | tee -a "$SUMMARY_LOG"
echo "  seeds=${SEEDS[*]}  rates=${RATES[*]}" | tee -a "$SUMMARY_LOG"
echo "  ac_n_components=${AC_N_COMPONENTS}" | tee -a "$SUMMARY_LOG"
echo "  Started: $(date)" | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"

run_experiment() {
    local label="$1"
    local logfile="$LOGS_DIR/${label}.log"
    shift
    local cmd="python pipeline.py --dataset MNIST $@"

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

    CA=$(grep  "Clean accuracy" "$logfile" | tail -1 | awk '{print $NF}')
    ASR=$(grep "ASR (avg)"      "$logfile" | tail -1 | awk '{print $NF}')

    echo "   time:      ${MINUTES}m ${SECONDS}s" | tee -a "$SUMMARY_LOG"
    echo "   clean_acc: $CA"  | tee -a "$SUMMARY_LOG"
    echo "   asr (avg): $ASR" | tee -a "$SUMMARY_LOG"
    awk '/Pipeline complete/{f=1;next} /Results saved to:/{f=0} f && /^  k|^  --/' \
        "$logfile" | while IFS= read -r line; do
        echo "$line" | tee -a "$SUMMARY_LOG"
    done
}

# ===========================================================================
# GROUP 1 — Geiping reconstruction
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 1: Geiping reconstruction"                           | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for RATE in "${RATES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run_experiment "mn_geiping_r${RATE}_seed${SEED}" \
            --poison_rate           "$RATE" \
            --subsample_rate        "$SUBSAMPLE" \
            --reconstruction_method geiping \
            --ac_n_components       "$AC_N_COMPONENTS" \
            --seed                  "$SEED" \
            --no_plots
    done
done

# ===========================================================================
# GROUP 2 — BadNets baseline
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 2: BadNets baseline"                                  | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for RATE in "${RATES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run_experiment "mn_badnets_r${RATE}_seed${SEED}" \
            --poison_rate           "$RATE" \
            --subsample_rate        "$SUBSAMPLE" \
            --reconstruction_method badnets \
            --ac_n_components       "$AC_N_COMPONENTS" \
            --seed                  "$SEED" \
            --no_plots
    done
done

# ===========================================================================
# Aggregate results
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  Aggregating results across seeds"                           | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

python - <<PYEOF | tee -a "$SUMMARY_LOG"
import json
import numpy as np
import os

DATASET    = "MNIST"
SUBSAMPLE  = "0.25"
SEEDS      = [41, 42, 43]
RATES      = ["0.1", "0.15", "0.33"]
N_COMPS    = [2, 4, 6, 10]

def load_f1(method, rate, seed, k):
    exp_id = (
        f"{DATASET}_rotating"
        f"_r{rate}"
        f"_sub{SUBSAMPLE}"
        f"_recon{method}"
        f"_replace0"
        f"_noise0.0"
        f"_pre0"
        f"_seed{seed}"
    )
    ac_path  = f"results/{exp_id}/n_components_{k}/ac_detection_results.json"
    raw_path = f"results/{exp_id}/n_components_{k}/raw_detection_results.json"
    if not os.path.exists(ac_path):
        return None, None
    with open(ac_path)  as f: ac_data  = json.load(f)
    with open(raw_path) as f: raw_data = json.load(f)
    return ac_data['overall_f1'] * 100, raw_data['overall_f1'] * 100

for method, label in [('geiping', 'Geiping'), ('badnets', 'BadNets')]:
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"  {'rate':<8} " + " ".join(f"{'k='+str(k):>14}" for k in N_COMPS))
    print(f"  {'-'*60}")
    for rate in RATES:
        row_parts = []
        for k in N_COMPS:
            ac_vals = []
            for seed in SEEDS:
                ac_f1, _ = load_f1(method, rate, seed, k)
                if ac_f1 is not None:
                    ac_vals.append(ac_f1)
            if ac_vals:
                row_parts.append(f"{np.mean(ac_vals):5.1f}±{np.std(ac_vals):4.1f}")
            else:
                row_parts.append("     n/a")
        print(f"  r={rate:<6} " + " ".join(f"{p:>14}" for p in row_parts))
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
echo "  All MNIST experiments complete" | tee -a "$SUMMARY_LOG"
echo "  Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s" | tee -a "$SUMMARY_LOG"
echo "  Finished:   $(date)" | tee -a "$SUMMARY_LOG"
echo "  Summary:    $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"
echo "  Logs:       $LOGS_DIR/" | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"

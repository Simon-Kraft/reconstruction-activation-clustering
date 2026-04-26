#!/bin/bash
# scripts/run_reconstruction_comparison.sh
#
# Compares Geiping (cosine inversion) vs DLG (L2 inversion) reconstruction
# across 5 seeds on MNIST at p=15%. Reports AC F1, Raw F1, PSNR, ASR, and
# clean accuracy for each method.
#
# Usage:
#   chmod +x scripts/run_reconstruction_comparison.sh
#   ./scripts/run_reconstruction_comparison.sh

set -e

LOGS_DIR="logs/recon_comparison"
mkdir -p "$LOGS_DIR"

SUMMARY_LOG="$LOGS_DIR/summary.log"
TOTAL_START=$(date +%s)

DATASET="MNIST"
RATE="0.15"
SUBSAMPLE="0.25"
SEEDS=(42)

echo "============================================================" | tee "$SUMMARY_LOG"
echo "  Reconstruction Comparison: Geiping vs DLG" | tee -a "$SUMMARY_LOG"
echo "  dataset=${DATASET}  rate=${RATE}  subsample=${SUBSAMPLE}" | tee -a "$SUMMARY_LOG"
echo "  seeds=${SEEDS[*]}" | tee -a "$SUMMARY_LOG"
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

    CA=$(grep     "Clean accuracy"      "$logfile" | tail -1 | awk '{print $NF}')
    ASR=$(grep    "ASR (avg)"           "$logfile" | tail -1 | awk '{print $NF}')
    PSNR=$(grep   "Reconstruction PSNR" "$logfile" | tail -1 | grep -o 'mean=[0-9.]*' | cut -d= -f2)

    echo "   time:      ${MINUTES}m ${SECONDS}s" | tee -a "$SUMMARY_LOG"
    echo "   clean_acc: $CA"  | tee -a "$SUMMARY_LOG"
    echo "   asr (avg): $ASR" | tee -a "$SUMMARY_LOG"
    echo "   PSNR (dB): $PSNR" | tee -a "$SUMMARY_LOG"
    awk '/Pipeline complete/{f=1;next} /Results saved to:/{f=0} f && /^  k|^  --/' \
        "$logfile" | while IFS= read -r line; do
        echo "$line" | tee -a "$SUMMARY_LOG"
    done
}

# ===========================================================================
# GROUP 1 — Geiping (cosine similarity inversion)
# ===========================================================================
# echo "" | tee -a "$SUMMARY_LOG"
# echo "###########################################################" | tee -a "$SUMMARY_LOG"
# echo "  GROUP 1: Geiping reconstruction"                           | tee -a "$SUMMARY_LOG"
# echo "###########################################################" | tee -a "$SUMMARY_LOG"

# for SEED in "${SEEDS[@]}"; do
#     run_experiment "recon_geiping_seed${SEED}" \
#         --poison_rate           "$RATE" \
#         --subsample_rate        "$SUBSAMPLE" \
#         --reconstruction_method geiping \
#         --seed                  "$SEED" \
#         --no_plots
# done

# ===========================================================================
# GROUP 2 — DLG (L2 gradient inversion)
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 2: DLG reconstruction"                               | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for SEED in "${SEEDS[@]}"; do
    run_experiment "recon_dlg_seed${SEED}" \
        --poison_rate           "$RATE" \
        --subsample_rate        "$SUBSAMPLE" \
        --reconstruction_method dlg \
        --seed                  "$SEED" \
        --no_plots
done

# ===========================================================================
# Aggregate and compare
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  Aggregating results"                                        | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

python - <<PYEOF | tee -a "$SUMMARY_LOG"
import json
import re
import numpy as np
import os

DATASET   = "MNIST"
RATE      = "0.15"
SUBSAMPLE = "0.25"
SEEDS     = [41, 42, 43, 44, 45]
LOGS_DIR  = "logs/recon_comparison"

def load_results(method, seed):
    exp_id = (
        f"{DATASET}_rotating"
        f"_r{RATE}"
        f"_sub{SUBSAMPLE}"
        f"_recon{method}"
        f"_replace0"
        f"_noise0.0"
        f"_pre0"
        f"_seed{seed}"
    )
    ac_path  = f"results/{exp_id}/ac_detection_results.json"
    raw_path = f"results/{exp_id}/raw_detection_results.json"

    if not os.path.exists(ac_path):
        print(f"  Missing: {ac_path}")
        return None

    with open(ac_path)  as f: ac_data  = json.load(f)
    with open(raw_path) as f: raw_data = json.load(f)

    # Extract mean PSNR from the run log
    logfile = f"{LOGS_DIR}/recon_{method}_seed{seed}.log"
    psnr = None
    if os.path.exists(logfile):
        with open(logfile) as f:
            for line in f:
                m = re.search(r'Reconstruction PSNR.*mean=([0-9.]+)', line)
                if m:
                    psnr = float(m.group(1))

    return {
        'ac_f1':  ac_data['overall_f1'] * 100,
        'raw_f1': raw_data['overall_f1'] * 100,
        'psnr':   psnr,
    }

print()
print(f"{'Metric':<20} {'Geiping':>20} {'DLG':>20}")
print("-" * 62)

for metric_key, metric_label, fmt in [
    ('ac_f1',  'AC F1 (%)',   '{:.2f} +/- {:.2f}'),
    ('raw_f1', 'Raw F1 (%)',  '{:.2f} +/- {:.2f}'),
    ('psnr',   'PSNR (dB)',   '{:.2f} +/- {:.2f}'),
]:
    row = {}
    for method in ['geiping', 'dlg']:
        vals = []
        for seed in SEEDS:
            r = load_results(method, seed)
            if r is not None and r[metric_key] is not None:
                vals.append(r[metric_key])
        row[method] = (np.mean(vals), np.std(vals)) if vals else (float('nan'), float('nan'))

    g_str = fmt.format(*row['geiping']) if not np.isnan(row['geiping'][0]) else 'n/a'
    d_str = fmt.format(*row['dlg'])     if not np.isnan(row['dlg'][0])     else 'n/a'
    print(f"{metric_label:<20} {g_str:>20} {d_str:>20}")

print()
print("Per-seed breakdown:")
for method, label in [('geiping', 'Geiping'), ('dlg', 'DLG')]:
    ac_f1s, psnrs = [], []
    for seed in SEEDS:
        r = load_results(method, seed)
        if r is not None:
            ac_f1s.append(r['ac_f1'])
            if r['psnr'] is not None:
                psnrs.append(r['psnr'])
    print(f"  {label}  AC F1:  {[f'{v:.1f}' for v in ac_f1s]}")
    if psnrs:
        print(f"  {label}  PSNR:   {[f'{v:.1f}' for v in psnrs]}")
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
echo "  Comparison complete" | tee -a "$SUMMARY_LOG"
echo "  Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s" | tee -a "$SUMMARY_LOG"
echo "  Finished:   $(date)" | tee -a "$SUMMARY_LOG"
echo "  Summary:    $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"
echo "  Logs:       $LOGS_DIR/" | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"

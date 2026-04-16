#!/bin/bash
# run_noise_ablation.sh — Noise and pretraining ablation at p=15%
#
# Tests how gradient noise and reconstruction model pretraining affect
# AC detection and attack success rate.
#
# Usage:
#   chmod +x run_noise_ablation.sh
#   ./run_noise_ablation.sh

set -e

LOGS_DIR="logs/noise_ablation"
mkdir -p "$LOGS_DIR"

SUMMARY_LOG="$LOGS_DIR/summary.log"
RESULTS_CSV="$LOGS_DIR/noise_ablation_results.csv"
TOTAL_START=$(date +%s)

echo "============================================================" | tee "$SUMMARY_LOG"
echo "  Noise & Pretraining Ablation at p=15%"                     | tee -a "$SUMMARY_LOG"
echo "  Started: $(date)"                                           | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"

# Pipe-delimited to avoid CSV parsing issues
echo "dataset|noise_std|pretrain_epochs|poison_rate|ac_f1|raw_f1|clean_acc|asr" > "$RESULTS_CSV"

# ---------------------------------------------------------------------------
# Helper
# Args: label dataset noise_std poison_rate pretrain_epochs(default=0)
# ---------------------------------------------------------------------------
run_experiment() {
    local label="$1"
    local dataset="$2"
    local noise_std="$3"
    local poison_rate="$4"
    local pretrain_epochs="${5:-0}"
    local logfile="$LOGS_DIR/${label}.log"

    local cmd="python pipeline.py --dataset $dataset --noise_std $noise_std --poison_rate $poison_rate --pretrain_epochs $pretrain_epochs --no_plots"

    echo "" | tee -a "$SUMMARY_LOG"
    echo "── $label ──" | tee -a "$SUMMARY_LOG"
    echo "   cmd: $cmd" | tee -a "$SUMMARY_LOG"

    START=$(date +%s)
    echo "============================================================" > "$logfile"
    echo "  Experiment: $label" >> "$logfile"
    echo "  Command:    $cmd"   >> "$logfile"
    echo "  Started:    $(date)" >> "$logfile"
    echo "============================================================" >> "$logfile"

    eval "$cmd" 2>&1 | tee -a "$logfile"

    END=$(date +%s)
    ELAPSED=$(( END - START ))
    MINUTES=$(( ELAPSED / 60 ))
    SECONDS=$(( ELAPSED % 60 ))

    AC_F1=$(grep  "AC overall F1"  "$logfile" | tail -1 | awk '{print $NF}' | tr -d '%')
    RAW_F1=$(grep "Raw overall F1" "$logfile" | tail -1 | awk '{print $NF}' | tr -d '%')
    CA=$(grep     "Clean accuracy" "$logfile" | tail -1 | awk '{print $NF}' | tr -d '%')
    ASR=$(grep    "ASR (0→1)"      "$logfile" | tail -1 | awk '{print $NF}' | tr -d '%')

    echo "   time:         ${MINUTES}m ${SECONDS}s" | tee -a "$SUMMARY_LOG"
    echo "   clean_acc:    $CA"                      | tee -a "$SUMMARY_LOG"
    echo "   asr (0→1):    $ASR"                     | tee -a "$SUMMARY_LOG"
    echo "   AC F1:        $AC_F1"                   | tee -a "$SUMMARY_LOG"
    echo "   Raw F1:       $RAW_F1"                  | tee -a "$SUMMARY_LOG"

    echo "$dataset|$noise_std|$pretrain_epochs|$poison_rate|$AC_F1|$RAW_F1|$CA|$ASR" >> "$RESULTS_CSV"
}

# ===========================================================================
# FashionMNIST — noise ablation at p=15% (pretrain_epochs=0)
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  FashionMNIST — Noise Ablation (p=15%, pretrain=0)"        | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for noise in 0.00 0.01 0.05 0.10 0.20; do
    run_experiment "noise_fmnist_${noise}_r0.15" FashionMNIST $noise 0.15 0
done


# ===========================================================================
# FashionMNIST — pretraining ablation at p=15% (noise_std=0.00)
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  FashionMNIST — Pretraining Ablation (p=15%, noise=0.00)"  | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for pretrain in 1 5 10; do
    run_experiment "pretrain_fmnist_${pretrain}ep_r0.15" FashionMNIST 0.00 0.15 $pretrain
done


# ===========================================================================
# MNIST — noise ablation at p=15% (pretrain_epochs=0)
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  MNIST — Noise Ablation (p=15%, pretrain=0)"               | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for noise in 0.00 0.01 0.05 0.10 0.20; do
    run_experiment "noise_mnist_${noise}_r0.15" MNIST $noise 0.15 0
done

# ===========================================================================
# MNIST — pretraining ablation at p=15% (noise_std=0.00)
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  MNIST — Pretraining Ablation (p=15%, noise=0.00)"         | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for pretrain in 1 5 10; do
    run_experiment "pretrain_mnist_${pretrain}ep_r0.15" MNIST 0.00 0.15 $pretrain
done


# ===========================================================================
# DONE
# ===========================================================================
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_MINUTES=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECONDS=$(( TOTAL_ELAPSED % 60 ))

echo "" | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"
echo "  All ablation experiments complete"                          | tee -a "$SUMMARY_LOG"
echo "  Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"           | tee -a "$SUMMARY_LOG"
echo "  Finished:   $(date)"                                        | tee -a "$SUMMARY_LOG"
echo "  Results:    $RESULTS_CSV"                                   | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"
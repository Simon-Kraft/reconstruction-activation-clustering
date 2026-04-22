#!/bin/bash
# run_mnist_experiments.sh — Run all experiments for MNIST.
#
# Usage:
#   chmod +x run_mnist_experiments.sh
#   ./run_mnist_experiments.sh

set -e

LOGS_DIR="logs/mnist"
mkdir -p "$LOGS_DIR"

SUMMARY_LOG="$LOGS_DIR/summary.log"
TOTAL_START=$(date +%s)

echo "============================================================" | tee "$SUMMARY_LOG"
echo "  MNIST Experiment Suite" | tee -a "$SUMMARY_LOG"
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

    AC_F1=$(grep "AC overall F1" "$logfile"    | tail -1 | awk '{print $NF}')
    RAW_F1=$(grep "Raw overall F1" "$logfile"  | tail -1 | awk '{print $NF}')
    CA=$(grep "Clean accuracy" "$logfile"      | tail -1 | awk '{print $NF}')
    ASR=$(grep "ASR (0→1)" "$logfile"          | tail -1 | awk '{print $NF}')

    echo "   time:      ${MINUTES}m ${SECONDS}s" | tee -a "$SUMMARY_LOG"
    echo "   clean_acc: $CA" | tee -a "$SUMMARY_LOG"
    echo "   asr (0→1): $ASR" | tee -a "$SUMMARY_LOG"
    echo "   AC F1:     $AC_F1" | tee -a "$SUMMARY_LOG"
    echo "   Raw F1:    $RAW_F1" | tee -a "$SUMMARY_LOG"
}

# ===========================================================================
# GROUP 1 — Core Replication
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 1: Core Replication (Geiping, fc1, 3 poison rates)" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

run_experiment "mn_g1_geiping_fc1_r0.10" --poison_rate 0.10 --no_plots
run_experiment "mn_g1_geiping_fc1_r0.15" --poison_rate 0.15 --no_plots
run_experiment "mn_g1_geiping_fc1_r0.33" --poison_rate 0.33 --no_plots

# ===========================================================================
# GROUP 2 — Clean Originals Baseline (BadNets)
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 2: Clean Originals Baseline (BadNets, fc1)"         | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

run_experiment "mn_g2_badnets_fc1_r0.10" --poison_rate 0.10 --reconstruction_method badnets --no_plots
run_experiment "mn_g2_badnets_fc1_r0.15" --poison_rate 0.15 --reconstruction_method badnets --no_plots
run_experiment "mn_g2_badnets_fc1_r0.33" --poison_rate 0.33 --reconstruction_method badnets --no_plots

# ===========================================================================
# DONE
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
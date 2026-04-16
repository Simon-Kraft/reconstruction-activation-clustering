#!/bin/bash
# run_multilayer_eval.sh — Multi-layer fusion evaluation for Section 5.4.3
#
# Usage:
#   chmod +x run_multilayer_eval.sh
#   ./run_multilayer_eval.sh
#   python plot_multilayer_results.py

set -e

LOGS_DIR="logs/multilayer"
mkdir -p "$LOGS_DIR"

SUMMARY_LOG="$LOGS_DIR/summary.log"
RESULTS_CSV="$LOGS_DIR/multilayer_results.csv"
TOTAL_START=$(date +%s)

echo "============================================================" | tee "$SUMMARY_LOG"
echo "  Multi-Layer Fusion Evaluation (Section 5.4.3)"             | tee -a "$SUMMARY_LOG"
echo "  Started: $(date)"                                           | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"

# Write CSV header — pipe-delimited to avoid conflict with layer names
echo "dataset|layers|poison_rate|ac_f1|raw_f1|clean_acc|asr" > "$RESULTS_CSV"

run_experiment() {
    local label="$1"
    local dataset="$2"
    local layers="$3"
    local poison_rate="$4"
    local logfile="$LOGS_DIR/${label}.log"

    local cmd="python pipeline.py --dataset $dataset --layers $layers --poison_rate $poison_rate --no_plots"

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

    # Extract metrics — strip % signs and whitespace
    AC_F1=$(grep  "AC overall F1"  "$logfile" | tail -1 | awk '{print $NF}' | tr -d '%')
    RAW_F1=$(grep "Raw overall F1" "$logfile" | tail -1 | awk '{print $NF}' | tr -d '%')
    CA=$(grep     "Clean accuracy" "$logfile" | tail -1 | awk '{print $NF}' | tr -d '%')
    ASR=$(grep    "ASR (0→1)"      "$logfile" | tail -1 | awk '{print $NF}' | tr -d '%')

    echo "   time:      ${MINUTES}m ${SECONDS}s" | tee -a "$SUMMARY_LOG"
    echo "   clean_acc: $CA"                      | tee -a "$SUMMARY_LOG"
    echo "   asr (0→1): $ASR"                     | tee -a "$SUMMARY_LOG"
    echo "   AC F1:     $AC_F1"                   | tee -a "$SUMMARY_LOG"
    echo "   Raw F1:    $RAW_F1"                  | tee -a "$SUMMARY_LOG"

    # Pipe-delimited to avoid conflict with comma in layer names
    echo "$dataset|$layers|$poison_rate|$AC_F1|$RAW_F1|$CA|$ASR" >> "$RESULTS_CSV"
}

# ===========================================================================
# MNIST
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  MNIST — Multi-Layer Fusion"                                | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for rate in 0.10 0.15 0.33; do
    run_experiment "ml_mnist_fc1_r${rate}"              MNIST fc1             $rate
    run_experiment "ml_mnist_conv1+fc1_r${rate}"        MNIST conv1,fc1       $rate
    run_experiment "ml_mnist_conv1+conv2+fc1_r${rate}"  MNIST conv1,conv2,fc1 $rate
done

# ===========================================================================
# FashionMNIST
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  FashionMNIST — Multi-Layer Fusion"                        | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for rate in 0.10 0.15 0.33; do
    run_experiment "ml_fmnist_fc1_r${rate}"              FashionMNIST fc1             $rate
    run_experiment "ml_fmnist_conv1+fc1_r${rate}"        FashionMNIST conv1,fc1       $rate
    run_experiment "ml_fmnist_conv1+conv2+fc1_r${rate}"  FashionMNIST conv1,conv2,fc1 $rate
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
echo "  All multi-layer experiments complete"                       | tee -a "$SUMMARY_LOG"
echo "  Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"           | tee -a "$SUMMARY_LOG"
echo "  Finished:   $(date)"                                        | tee -a "$SUMMARY_LOG"
echo "  Results:    $RESULTS_CSV"                                   | tee -a "$SUMMARY_LOG"
echo "  Run: python plot_multilayer_results.py"                     | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"
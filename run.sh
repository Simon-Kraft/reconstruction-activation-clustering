#!/bin/bash
# run_all_experiments.sh — Run all experiments and log outputs.
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
#
# Logs are saved to logs/ directory, one file per experiment.
# Summary of all results is saved to logs/summary.log

set -e

LOGS_DIR="logs"
mkdir -p "$LOGS_DIR"

SUMMARY_LOG="$LOGS_DIR/summary.log"
TOTAL_START=$(date +%s)

echo "============================================================" | tee "$SUMMARY_LOG"
echo "  Full Experiment Suite" | tee -a "$SUMMARY_LOG"
echo "  Started: $(date)" | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"

# ---------------------------------------------------------------------------
# Helper — run one experiment, log output, extract F1 scores
# ---------------------------------------------------------------------------
run_experiment() {
    local label="$1"
    local logfile="$LOGS_DIR/${label}.log"
    shift
    local cmd="python pipeline.py $@"

    echo "" | tee -a "$SUMMARY_LOG"
    echo "── $label ──" | tee -a "$SUMMARY_LOG"
    echo "   cmd: $cmd" | tee -a "$SUMMARY_LOG"

    START=$(date +%s)
    echo "============================================================" > "$logfile"
    echo "  Experiment: $label" >> "$logfile"
    echo "  Command:    $cmd" >> "$logfile"
    echo "  Started:    $(date)" >> "$logfile"
    echo "============================================================" >> "$logfile"

    # Run and tee to both log file and terminal
    eval "$cmd" 2>&1 | tee -a "$logfile"

    END=$(date +%s)
    ELAPSED=$(( END - START ))
    MINUTES=$(( ELAPSED / 60 ))
    SECONDS=$(( ELAPSED % 60 ))

    # Extract key metrics from log
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
# EXPERIMENT GROUP 1 — Core Replication
# Geiping reconstruction, fc1 only, three poison rates
# Matches Chen et al. (2018) Table 1 setup but with gradient-inverted poison
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 1: Core Replication (Geiping, fc1, 3 poison rates)" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

run_experiment "g1_geiping_fc1_r0.10" --poison_rate 0.10 --no_plots
run_experiment "g1_geiping_fc1_r0.15" --poison_rate 0.15 --no_plots
run_experiment "g1_geiping_fc1_r0.33" --poison_rate 0.33 --no_plots

# ===========================================================================
# EXPERIMENT GROUP 2 — Clean Originals Baseline (BadNets)
# Standard direct poisoning, fc1 only, three poison rates
# Establishes what AC and raw clustering look like without reconstruction
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 2: Clean Originals Baseline (BadNets, fc1)"         | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

run_experiment "g2_badnets_fc1_r0.10" --poison_rate 0.10 --use_reconstruction 0 --no_plots
run_experiment "g2_badnets_fc1_r0.15" --poison_rate 0.15 --use_reconstruction 0 --no_plots
run_experiment "g2_badnets_fc1_r0.33" --poison_rate 0.33 --use_reconstruction 0 --no_plots

# ===========================================================================
# EXPERIMENT GROUP 3 — Multi-Layer Fusion
# All meaningful layer combinations for PaperCNN
# Tests whether adding earlier layers improves detection
# Available layers: conv1, conv2, fc1, fc2
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 3: Multi-Layer Fusion (Geiping, r=0.33)"            | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

# Single layers (baselines within this group)
run_experiment "g3_layers_conv1_r0.33"            --poison_rate 0.33 --layers conv1            --no_plots
run_experiment "g3_layers_conv2_r0.33"            --poison_rate 0.33 --layers conv2            --no_plots
run_experiment "g3_layers_fc1_r0.33"              --poison_rate 0.33 --layers fc1              --no_plots

# Two-layer combinations
run_experiment "g3_layers_conv1+conv2_r0.33"      --poison_rate 0.33 --layers conv1,conv2      --no_plots
run_experiment "g3_layers_conv1+fc1_r0.33"        --poison_rate 0.33 --layers conv1,fc1        --no_plots
run_experiment "g3_layers_conv2+fc1_r0.33"        --poison_rate 0.33 --layers conv2,fc1        --no_plots

# Three-layer combinations
run_experiment "g3_layers_conv1+conv2+fc1_r0.33"  --poison_rate 0.33 --layers conv1,conv2,fc1  --no_plots

# Repeat most interesting combinations at lower poison rates
# where single-layer fc1 struggles most
run_experiment "g3_layers_conv2+fc1_r0.10"        --poison_rate 0.10 --layers conv2,fc1        --no_plots
run_experiment "g3_layers_conv2+fc1_r0.15"        --poison_rate 0.15 --layers conv2,fc1        --no_plots
run_experiment "g3_layers_conv1+conv2+fc1_r0.10"  --poison_rate 0.10 --layers conv1,conv2,fc1  --no_plots
run_experiment "g3_layers_conv1+conv2+fc1_r0.15"  --poison_rate 0.15 --layers conv1,conv2,fc1  --no_plots

# ===========================================================================
# EXPERIMENT GROUP 4 — Noise Ablation
# Tests AC detection robustness when intercepted gradients are corrupted
# Simulates imperfect gradient interception in federated learning
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 4: Noise Ablation (Geiping, fc1, r=0.33)"           | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

run_experiment "g4_noise_0.00_r0.33" --poison_rate 0.33 --noise_std 0.00 --no_plots
run_experiment "g4_noise_0.01_r0.33" --poison_rate 0.33 --noise_std 0.01 --no_plots
run_experiment "g4_noise_0.05_r0.33" --poison_rate 0.33 --noise_std 0.05 --no_plots
run_experiment "g4_noise_0.10_r0.33" --poison_rate 0.33 --noise_std 0.10 --no_plots
run_experiment "g4_noise_0.20_r0.33" --poison_rate 0.33 --noise_std 0.20 --no_plots

# ===========================================================================
# EXPERIMENT GROUP 5 — Pretrained Reconstruction Model
# Tests whether a better reconstruction model changes AC detection
# pretrain=5 gives higher quality reconstructions from the Geiping inversion
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 5: Pretrained Reconstruction (fc1, r=0.33)"         | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

run_experiment "g5_pretrain_0ep_r0.33"  --poison_rate 0.33 --pretrain_epochs 0 --no_plots
run_experiment "g5_pretrain_1ep_r0.33"  --poison_rate 0.33 --pretrain_epochs 1 --no_plots
run_experiment "g5_pretrain_5ep_r0.33"  --poison_rate 0.33 --pretrain_epochs 5 --no_plots
run_experiment "g5_pretrain_20ep_r0.33" --poison_rate 0.33 --pretrain_epochs 20 --no_plots



# ===========================================================================
# DONE
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
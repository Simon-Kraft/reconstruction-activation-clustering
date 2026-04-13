#!/bin/bash
# run_experiments.sh — Run all three poison rate experiments sequentially.
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
# Each run saves results to its own folder:
#   results/MNIST_rotating_r0.1_sub0.2_noise0.0_pre0/
#   results/MNIST_rotating_r0.15_sub0.2_noise0.0_pre0/
#   results/MNIST_rotating_r0.33_sub0.2_noise0.0_pre0/

set -e  # stop on first error

echo "============================================================"
echo "  Running all three poison rate experiments"
echo "  $(date)"
echo "============================================================"

START=$(date +%s)

echo ""
echo "── Run 1/3: poison_rate=0.10 ──"
python pipeline.py --poison_rate 0.10

echo ""
echo "── Run 2/3: poison_rate=0.15 ──"
python pipeline.py --poison_rate 0.15

echo ""
echo "── Run 3/3: poison_rate=0.33 ──"
python pipeline.py --poison_rate 0.33

END=$(date +%s)
ELAPSED=$(( END - START ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
echo "  All experiments complete"
echo "  Total time: ${MINUTES}m ${SECONDS}s"
echo "  $(date)"
echo "============================================================"
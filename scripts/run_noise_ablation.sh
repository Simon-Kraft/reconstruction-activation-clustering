#!/bin/bash
# scripts/run_noise_ablation.sh — Noise and pretraining ablation at p=15%
#
# Tests how gradient noise and reconstruction model pretraining affect
# AC detection and attack success rate, averaged across multiple seeds.
#
# Usage:
#   chmod +x scripts/run_noise_ablation.sh
#   ./scripts/run_noise_ablation.sh

set -e

LOGS_DIR="logs/noise_ablation"
mkdir -p "$LOGS_DIR"

SUMMARY_LOG="$LOGS_DIR/summary.log"
TOTAL_START=$(date +%s)

SEEDS=(41 42 43)
AC_N_COMPONENTS="2,4,6,10"
POISON_RATE="0.15"
SUBSAMPLE="0.25"

echo "============================================================" | tee "$SUMMARY_LOG"
echo "  Noise & Pretraining Ablation at p=15%"                     | tee -a "$SUMMARY_LOG"
echo "  seeds=${SEEDS[*]}"                                          | tee -a "$SUMMARY_LOG"
echo "  ac_n_components=${AC_N_COMPONENTS}"                         | tee -a "$SUMMARY_LOG"
echo "  Started: $(date)"                                           | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"

run_experiment() {
    local label="$1"
    local logfile="$LOGS_DIR/${label}.log"
    shift
    local cmd="python pipeline.py $*"

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
# GROUP 1 — FashionMNIST noise ablation
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 1: FashionMNIST — Noise Ablation (p=15%)"           | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for noise in 0.00 0.01 0.05 0.10 0.20; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "noise_fmnist_${noise}_r0.15_seed${seed}" \
            --dataset FashionMNIST \
            --noise_std             "$noise" \
            --poison_rate           "$POISON_RATE" \
            --subsample_rate        "$SUBSAMPLE" \
            --reconstruction_method geiping \
            --ac_n_components       "$AC_N_COMPONENTS" \
            --seed                  "$seed" \
            --no_plots
    done
done

# ===========================================================================
# GROUP 2 — FashionMNIST pretraining ablation
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 2: FashionMNIST — Pretraining Ablation (p=15%)"     | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for pretrain in 1 5 10; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "pretrain_fmnist_${pretrain}ep_r0.15_seed${seed}" \
            --dataset FashionMNIST \
            --noise_std             0.00 \
            --poison_rate           "$POISON_RATE" \
            --subsample_rate        "$SUBSAMPLE" \
            --reconstruction_method geiping \
            --pretrain_epochs       "$pretrain" \
            --ac_n_components       "$AC_N_COMPONENTS" \
            --seed                  "$seed" \
            --no_plots
    done
done

# ===========================================================================
# GROUP 3 — MNIST noise ablation
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 3: MNIST — Noise Ablation (p=15%)"                  | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for noise in 0.00 0.01 0.05 0.10 0.20; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "noise_mnist_${noise}_r0.15_seed${seed}" \
            --dataset MNIST \
            --noise_std             "$noise" \
            --poison_rate           "$POISON_RATE" \
            --subsample_rate        "$SUBSAMPLE" \
            --reconstruction_method geiping \
            --ac_n_components       "$AC_N_COMPONENTS" \
            --seed                  "$seed" \
            --no_plots
    done
done

# ===========================================================================
# GROUP 4 — MNIST pretraining ablation
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  GROUP 4: MNIST — Pretraining Ablation (p=15%)"            | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

for pretrain in 1 5 10; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "pretrain_mnist_${pretrain}ep_r0.15_seed${seed}" \
            --dataset MNIST \
            --noise_std             0.00 \
            --poison_rate           "$POISON_RATE" \
            --subsample_rate        "$SUBSAMPLE" \
            --reconstruction_method geiping \
            --pretrain_epochs       "$pretrain" \
            --ac_n_components       "$AC_N_COMPONENTS" \
            --seed                  "$seed" \
            --no_plots
    done
done

# ===========================================================================
# Aggregate results across seeds
# ===========================================================================
echo "" | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"
echo "  Aggregating results across seeds"                           | tee -a "$SUMMARY_LOG"
echo "###########################################################" | tee -a "$SUMMARY_LOG"

python - <<'PYEOF' | tee -a "$SUMMARY_LOG"
import re
import numpy as np

SUMMARY_LOG = "logs/noise_ablation/summary.log"
N_COMPS = [2, 4, 6, 10]

with open(SUMMARY_LOG) as f:
    text = f.read()

# Parse all blocks into data dict
# key: (label_prefix, seed) -> {k: {ac_acc, ac_f1, raw_acc, raw_f1}}
data = {}
for block_label, block_body in re.findall(
    r'── ([^\n]+) ──\n(.*?)(?=── |\Z)', text, re.DOTALL
):
    block_label = block_label.strip()
    # Match noise blocks: noise_fmnist_0.05_r0.15_seed41
    m = re.match(r'(noise_(?:fmnist|mnist)_[0-9.]+_r[0-9.]+)_seed(\d+)', block_label)
    if not m:
        # Match pretrain blocks: pretrain_fmnist_5ep_r0.15_seed41
        m = re.match(r'(pretrain_(?:fmnist|mnist)_\d+ep_r[0-9.]+)_seed(\d+)', block_label)
    if not m:
        continue
    prefix, seed = m.group(1), int(m.group(2))
    per_k = {}
    for row in re.finditer(
        r'k=(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%', block_body
    ):
        k = int(row.group(1))
        per_k[k] = {
            'ac_acc':  float(row.group(2)),
            'ac_f1':   float(row.group(3)),
            'raw_acc': float(row.group(4)),
            'raw_f1':  float(row.group(5)),
        }
    if per_k:
        data.setdefault(prefix, []).append(per_k)

def show_group(title, prefixes):
    print(f"\n{'='*70}")
    print(f"  {title}")
    for prefix in prefixes:
        seed_entries = data.get(prefix, [])
        if not seed_entries:
            continue
        print(f"\n  {prefix}  (n={len(seed_entries)} seeds)")
        header = f"    {'k':<5}" + "".join(f"  {'AC acc':>14}  {'AC F1':>14}  {'Raw acc':>14}  {'Raw F1':>14}"
                                            for _ in [None])
        print(f"    {'k':<5}  {'AC acc':>14}  {'AC F1':>14}  {'Raw acc':>14}  {'Raw F1':>14}")
        print(f"    {'-'*65}")
        for k in N_COMPS:
            vals = {m: [e[k][m] for e in seed_entries if k in e] for m in ('ac_acc','ac_f1','raw_acc','raw_f1')}
            if not vals['ac_f1']:
                continue
            row = f"    k={k:<3}"
            for metric in ('ac_acc', 'ac_f1', 'raw_acc', 'raw_f1'):
                v = vals[metric]
                row += f"  {np.mean(v):6.2f}±{np.std(v):5.2f}%"
            print(row)

# FashionMNIST noise
show_group("FashionMNIST — Noise Ablation (p=15%, pretrain=0)", [
    f"noise_fmnist_{n}_r0.15" for n in ["0.00","0.01","0.05","0.10","0.20"]
])

# FashionMNIST pretrain
show_group("FashionMNIST — Pretraining Ablation (p=15%, noise=0.00)", [
    f"pretrain_fmnist_{p}ep_r0.15" for p in [1, 5, 10]
])

# MNIST noise
show_group("MNIST — Noise Ablation (p=15%, pretrain=0)", [
    f"noise_mnist_{n}_r0.15" for n in ["0.00","0.01","0.05","0.10","0.20"]
])

# MNIST pretrain
show_group("MNIST — Pretraining Ablation (p=15%, noise=0.00)", [
    f"pretrain_mnist_{p}ep_r0.15" for p in [1, 5, 10]
])
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
echo "  All ablation experiments complete"                          | tee -a "$SUMMARY_LOG"
echo "  Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"           | tee -a "$SUMMARY_LOG"
echo "  Finished:   $(date)"                                        | tee -a "$SUMMARY_LOG"
echo "  Summary:    $SUMMARY_LOG"                                   | tee -a "$SUMMARY_LOG"
echo "  Logs:       $LOGS_DIR/"                                     | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"

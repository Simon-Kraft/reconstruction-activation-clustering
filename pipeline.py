"""
pipeline.py — End-to-end backdoor detection pipeline.

Matches Chen et al. (2018) experimental setup:
  - Rotating poison: class lm → (lm+1)%10 for all 10 classes
  - AC detection on fc1 activations
  - Raw clustering baseline on pixel values

Run with:
    python pipeline.py --dataset MNIST --poison_rate 0.15
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse

import config as C
from data             import load_dataset, build_poisoned_dataset
from data.trigger     import TriggerConfig
from models           import PaperCNN, train, compute_asr, save_model
from models.train     import evaluate, load_model
from clustering       import extract_activations
from clustering.extractor import extract_raw_pixels
from ac_sweep         import run_ac_sweep

def parse_args():
    parser = argparse.ArgumentParser(
        description='Backdoor detection pipeline — rotating poison + AC'
    )
    parser.add_argument('--dataset',           type=str,   default=None,
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10'],
                        help='Dataset to use')
    parser.add_argument('--poison_rate',       type=float, default=None,
                        help='Fraction of each class to poison (e.g. 0.15)')
    parser.add_argument('--subsample_rate',    type=float, default=None,
                        help='Fraction of training set to use (default 0.25)')
    parser.add_argument('--noise_std',         type=float, default=None,
                        help='Gaussian noise std on intercepted gradients')
    parser.add_argument('--pretrain_epochs',   type=int,   default=None,
                        help='Epochs to pretrain reconstruction model')
    parser.add_argument('--reconstruction_method', type=str, default=None,
                        choices=['geiping', 'dlg', 'badnets'],
                        help='geiping = cosine inversion, dlg = L2 inversion, badnets = no reconstruction')
    parser.add_argument('--replace_originals', action='store_true',
                        help='Replace reconstructed source images instead of appending')
    parser.add_argument('--layer',            type=str,   default=None,
                        help="Comma-separated layer names, e.g. 'fc1' or 'conv1,fc1'")
    parser.add_argument('--seed',              type=int,   default=None,
                        help='Random seed')
    parser.add_argument('--ac_n_components',    type=str,   default=None,
                        help='Comma-separated list of ICA/PCA components to evaluate, e.g. "2,4,6,10"')
    parser.add_argument('--no_plots',          action='store_true',
                        help='Suppress all visualisation')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1 — Load dataset
# ---------------------------------------------------------------------------
def step_load_dataset():
    print("\n── Step 1: Load dataset ──")
    dataset_info = load_dataset(C.DATASET_NAME, data_dir=C.DATASETS_DIR)
    test_loader  = DataLoader(
        dataset_info.test,
        batch_size = C.TEST_BATCH_SIZE,
        shuffle    = False,
    )
    return dataset_info, test_loader


# ---------------------------------------------------------------------------
# Step 2 — Build rotating poisoned dataset
# ---------------------------------------------------------------------------
def step_build_dataset(dataset_info):
    print("\n── Step 2: Build rotating poisoned dataset ──")
    model = PaperCNN.for_dataset(dataset_info).to(C.DEVICE)
    mixed = build_poisoned_dataset(
        cfg        = C.POISON_CFG,
        model      = model,
        device     = C.DEVICE,
        cache_path = C.CACHE_DATASET_PATH,
    )
    return mixed


# ---------------------------------------------------------------------------
# Step 3 — Train backdoor model
# ---------------------------------------------------------------------------
def step_train(mixed_dataset, dataset_info, test_loader):
    print("\n── Step 3: Train backdoor model ──")
    model = PaperCNN.for_dataset(dataset_info).to(C.DEVICE)

    if os.path.exists(C.BACKDOOR_MODEL_PATH):
        print("  Loading cached model to save time")
        return load_model(model, C.BACKDOOR_MODEL_PATH, C.DEVICE)

    train_loader = DataLoader(
        mixed_dataset,
        batch_size = C.TRAIN_BATCH_SIZE,
        shuffle    = True,
    )
    trained = train(
        model        = model,
        train_loader = train_loader,
        test_loader  = test_loader,
        device       = C.DEVICE,
        epochs       = C.TRAIN_EPOCHS,
        lr           = C.TRAIN_LR,
        label        = "BACKDOOR",
    )
    save_model(trained, C.BACKDOOR_MODEL_PATH)
    return trained


# ---------------------------------------------------------------------------
# Step 4 — Verify backdoor
# ---------------------------------------------------------------------------
def step_verify(model, dataset_info, test_loader):
    print("\n── Step 4: Verify backdoor ──")
    ca      = evaluate(model, test_loader, C.DEVICE)
    trigger = TriggerConfig.for_dataset(
        img_size = dataset_info.img_size,
        mean     = dataset_info.mean,
        std      = dataset_info.std,
    )
    asr = compute_asr(
        model        = model,
        dataset_info = dataset_info,
        trigger      = trigger,
        device       = C.DEVICE,
        all_classes  = True,
    )
    print(f"\n  Clean accuracy: {ca:.2%}")
    if asr < 0.5:
        print("  ⚠️  Low avg ASR — consider increasing poison_rate or train_epochs.")
    return ca, asr


# ---------------------------------------------------------------------------
# Step 5 — Extract activations (fc1 + raw pixels)
# ---------------------------------------------------------------------------
def step_extract(model, mixed_dataset):
    print("\n── Step 5: Extract activations ──")
    ac_extraction  = extract_activations(
        model       = model,
        dataset     = mixed_dataset,
        layer_name  = C.AC_LAYER,
        device      = C.DEVICE,
    )
    raw_extraction = extract_raw_pixels(mixed_dataset)
    return ac_extraction, raw_extraction


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # ── Apply argparse overrides ──────────────────────────────────────────
    if args.dataset is not None:
        C.DATASET_NAME            = args.dataset
        C.POISON_CFG.dataset_name = args.dataset
    if args.poison_rate           is not None: C.POISON_CFG.poison_rate           = args.poison_rate
    if args.subsample_rate        is not None: C.POISON_CFG.subsample_rate        = args.subsample_rate
    if args.noise_std             is not None: C.POISON_CFG.noise_std             = args.noise_std
    if args.pretrain_epochs       is not None: C.POISON_CFG.pretrain_epochs       = args.pretrain_epochs
    if args.reconstruction_method is not None: C.POISON_CFG.reconstruction_method = args.reconstruction_method
    if args.replace_originals:                 C.POISON_CFG.replace_originals     = True
    if args.layer                 is not None: C.AC_LAYER   = args.layer
    if args.seed                  is not None: C.SEED       = args.seed
    if args.no_plots:                          C.SHOW_PLOTS = False

    n_components_list = (
        [int(x) for x in args.ac_n_components.split(',')]
        if args.ac_n_components is not None
        else [C.AC_N_COMPONENTS]
    )

    # Recompute analysis threshold after poison_rate may have changed
    C.ANALYSIS_CFG.max_poison_rate = C.POISON_CFG.poison_rate + 0.05

    # ── Recompute paths after all overrides ───────────────────────────────
    _EXP_ID = (
        f"{C.DATASET_NAME}_rotating"
        f"_r{C.POISON_CFG.poison_rate}"
        f"_sub{C.POISON_CFG.subsample_rate}"
        f"_recon{C.POISON_CFG.reconstruction_method}"
        f"_replace{int(C.POISON_CFG.replace_originals)}"
        f"_noise{C.POISON_CFG.noise_std}"
        f"_pre{C.POISON_CFG.pretrain_epochs}"
        f"_seed{C.SEED}"
    )
    C.CACHE_DATASET_PATH  = C.DATASETS_DIR   + f'{_EXP_ID}.pt'
    C.BACKDOOR_MODEL_PATH = C.CHECKPOINT_DIR + f'{_EXP_ID}.pt'
    C.RESULTS_DIR         = f'results/{_EXP_ID}/'

    # ── Create directories ────────────────────────────────────────────────
    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(C.RESULTS_DIR,    exist_ok=True)
    os.makedirs(C.DATASETS_DIR,   exist_ok=True)

    # ── Set seeds ─────────────────────────────────────────────────────────
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)

    print("=" * 60)
    print("  Backdoor Detection Pipeline  (Rotating Poison)")
    print(f"  device            = {C.DEVICE}")
    print(f"  seed              = {C.SEED}")
    print(f"  dataset           = {C.DATASET_NAME}")
    print(f"  poison scheme     = lm → (lm+1) mod n  for all classes")
    print(f"  poison_rate       = {C.POISON_CFG.poison_rate:.0%}")
    print(f"  subsample         = {C.POISON_CFG.subsample_rate:.0%}")
    print(f"  reconstruction    = {C.POISON_CFG.reconstruction_method}")
    print(f"  replace_originals = {C.POISON_CFG.replace_originals}")
    print(f"  pretrain          = {C.POISON_CFG.pretrain_epochs} epochs")
    print(f"  noise_std         = {C.POISON_CFG.noise_std}")
    print(f"  layer             = {C.AC_LAYER}")
    print(f"  ac_n_components   = {n_components_list}")
    print("=" * 60)

    # ── Steps 1–5: run once ───────────────────────────────────────────────
    dataset_info, test_loader     = step_load_dataset()
    mixed_dataset                 = step_build_dataset(dataset_info)
    model                         = step_train(mixed_dataset, dataset_info, test_loader)
    ca, asr                       = step_verify(model, dataset_info, test_loader)
    ac_extraction, raw_extraction = step_extract(model, mixed_dataset)

    # ── Steps 6–9: sweep over n_components (logic lives in ac_sweep.py) ──
    all_results = run_ac_sweep(
        ac_extraction     = ac_extraction,
        raw_extraction    = raw_extraction,
        n_components_list = n_components_list,
        base_results_dir  = C.RESULTS_DIR,
        mixed_dataset     = mixed_dataset,
        dataset_info      = dataset_info,
    )

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pipeline complete")
    print(f"  Clean accuracy:  {ca:.2%}")
    print(f"  ASR (avg):       {asr:.2%}")
    print()
    print(f"  {'k':<5}  {'AC acc':>8}  {'AC F1':>8}  {'Raw acc':>8}  {'Raw F1':>8}")
    print(f"  {'-'*45}")
    for k, (ac_r, raw_r) in all_results.items():
        print(
            f"  k={k:<3}  "
            f"{ac_r.overall_accuracy:>8.2%}  {ac_r.overall_f1:>8.2%}  "
            f"{raw_r.overall_accuracy:>8.2%}  {raw_r.overall_f1:>8.2%}"
        )
    print(f"\n  Results saved to: {C.RESULTS_DIR}")
    print("=" * 60)
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
import sys
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
    parser.add_argument('--tv_weight',         type=float, default=None,
                        help=f'Total-variation weight for Geiping reconstruction '
                             f'(default: {C.POISON_CFG.recon_tv_weight})')
    parser.add_argument('--iterations',        type=int,   default=None,
                        help=f'Optimisation steps for Geiping reconstruction '
                             f'(default: {C.POISON_CFG.recon_iterations})')
    parser.add_argument('--reconstruction_method', type=str, default=None,
                        choices=['geiping', 'badnets'],
                        help='geiping = cosine-similarity gradient inversion, badnets = no reconstruction')
    parser.add_argument('--layer',            type=str,   default=None,
                        help="Comma-separated layer names, e.g. 'fc1' or 'conv1,fc1'")
    parser.add_argument('--seed',              type=int,   default=None,
                        help='Random seed')
    parser.add_argument('--ac_n_components',    type=str,   default=None,
                        help='Comma-separated list of ICA/PCA components to evaluate, e.g. "2,4,6,10"')
    parser.add_argument('--no_plots',          action='store_true',
                        help='Suppress all visualisation')
    parser.add_argument('--lr_decay',          action='store_true',
                        help="Decay the reconstruction learning rate 10x at 3/8, 5/8, "
                             "7/8 of iterations (Geiping et al., Appendix C). Off by default.")
    parser.add_argument('--device',            type=str,   default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'],
                        help="Compute device. 'auto' = cuda if available else cpu "
                             "(unchanged default behaviour). 'mps' explicitly opts into "
                             "Apple Silicon GPU acceleration — since MPS cannot run "
                             "gradient-inversion's double-backward through real max "
                             "pooling, choosing 'mps' automatically switches PaperCNN "
                             "(both the reconstruction model and the backdoor model) to "
                             "a differentiable soft-max-pool approximation. CPU/CUDA "
                             "runs are unaffected and use the exact architecture as before.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1 — Load dataset
# ---------------------------------------------------------------------------
def step_load_dataset():
    print("\n── Step 1: Load dataset ──")
    dataset_info = load_dataset(C.DATASET_NAME, data_dir=C.RAW_DATA_DIR)
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
    model = PaperCNN.for_dataset(
        dataset_info, differentiable_pool=C.USE_DIFFERENTIABLE_POOL
    ).to(C.DEVICE)
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
    model = PaperCNN.for_dataset(
        dataset_info, differentiable_pool=C.USE_DIFFERENTIABLE_POOL
    ).to(C.DEVICE)

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


class _Tee:
    """Duplicates writes to both the original stream and a log file."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


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
    if args.tv_weight             is not None: C.POISON_CFG.recon_tv_weight       = args.tv_weight
    if args.iterations            is not None: C.POISON_CFG.recon_iterations      = args.iterations
    if args.reconstruction_method is not None: C.POISON_CFG.reconstruction_method = args.reconstruction_method
    if args.layer                 is not None: C.AC_LAYER   = args.layer
    if args.seed                  is not None: C.SEED       = args.seed
    if args.no_plots:                          C.SHOW_PLOTS = False
    if args.lr_decay:                          C.POISON_CFG.recon_lr_decay = True

    # ── Resolve device ──────────────────────────────────────────────────────
    # 'auto' preserves the original default exactly (cuda if available, else
    # cpu) — MPS is never silently auto-selected, only via explicit --device
    # mps. Choosing mps switches PaperCNN to the differentiable-pool variant
    # for both the reconstruction model and the backdoor model, so the two
    # stay architecturally identical to each other (just a different pooling
    # op than the CPU/CUDA default).
    if args.device == 'auto':
        pass  # C.DEVICE already computed at import time (cuda > cpu)
    elif args.device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available on this machine.")
        C.DEVICE = torch.device('mps')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available on this machine.")
        C.DEVICE = torch.device('cuda')
    else:
        C.DEVICE = torch.device('cpu')
    C.USE_DIFFERENTIABLE_POOL = (C.DEVICE.type == 'mps')

    n_components_list = (
        [int(x) for x in args.ac_n_components.split(',')]
        if args.ac_n_components is not None
        else [C.AC_N_COMPONENTS]
    )

    # Recompute analysis threshold after poison_rate may have changed
    C.ANALYSIS_CFG.max_poison_rate = C.POISON_CFG.poison_rate + 0.05

    # ── Recompute paths after all overrides ───────────────────────────────
    # Everything produced by this run — cached poisoned dataset, trained
    # checkpoint, results, and the run's own log — lives together under
    # outputs/<dataset>/<exp_id>/ so a single experiment is easy to find or
    # delete, and all runs for one dataset sit next to each other. The
    # reconstruction method leads the exp_id so geiping/badnets runs for
    # the same dataset sort and group together in a folder listing.
    # lr_decay, softpool, tv_weight and iterations are only appended when
    # explicitly overridden on the CLI, so existing exp_id strings / cached
    # outputs for ordinary runs that don't touch these flags are completely
    # unchanged and still hit their caches.
    _lr_decay_tag = "_lrdecay1" if C.POISON_CFG.recon_lr_decay   else ""
    _softpool_tag = "_softpool1" if C.USE_DIFFERENTIABLE_POOL    else ""
    _tv_tag       = f"_tv{args.tv_weight}"   if args.tv_weight  is not None else ""
    _iter_tag     = f"_iter{args.iterations}" if args.iterations is not None else ""
    _EXP_ID = (
        f"{C.POISON_CFG.reconstruction_method}"
        f"_rotating"
        f"_r{C.POISON_CFG.poison_rate}"
        f"_sub{C.POISON_CFG.subsample_rate}"
        f"_noise{C.POISON_CFG.noise_std}"
        f"_pre{C.POISON_CFG.pretrain_epochs}"
        f"{_tv_tag}"
        f"{_iter_tag}"
        f"{_lr_decay_tag}"
        f"{_softpool_tag}"
        f"_seed{C.SEED}"
    )
    C.EXPERIMENT_DIR      = os.path.join(C.OUTPUTS_DIR, C.DATASET_NAME, _EXP_ID)
    C.CACHE_DATASET_PATH  = os.path.join(C.EXPERIMENT_DIR, 'dataset.pt')
    C.BACKDOOR_MODEL_PATH = os.path.join(C.EXPERIMENT_DIR, 'model.pt')
    C.RESULTS_DIR         = os.path.join(C.EXPERIMENT_DIR, 'results') + os.sep
    C.RUN_LOG_PATH        = os.path.join(C.EXPERIMENT_DIR, 'run.log')

    # ── Create directories ────────────────────────────────────────────────
    os.makedirs(C.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(C.RESULTS_DIR,  exist_ok=True)

    # ── Tee all console output into the experiment's own run.log ─────────
    _log_file = open(C.RUN_LOG_PATH, 'w')
    sys.stdout = _Tee(sys.__stdout__, _log_file)

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
    print(f"  pretrain          = {C.POISON_CFG.pretrain_epochs} epochs")
    print(f"  noise_std         = {C.POISON_CFG.noise_std}")
    print(f"  tv_weight         = {C.POISON_CFG.recon_tv_weight}")
    print(f"  iterations        = {C.POISON_CFG.recon_iterations}")
    print(f"  lr_decay          = {C.POISON_CFG.recon_lr_decay}")
    print(f"  differentiable_pool = {C.USE_DIFFERENTIABLE_POOL}")
    print(f"  layer             = {C.AC_LAYER}")
    print(f"  ac_n_components   = {n_components_list}")
    print(f"  experiment_dir    = {C.EXPERIMENT_DIR}")
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
    print(f"\n  Experiment saved to: {C.EXPERIMENT_DIR}")
    print("=" * 60)

    sys.stdout = sys.__stdout__
    _log_file.close()
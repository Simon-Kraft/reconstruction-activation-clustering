"""
pipeline.py — End-to-end backdoor detection pipeline.

Correct flow:
  1. Load full MNIST (60k train, 10k test)
  2. (Optional) Pretrain the reconstruction model
  3. Pick N_POISON samples from train, reconstruct via DLG, add trigger,
     flip label → TARGET_CLASS. Replace those samples in the training set.
     [result cached to disk]
  4. Train a backdoor model on the full mixed dataset (~60k samples)
  5. Extract activations for TARGET_CLASS samples only
  6. Run Activation Clustering detection
  7. Save metrics and print summary

Edit config.py to change any hyperparameter.
"""

import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import config as C
from model import LargeCNN, MidCNN, SmallCNN
from train   import train
from poison  import build_poisoned_dataset, MixedDataset
from extract import extract_features
from detect  import run_detection

torch.manual_seed(C.SEED)
np.random.seed(C.SEED)

MODELS = {
    'LargeCNN': LargeCNN,
    'MidCNN':   MidCNN,
    'SmallCNN': SmallCNN,
}

# ---------------------------------------------------------------------------
# Step 1 — Load MNIST
# ---------------------------------------------------------------------------
def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((C.MEAN,), (C.STD,)),
    ])
    train_raw = datasets.MNIST(C.DATA_DIR, train=True,  download=True, transform=transform)
    test_raw  = datasets.MNIST(C.DATA_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_raw, batch_size=C.TEST_BATCH_SIZE, shuffle=False)
    print(f"✅ MNIST loaded  — train: {len(train_raw)}, test: {len(test_raw)}")
    return train_raw, test_loader


# ---------------------------------------------------------------------------
# Step 2 — (Optional) Pretrain reconstruction model
# ---------------------------------------------------------------------------
def get_recon_model(train_raw, test_loader):
    model = MODELS[C.MODEL]().to(C.DEVICE)
    if C.RECON_PRETRAIN_EPOCHS > 0:
        print(f"\n── Pretraining reconstruction model "
              f"({C.RECON_PRETRAIN_EPOCHS} epochs) ──")
        pretrain_loader = DataLoader(train_raw, batch_size=C.TRAIN_BATCH_SIZE, shuffle=True)
        train(model, pretrain_loader, test_loader,
              epochs=C.RECON_PRETRAIN_EPOCHS, label="[RECON-PRETRAIN]")
    else:
        print("\n── Reconstruction model: randomly initialised (0 pretrain epochs) ──")
    return model


# ---------------------------------------------------------------------------
# Step 3 — Build (or load) mixed poisoned dataset
# ---------------------------------------------------------------------------
def get_mixed_dataset(train_raw, recon_model):
    if os.path.exists(C.RECON_DATASET_PATH):
        print(f"\n── Loading cached mixed dataset from {C.RECON_DATASET_PATH} ──")
        return MixedDataset.load(C.RECON_DATASET_PATH)

    print(f"\n── Building mixed dataset "
          f"(n_poison={C.N_POISON}, target_class={C.TARGET_CLASS}, "
          f"noise_std={C.DLG_NOISE_STD}) ──")
    dataset = build_poisoned_dataset(
        train_raw      = train_raw,
        recon_model    = recon_model,
        n_poison       = C.N_POISON,
        target_class   = C.TARGET_CLASS,
        dlg_iterations = C.DLG_ITERATIONS,
        dlg_noise_std  = C.DLG_NOISE_STD,
    )
    dataset.save(C.RECON_DATASET_PATH)
    return dataset


# ---------------------------------------------------------------------------
# Step 4 — Train backdoor model on the full mixed dataset
# ---------------------------------------------------------------------------
def train_backdoor_model(mixed_dataset, test_loader, recon_model):
    print(f"\n── Training backdoor model on {len(mixed_dataset)} samples "
          f"({C.TRAIN_EPOCHS} epochs) ──")
    mixed_loader = DataLoader(mixed_dataset, batch_size=C.TRAIN_BATCH_SIZE, shuffle=True)
    backdoor_model = train(recon_model, mixed_loader, test_loader,
                           epochs=C.TRAIN_EPOCHS, label="[BACKDOOR]")
    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    torch.save(backdoor_model.state_dict(), C.BACKDOOR_MODEL_PATH)
    print(f"Model saved → {C.BACKDOOR_MODEL_PATH}")
    return backdoor_model


# ---------------------------------------------------------------------------
# Step 5 — Extract activations (target class only)
# ---------------------------------------------------------------------------
def get_activations(backdoor_model, mixed_dataset):
    print(f"\n── Extracting activations (class {C.TARGET_CLASS} only) ──")
    extraction = extract_features(
        model        = backdoor_model,
        dataset      = mixed_dataset,
        target_class = C.TARGET_CLASS,
    )
    print("Feature shapes:")
    for name, feat in extraction["feats"].items():
        print(f"   {name:6s} → {feat.shape}")
    return extraction


# ---------------------------------------------------------------------------
# Step 6 & 7 — Detect and report
# ---------------------------------------------------------------------------
def detect_and_report(extraction):
    print(f"\n── Running Activation Clustering detection ──")
    df = run_detection(extraction)

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    torch.save({"metrics": df.to_dict(), "extraction": extraction}, C.RESULTS_METRICS_PATH)

    print("\n📊 Detection metrics per layer:")
    print(df.round(4).to_string())

    best_layer = df["composite"].idxmax()
    print(f"\n🏆 Best layer by composite score: '{best_layer}'")
    print(f"   silhouette = {df.loc[best_layer, 'silhouette']:.4f}")
    print(f"   purity     = {df.loc[best_layer, 'purity']:.4f}")
    print(f"   lda        = {df.loc[best_layer, 'lda']:.4f}")
    print(f"   composite  = {df.loc[best_layer, 'composite']:.4f}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Backdoor AC Pipeline")
    print(f"  device={C.DEVICE}  seed={C.SEED}")
    print(f"  pretrain_epochs={C.RECON_PRETRAIN_EPOCHS}  noise_std={C.DLG_NOISE_STD}")
    print("=" * 60)

    train_raw, test_loader = load_mnist()
    recon_model            = get_recon_model(train_raw, test_loader)
    mixed_dataset          = get_mixed_dataset(train_raw, recon_model)
    backdoor_model         = train_backdoor_model(mixed_dataset, test_loader, recon_model)
    extraction             = get_activations(backdoor_model, mixed_dataset)
    metrics_df             = detect_and_report(extraction)

    print("\n✅ Pipeline complete. Results saved to", C.RESULTS_DIR)
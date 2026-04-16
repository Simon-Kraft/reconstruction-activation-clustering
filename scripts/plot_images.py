"""
report/figures/visualize_reconstructions.py

Place this script in a subfolder, e.g. report/figures/.
Outputs (PDF + PNG) are saved in the same directory as this script.
The project root is inferred automatically so it can be run from anywhere:

    python report/figures/visualize_reconstructions.py

Layout — 4 images total:

         MNIST (bold)          FashionMNIST (bold)
         [original img]        [original img]
         0                     T-shirt

         [recon+trig img]      [recon+trig img]
         1 (poisoned)          Trouser (poisoned)
"""

import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data.loader         import load_dataset
from data.trigger        import TriggerConfig
from data.reconstruction import ReconConfig, intercept_gradients, reconstruct
from models.cnn          import PaperCNN

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size']   = 8

DEVICE = torch.device('cpu')
SEED   = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SOURCE_CLASS   = 0
TARGET_CLASS   = 1
DLG_ITERATIONS = 75
DLG_LR         = 0.1
DLG_TV_WEIGHT  = 1e-4

DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets/')

FASHION_NAMES = {
    0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',  4: 'Coat',
    5: 'Sandal',  6: 'Shirt',   7: 'Sneaker',  8: 'Bag',     9: 'Boot',
}


def reconstruct_one(model, img, label, dataset_info):
    recon_cfg = ReconConfig(
        iterations  = DLG_ITERATIONS,
        lr          = DLG_LR,
        tv_weight   = DLG_TV_WEIGHT,
        noise_std   = 0.0,
        clamp_range = dataset_info.clamp_range,
        verbose     = False,
    )
    grads    = intercept_gradients(model, img, int(label), dev=DEVICE)
    recon, _ = reconstruct(
        model            = model,
        target_gradients = grads,
        img_shape        = torch.Size([1, *img.shape]),
        cfg              = recon_cfg,
        dev              = DEVICE,
    )
    return recon.squeeze(0).cpu()


def to_display(tensor, mean, std):
    arr = tensor.detach().float().numpy()
    for c in range(arr.shape[0]):
        arr[c] = arr[c] * std[c] + mean[c]
    return arr.clip(0, 1).squeeze(0)


def get_first_of_class(dataset, cls):
    for i in range(len(dataset)):
        img, lbl = dataset[i]
        if int(lbl) == cls:
            return img
    raise ValueError(f"Class {cls} not found in dataset.")


# ── Collect images ────────────────────────────────────────────────────────────
columns = []

for dataset_name in ['MNIST', 'FashionMNIST']:
    print(f"Processing {dataset_name}...")
    dataset_info = load_dataset(dataset_name, data_dir=DATA_DIR)
    model        = PaperCNN.for_dataset(dataset_info).to(DEVICE)
    trigger      = TriggerConfig.for_dataset(
        img_size = dataset_info.img_size,
        mean     = dataset_info.mean,
        std      = dataset_info.std,
    )
    mean = dataset_info.mean
    std  = dataset_info.std

    orig_img = get_first_of_class(dataset_info.test, SOURCE_CLASS)

    print(f"  Reconstructing class {SOURCE_CLASS}→{TARGET_CLASS}...")
    recon      = reconstruct_one(model, orig_img, SOURCE_CLASS, dataset_info)
    recon_trig = trigger.inject(recon)

    if dataset_name == 'MNIST':
        src_label = f'{SOURCE_CLASS}'
        tgt_label = f'{TARGET_CLASS} (poisoned)'
    else:
        src_label = f'{FASHION_NAMES[SOURCE_CLASS]}'
        tgt_label = f'{FASHION_NAMES[TARGET_CLASS]} (poisoned)'

    columns.append((
        dataset_name,
        to_display(orig_img.cpu(), mean, std),
        to_display(recon_trig,     mean, std),
        src_label,
        tgt_label,
    ))
    print(f"  Done.")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 2,
    figsize     = (2.8, 3.2),
    gridspec_kw = {'hspace': 0.20, 'wspace': 0.08},
)

for col_idx, (dataset_name, orig_arr, recon_trig_arr, src_label, tgt_label) in enumerate(columns):

    axes[0, col_idx].set_title(
        dataset_name,
        fontsize   = 10,
        fontweight = 'bold',
        pad        = 10,
    )

    for row_idx, (arr, img_label) in enumerate([
        (orig_arr,       src_label),
        (recon_trig_arr, tgt_label),
    ]):
        ax = axes[row_idx, col_idx]
        ax.imshow(arr, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color('#bbbbbb')
        ax.set_xlabel(img_label, fontsize=8.5, labelpad=2, color='#333333')
        if col_idx == 0:
            ax.set_ylabel(
                ['Original', 'Reconstructed+Trigger'][row_idx],
                fontsize = 7.5,
                labelpad = 8,
                fontweight = 'bold',
                color    = '#333333',
                va       = 'center',
            )

# ── Save outputs next to this script ─────────────────────────────────────────
out_pdf = os.path.join(SCRIPT_DIR, 'reconstruction_figure.pdf')
out_png = os.path.join(SCRIPT_DIR, 'reconstruction_figure.png')

plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_png}")
plt.show()
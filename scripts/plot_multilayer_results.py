"""
plot_multilayer_results.py
Reads logs/multilayer/multilayer_results.csv (pipe-delimited) and produces
a publication-quality figure showing AC F1 vs poison rate for each layer
configuration, one subplot per dataset.

Run from the project root:
    python plot_multilayer_results.py

If you already have results in the old comma-delimited format, pass --fix-csv
to attempt automatic conversion.
"""

import os
import argparse
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size']   = 9

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH   = 'logs/multilayer/multilayer_results.csv'
OUTPUT_DIR = 'logs/multilayer/'

DATASETS = ['MNIST', 'FashionMNIST']
RATES    = [0.10, 0.15, 0.33]

LAYER_STYLES = {
    'fc1':             {'label': 'fc1 only',           'color': '#2166ac', 'marker': 'o',  'ls': '-'},
    # 'conv1':           {'label': 'conv1 only',          'color': '#984ea3', 'marker': 'D',  'ls': ':'},
    # 'conv1,conv2':     {'label': 'conv1 + conv2',       'color': '#ff7f00', 'marker': 'v',  'ls': ':'},
    'conv1,fc1':       {'label': 'conv1 + fc1',         'color': '#d6604d', 'marker': 's',  'ls': '--'},
    'conv1,conv2,fc1': {'label': 'conv1 + conv2 + fc1', 'color': '#4dac26', 'marker': '^',  'ls': '-.'},
}


# ── Load CSV (pipe-delimited) ─────────────────────────────────────────────────
def load_results(csv_path):
    """
    Parse pipe-delimited CSV into:
        data[dataset][layers][poison_rate] = ac_f1 (float, 0-100 scale)
    """
    data = {}

    with open(csv_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    # First line is header
    header = lines[0].split('|')
    col    = {name: i for i, name in enumerate(header)}

    for line in lines[1:]:
        parts = line.split('|')
        if len(parts) < len(header):
            continue

        ds    = parts[col['dataset']].strip()
        lyr   = parts[col['layers']].strip()
        rate  = float(parts[col['poison_rate']].strip())

        raw = parts[col['ac_f1']].strip().replace('%', '').replace('_', '')
        try:
            ac_f1 = float(raw)
        except ValueError:
            ac_f1 = float('nan')

        # Normalise to 0-100 scale (values may come in as 0-1 or 0-100)
        if ac_f1 <= 1.0:
            ac_f1 *= 100.0

        data.setdefault(ds, {}).setdefault(lyr, {})[rate] = ac_f1

    return data


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(data, output_dir):
    fig, axes = plt.subplots(
        1, 2,
        figsize     = (7.0, 3.2),
        sharey      = True,
        gridspec_kw = {'wspace': 0.08},
    )

    x_ticks       = [0, 15, 33]
    x_tick_labels = ['10%', '15%', '33%']

    x_ticks       = [0, 1, 2]          # equal-spaced positions
    x_tick_labels = ['10%', '15%', '33%']

    for ax, dataset in zip(axes, DATASETS):
        for lyr, style in LAYER_STYLES.items():
            if dataset not in data or lyr not in data[dataset]:
                print(f"  Warning: no data for {dataset} / {lyr}")
                continue

            y_vals = [data[dataset][lyr].get(r, float('nan')) for r in RATES]

            ax.plot(
                x_ticks, y_vals,       # ← use positions not actual rates
                label     = style['label'],
                color     = style['color'],
                marker    = style['marker'],
                linestyle = style['ls'],
                linewidth = 1.6,
                markersize= 6,
            )

        ax.set_title(dataset, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel('Poison Rate $p$', fontsize=9)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_xlim(-0.3, 2.3)         # ← adjusted for new range

    axes[0].set_ylabel('AC F1 Score (%)', fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc           = 'lower center',
        ncol          = 3,
        fontsize      = 9,
        frameon       = True,
        bbox_to_anchor= (0.5, -0.16),
        markerscale   = 1.25,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_pdf = os.path.join(output_dir, 'multilayer_ac_f1.pdf')
    out_png = os.path.join(output_dir, 'multilayer_ac_f1.png')

    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=CSV_PATH,
                        help='Path to pipe-delimited results CSV')
    parser.add_argument('--out', default=OUTPUT_DIR,
                        help='Output directory for figures')
    args = parser.parse_args()

    print(f"Loading results from: {args.csv}")
    data = load_results(args.csv)

    print("Datasets found:", list(data.keys()))
    for ds, layers in data.items():
        for lyr, rates in layers.items():
            print(f"  {ds} / {lyr}: {rates}")

    plot(data, args.out)
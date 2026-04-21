"""
visualize_3d.py — Interactive 3D visualisation of fc1 activation clusters.

Loads the trained backdoor model and cached poisoned dataset produced by
pipeline.py, re-extracts fc1 activations, projects them to 3 principal
components, and saves four interactive Plotly HTML files.

Output files (written to results/MNIST_rotating_r0.1_sub0.25_noise0.0_pre0/):
    3d_ground_truth.html    — 3D scatter coloured by true poison label
    3d_clusters.html        — 3D scatter coloured by k-means cluster
    2d_overview.html        — 2×5 grid of 2D PCA projections, all classes
    detection_landscape.html — silhouette vs size_ratio for all classes

Usage:
    pip install plotly          # only new dependency
    python visualize_3d.py

Run from the project root (same directory as pipeline.py).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from data.builder               import MixedDataset
from data.loader                import load_dataset
from models.cnn                 import PaperCNN
from models.train               import load_model
from clustering.extractor  import extract_activations
from clustering.clustering import cluster_all_classes
from clustering.analyzer   import analyze_all_classes


# ── Colour palette (matches plots.py) ────────────────────────────────────────
COL = dict(
    clean    = '#2980b9',   # blue   — genuine samples (GT)
    poison   = '#e74c3c',   # red    — poisoned samples (GT)
    cluster_a = '#e67e22',  # orange — k-means cluster A (larger)
    cluster_b = '#8e44ad',  # purple — k-means cluster B (suspect)
    boundary  = '#7f8c8d',  # grey   — threshold lines
    flagged   = '#e74c3c',
    ok        = '#27ae60',
)


# ── Step 1: Load artifacts ────────────────────────────────────────────────────

def load_artifacts():
    """Load the cached MixedDataset and trained backdoor model."""
    print("Loading cached dataset …")
    mixed = MixedDataset.load(C.CACHE_DATASET_PATH)

    print("Loading trained model …")
    dataset_info = load_dataset(C.DATASET_NAME, data_dir=C.DATASETS_DIR)
    model = PaperCNN.for_dataset(dataset_info)
    load_model(model, C.BACKDOOR_MODEL_PATH, C.DEVICE)

    return mixed, model


# ── Step 2: Extract and cluster ──────────────────────────────────────────────

def extract_and_cluster(mixed, model):
    """Re-run Steps 5–7 of the pipeline (fast — no training)."""
    print("Extracting fc1 activations …")
    extraction = extract_activations(
        model      = model,
        dataset    = mixed,
        layer_names= C.AC_LAYERS,
        device     = C.DEVICE,
    )

    print("Clustering …")
    cluster_map = cluster_all_classes(
        extraction   = extraction,
        n_components = C.AC_N_COMPONENTS,
        method       = C.AC_METHOD,
        seed         = C.SEED,
    )

    print("Analysing …")
    analysis = analyze_all_classes(
        extraction  = extraction,
        cluster_map = cluster_map,
        cfg         = C.ANALYSIS_CFG,
        device      = C.DEVICE,
    )

    return extraction, cluster_map, analysis


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pca3(feats: np.ndarray, seed: int = 42) -> tuple[np.ndarray, list[float]]:
    """Normalise and reduce (N, D) → (N, 3) via PCA. Returns (pts, var_exp)."""
    fn     = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    n_comp = min(3, fn.shape[1], fn.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=seed)
    pts    = pca.fit_transform(fn).astype(np.float32)
    if pts.shape[1] < 3:                               # pad if n_comp < 3
        pts = np.hstack([pts, np.zeros((len(pts), 3 - pts.shape[1]))])
    var    = pca.explained_variance_ratio_.tolist()
    return pts, var


def _pca2(feats: np.ndarray, seed: int = 42) -> tuple[np.ndarray, list[float]]:
    """Normalise and reduce (N, D) → (N, 2) via PCA."""
    fn     = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    n_comp = min(2, fn.shape[1], fn.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=seed)
    pts    = pca.fit_transform(fn).astype(np.float32)
    var    = pca.explained_variance_ratio_.tolist()
    return pts, var


def _scene_style():
    return dict(
        bgcolor       = 'rgb(245,246,250)',
        xaxis = dict(gridcolor='#d0d4de', zerolinecolor='#c0c4ce', color='#606880'),
        yaxis = dict(gridcolor='#d0d4de', zerolinecolor='#c0c4ce', color='#606880'),
        zaxis = dict(gridcolor='#d0d4de', zerolinecolor='#c0c4ce', color='#606880'),
    )


def _layout_base(title: str = '') -> dict:
    return dict(
        paper_bgcolor = 'white',
        plot_bgcolor  = 'white',
        title         = dict(text=title, font=dict(size=13), x=0.05),
        font          = dict(family='monospace', size=11),
        legend        = dict(bgcolor='rgba(255,255,255,0.85)',
                             bordercolor='#ddd', borderwidth=1),
        margin        = dict(l=0, r=0, t=40, b=0),
    )


# ── Figure 1 & 2: 3D scatter per class (one per colour mode) ─────────────────

def build_3d_scatter(extraction, cluster_map, analysis, color_by: str) -> go.Figure:
    """
    One figure with a class-selector button bar.
    color_by: 'gt'       → ground truth (blue clean / red poisoned)
              'clusters' → k-means labels (orange A / purple B)
    """
    classes  = sorted(extraction.activations.keys())
    n_cls    = len(classes)
    n_traces = n_cls * 2          # 2 traces per class

    traces  : list[go.Scatter3d] = []
    buttons : list[dict]         = []

    for cls in classes:
        feats = extraction.activations[cls]
        gt    = extraction.flags[cls]
        cr    = cluster_map[cls]
        ar    = analysis[cls]
        pts, var = _pca3(feats)

        if color_by == 'gt':
            groups = [
                dict(mask=~gt,  name='Clean',    color=COL['clean'],     sz=3,   sym='circle',  op=0.45),
                dict(mask=gt,   name='Poisoned', color=COL['poison'],    sz=5,   sym='diamond', op=0.90),
            ]
        else:
            larger  = cr.larger_cluster
            smaller = cr.smaller_cluster
            groups = [
                dict(mask=cr.km_labels == larger,  name=f'Cluster A (n={cr.cluster_sizes[larger]})',
                     color=COL['cluster_a'], sz=3, sym='circle',  op=0.45),
                dict(mask=cr.km_labels == smaller, name=f'Cluster B — suspect (n={cr.cluster_sizes[smaller]})',
                     color=COL['cluster_b'], sz=5, sym='diamond', op=0.90),
            ]

        verdict = '⚠ FLAGGED' if ar.is_poisoned else '✓ clean'
        title_str = (
            f'Class {cls}  |  {verdict}  |  '
            f'sil={cr.silhouette:.3f}  size_ratio={cr.size_ratio:.3f}<br>'
            f'<sub>PC variance: {var[0]:.1%}+{var[1]:.1%}+{var[2]:.1%} '
            f'| n_poison={gt.sum()} / {len(gt)}</sub>'
        )

        for g in groups:
            idx = pts[g['mask']]
            traces.append(go.Scatter3d(
                x=idx[:, 0].tolist(), y=idx[:, 1].tolist(), z=idx[:, 2].tolist(),
                mode='markers',
                name=g['name'],
                marker=dict(size=g['sz'], color=g['color'], opacity=g['op'],
                            symbol=g['sym'], line=dict(width=0)),
                hovertemplate=f"<b>{g['name']}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>PC3: %{{z:.3f}}<extra></extra>",
                visible=(cls == 0),
            ))

        # Button: show only this class's 2 traces
        vis = [False] * n_traces
        vis[cls * 2]     = True
        vis[cls * 2 + 1] = True
        buttons.append(dict(
            label  = f'Class {cls}',
            method = 'update',
            args   = [{'visible': vis},
                      {'title': dict(text=title_str, font=dict(size=13), x=0.05),
                       'scene': dict(_scene_style(),
                                     camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)))}],
        ))

    # Build initial title for class 0
    cls0 = classes[0]
    cr0, ar0 = cluster_map[cls0], analysis[cls0]
    pts0, var0 = _pca3(extraction.activations[cls0])
    gt0 = extraction.flags[cls0]
    verdict0 = '⚠ FLAGGED' if ar0.is_poisoned else '✓ clean'
    init_title = (
        f'Class {cls0}  |  {verdict0}  |  '
        f'sil={cr0.silhouette:.3f}  size_ratio={cr0.size_ratio:.3f}<br>'
        f'<sub>PC variance: {var0[0]:.1%}+{var0[1]:.1%}+{var0[2]:.1%} '
        f'| n_poison={gt0.sum()} / {len(gt0)}</sub>'
    )

    layout = go.Layout(
        **_layout_base(init_title),
        scene       = dict(_scene_style(),
                           xaxis_title='PC 1', yaxis_title='PC 2', zaxis_title='PC 3',
                           camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                           aspectmode='cube'),
        updatemenus = [dict(
            type       = 'buttons',
            direction  = 'right',
            x=0.0, y=1.10, xanchor='left',
            buttons    = buttons,
            showactive = True,
            bgcolor    = 'white',
            bordercolor= '#ccc',
            font       = dict(size=11, family='monospace'),
        )],
        height = 680,
    )

    return go.Figure(data=traces, layout=layout)


# ── Figure 3: 2D overview grid ────────────────────────────────────────────────

def build_2d_overview(extraction, cluster_map, analysis) -> go.Figure:
    """
    2×5 grid of 2D PCA projections — one panel per class.
    Row 1: ground truth.  Row 2: k-means cluster assignment.
    Matches the layout of plots.py but is interactive.
    """
    classes = sorted(extraction.activations.keys())
    n_cls   = len(classes)
    rows, cols = 2, n_cls

    subplot_titles = (
        [f'class {c}' for c in classes] +
        [f'A={cluster_map[c].cluster_sizes[cluster_map[c].larger_cluster]}  '
         f'B={cluster_map[c].cluster_sizes[cluster_map[c].smaller_cluster]}  '
         f'(ratio={cluster_map[c].size_ratio:.2f})'
         for c in classes]
    )

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.12,
    )

    show_legend_gt  = {True: True, False: True}    # track first appearance
    show_legend_km  = {True: True, False: True}

    for col_idx, cls in enumerate(classes):
        col = col_idx + 1
        feats = extraction.activations[cls]
        gt    = extraction.flags[cls]
        cr    = cluster_map[cls]
        pts, _ = _pca2(feats)

        # ── Row 1: ground truth ───────────────────────────────────────────
        for mask, name, color, sz in [
            (~gt, 'Clean',    COL['clean'],  4),
            ( gt, 'Poisoned', COL['poison'], 7),
        ]:
            show = show_legend_gt[name == 'Clean'] if col_idx == 0 else False
            if col_idx == 0:
                show_legend_gt[name == 'Clean'] = False
            fig.add_trace(go.Scatter(
                x=pts[mask, 0].tolist(), y=pts[mask, 1].tolist(),
                mode='markers',
                name=name,
                marker=dict(size=sz, color=color,
                            opacity=0.5 if name == 'Clean' else 0.85),
                legendgroup=name,
                showlegend=(col_idx == 0),
                hovertemplate=f'<b>{name}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra>cls {cls}</extra>',
            ), row=1, col=col)

        # ── Row 2: k-means ────────────────────────────────────────────────
        larger, smaller = cr.larger_cluster, cr.smaller_cluster
        for km_val, name, color, sz in [
            (larger,  'Cluster A (larger)',  COL['cluster_a'], 4),
            (smaller, 'Cluster B (suspect)', COL['cluster_b'], 7),
        ]:
            fig.add_trace(go.Scatter(
                x=pts[cr.km_labels == km_val, 0].tolist(),
                y=pts[cr.km_labels == km_val, 1].tolist(),
                mode='markers',
                name=name,
                marker=dict(size=sz, color=color,
                            opacity=0.5 if km_val == larger else 0.85),
                legendgroup=name,
                showlegend=(col_idx == 0),
                hovertemplate=f'<b>{name}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra>cls {cls}</extra>',
            ), row=2, col=col)

    # Hide all axis labels to keep panels compact
    for r in (1, 2):
        for c in range(1, n_cls + 1):
            key = f'xaxis{(r-1)*n_cls + c}' if not (r == 1 and c == 1) else 'xaxis'
            fig.update_layout(**{key: dict(showticklabels=False, showgrid=False, zeroline=False)})
            key2 = f'yaxis{(r-1)*n_cls + c}' if not (r == 1 and c == 1) else 'yaxis'
            fig.update_layout(**{key2: dict(showticklabels=False, showgrid=False, zeroline=False)})

    # Row labels via annotations
    fig.add_annotation(text='<b>Ground truth</b>', x=-0.01, y=0.78, xref='paper', yref='paper',
                       showarrow=False, font=dict(size=12), textangle=-90)
    fig.add_annotation(text='<b>K-Means (AC view)</b>', x=-0.01, y=0.22, xref='paper', yref='paper',
                       showarrow=False, font=dict(size=12), textangle=-90)

    base = _layout_base(
        'Activation Clustering — All Classes · 2D PCA projection'
        '<br><sub>Row 1: ground truth (blue=clean, red=poisoned) · '
        'Row 2: k-means (orange=A, purple=B suspect)</sub>'
    )
    base['height'] = 520
    base['legend'] = dict(
        orientation='h', x=0.5, y=-0.04, xanchor='center',
        bgcolor='rgba(255,255,255,0.85)', bordercolor='#ddd', borderwidth=1,
        font=dict(size=11, family='monospace'),
    )
    fig.update_layout(**base)

    return fig


# ── Figure 4: Detection landscape ────────────────────────────────────────────

def build_detection_landscape(cluster_map, analysis, poison_rate: float) -> go.Figure:
    """
    Silhouette score vs size_ratio scatter — one point per class.
    Shaded quadrant = both thresholds exceeded (AC flags as poisoned).
    """
    classes = sorted(cluster_map.keys())
    x = [cluster_map[c].size_ratio   for c in classes]
    y = [cluster_map[c].silhouette    for c in classes]
    labels   = [f'Class {c}'          for c in classes]
    flagged  = [analysis[c].is_poisoned for c in classes]
    n_poison = [analysis[c].nPoison if hasattr(analysis[c], 'nPoison') else
                len(analysis[c].predicted_flags[analysis[c].predicted_flags])
                for c in classes]

    colors  = [COL['poison'] if f else COL['ok']   for f in flagged]
    symbols = ['diamond'     if f else 'circle'    for f in flagged]
    texts   = [
        f'<b>Class {c}</b><br>'
        f'Silhouette: {cluster_map[c].silhouette:.4f}<br>'
        f'Size ratio: {cluster_map[c].size_ratio:.4f}<br>'
        f'Flagged: {"yes" if analysis[c].is_poisoned else "no"}'
        for c in classes
    ]

    threshold_sil  = 0.10
    threshold_size = poison_rate + 0.05    # matches ANALYSIS_CFG.max_poison_rate

    fig = go.Figure()

    # Shaded "flagged" region
    fig.add_shape(type='rect',
                  x0=0, x1=threshold_size, y0=threshold_sil, y1=1.05,
                  fillcolor='rgba(231,76,60,0.07)', line=dict(width=0), layer='below')

    # Threshold lines
    fig.add_shape(type='line', x0=threshold_size, x1=threshold_size, y0=0, y1=1.05,
                  line=dict(color=COL['boundary'], dash='dash', width=1.2))
    fig.add_shape(type='line', x0=0, x1=0.6, y0=threshold_sil, y1=threshold_sil,
                  line=dict(color=COL['boundary'], dash='dash', width=1.2))

    # Annotations for threshold labels
    fig.add_annotation(x=threshold_size, y=0.02,
                       text=f'max_poison_rate = {threshold_size:.2f}',
                       showarrow=False, font=dict(size=10, color=COL['boundary']),
                       xanchor='left', xshift=4)
    fig.add_annotation(x=0.52, y=threshold_sil,
                       text=f'sil threshold = {threshold_sil}',
                       showarrow=False, font=dict(size=10, color=COL['boundary']),
                       yshift=8)
    fig.add_annotation(x=threshold_size / 2, y=0.98,
                       text='FLAGGED ZONE', showarrow=False,
                       font=dict(size=10, color=COL['poison']))

    # Data points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+text',
        text=[str(c) for c in classes],
        textposition='top center',
        textfont=dict(size=11, family='monospace'),
        marker=dict(size=16, color=colors, symbol=symbols, opacity=0.9,
                    line=dict(color='white', width=1.5)),
        customdata=list(zip(labels, [f"{v:.4f}" for v in y],
                            [f"{v:.4f}" for v in x], flagged)),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'Silhouette: %{customdata[1]}<br>'
            'Size ratio: %{customdata[2]}<br>'
            'Flagged: %{customdata[3]}<extra></extra>'
        ),
        showlegend=False,
    ))

    # Manual legend
    for name, color, sym in [
        ('Flagged as poisoned', COL['poison'], 'diamond'),
        ('Not flagged (missed)', COL['ok'],    'circle'),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            name=name,
            marker=dict(size=10, color=color, symbol=sym),
        ))

    base = _layout_base(
        'Detection landscape — silhouette score vs cluster size ratio'
        '<br><sub>Both thresholds must be exceeded to flag a class as poisoned</sub>'
    )
    base['margin'] = dict(l=60, r=20, t=70, b=60)
    base['height'] = 520
    base['legend'] = dict(x=0.98, y=0.05, xanchor='right',
                          bgcolor='rgba(255,255,255,0.9)', bordercolor='#ddd', borderwidth=1)
    fig.update_layout(
        **base,
        xaxis=dict(title='Size ratio (smaller cluster / total)', range=[0, 0.6],
                   gridcolor='#eee', zerolinewidth=0),
        yaxis=dict(title='Silhouette score', range=[0, 1.05],
                   gridcolor='#eee', zerolinewidth=0),
    )

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    # Load
    mixed, model = load_artifacts()

    # Extract + cluster + analyse
    extraction, cluster_map, analysis = extract_and_cluster(mixed, model)

    print("\nBuilding figures …")

    # Figure 1 — 3D, ground truth coloring
    fig1 = build_3d_scatter(extraction, cluster_map, analysis, color_by='gt')
    path1 = os.path.join(C.RESULTS_DIR, '3d_ground_truth.html')
    fig1.write_html(path1, include_plotlyjs='cdn')
    print(f"  Saved → {path1}")

    # Figure 2 — 3D, cluster label coloring
    fig2 = build_3d_scatter(extraction, cluster_map, analysis, color_by='clusters')
    path2 = os.path.join(C.RESULTS_DIR, '3d_clusters.html')
    fig2.write_html(path2, include_plotlyjs='cdn')
    print(f"  Saved → {path2}")

    # Figure 3 — 2D overview grid, all classes
    fig3 = build_2d_overview(extraction, cluster_map, analysis)
    path3 = os.path.join(C.RESULTS_DIR, '2d_overview.html')
    fig3.write_html(path3, include_plotlyjs='cdn')
    print(f"  Saved → {path3}")

    # Figure 4 — Detection landscape
    fig4 = build_detection_landscape(cluster_map, analysis,
                                     poison_rate=C.POISON_CFG.poison_rate)
    path4 = os.path.join(C.RESULTS_DIR, 'detection_landscape.html')
    fig4.write_html(path4, include_plotlyjs='cdn')
    print(f"  Saved → {path4}")

    print(f"\nDone. Open any .html file in a browser for interactive plots.")


if __name__ == '__main__':
    main()
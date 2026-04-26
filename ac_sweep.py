"""
ac_sweep.py — AC clustering and evaluation sweep over n_components values.

Encapsulates steps 6–9 (cluster, analyse, evaluate, visualise) so that
pipeline.py stays free of loop logic. Results for each k are saved to
{base_results_dir}/n_components_{k}/ when multiple values are given, or
directly to base_results_dir when only one value is provided.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import config as C
from clustering           import cluster_all_classes, analyze_all_classes
from evaluate             import evaluate_detection, print_combined_table, EvalResult
from visualization        import (
    plot_activation_scatter,
    plot_silhouette_bars,
    plot_reconstructed_samples,
)


def run_ac_sweep(
    ac_extraction,
    raw_extraction,
    n_components_list: List[int],
    base_results_dir:  str,
    mixed_dataset,
    dataset_info,
) -> dict[int, Tuple[EvalResult, EvalResult]]:
    """
    Run steps 6–9 for each value in n_components_list.

    Args:
        ac_extraction:     activation extraction result (from step 5)
        raw_extraction:    raw-pixel extraction result (from step 5)
        n_components_list: list of ICA/PCA component counts to sweep
        base_results_dir:  root results directory for this experiment
        mixed_dataset:     poisoned training dataset (for visualisation)
        dataset_info:      dataset metadata (for visualisation)

    Returns:
        dict mapping each k -> (ac_result, raw_result)
    """
    use_subdirs = len(n_components_list) > 1
    all_results: dict[int, Tuple[EvalResult, EvalResult]] = {}

    for k in n_components_list:
        print(f"\n{'=' * 60}")
        print(f"  AC evaluation  (n_components={k})")
        print(f"{'=' * 60}")

        results_dir = (
            os.path.join(base_results_dir, f"n_components_{k}")
            if use_subdirs else base_results_dir
        )
        os.makedirs(results_dir, exist_ok=True)

        # ── Step 6: Cluster ───────────────────────────────────────────────
        print("\n── Step 6: Cluster activations ──")
        ac_cluster_map = cluster_all_classes(
            extraction   = ac_extraction,
            n_components = k,
            method       = C.AC_METHOD,
            seed         = C.SEED,
        )
        raw_cluster_map = cluster_all_classes(
            extraction   = raw_extraction,
            n_components = k,
            method       = C.AC_METHOD,
            seed         = C.SEED,
        )

        # ── Step 7: Analyse ───────────────────────────────────────────────
        print("\n── Step 7: Analyse clusters ──")
        ac_analysis = analyze_all_classes(
            extraction  = ac_extraction,
            cluster_map = ac_cluster_map,
            cfg         = C.ANALYSIS_CFG,
            device      = C.DEVICE,
            label       = 'AC Method',
        )
        raw_analysis = analyze_all_classes(
            extraction  = raw_extraction,
            cluster_map = raw_cluster_map,
            cfg         = C.ANALYSIS_CFG,
            device      = C.DEVICE,
            label       = 'RAW Method',
        )

        # ── Step 8: Evaluate ──────────────────────────────────────────────
        print("\n── Step 8: Evaluate detection ──")
        ac_result = evaluate_detection(
            extraction  = ac_extraction,
            analysis    = ac_analysis,
            cluster_map = ac_cluster_map,
        )
        raw_result = evaluate_detection(
            extraction  = raw_extraction,
            analysis    = raw_analysis,
            cluster_map = raw_cluster_map,
        )
        print_combined_table(
            ac_result      = ac_result,
            raw_result     = raw_result,
            ac_analysis    = ac_analysis,
            ac_cluster_map = ac_cluster_map,
            poison_rate    = C.POISON_CFG.poison_rate,
            method         = C.AC_METHOD,
        )
        ac_result.save(os.path.join(results_dir, "ac_detection_results.json"))
        raw_result.save(os.path.join(results_dir, "raw_detection_results.json"))

        import json as _json
        sil_data = {
            "mean_silhouette": float(
                sum(r.silhouette_kd for r in ac_analysis.values()) / len(ac_analysis)
            ),
            "per_class": {
                str(cls): {
                    "silhouette":    r.silhouette,
                    "silhouette_kd": r.silhouette_kd,
                    "silhouette_2d": r.silhouette_2d,
                }
                for cls, r in ac_analysis.items()
            },
        }
        sil_path = os.path.join(results_dir, "silhouette_results.json")
        with open(sil_path, "w") as _f:
            _json.dump(sil_data, _f, indent=2)
        print(f"  Silhouette scores saved → {sil_path}")

        # ── Step 9: Visualise ─────────────────────────────────────────────
        print("\n── Step 9: Visualise ──")
        plot_activation_scatter(
            extraction  = ac_extraction,
            cluster_map = ac_cluster_map,
            results_dir = results_dir,
            seed        = C.SEED,
            save        = True,
            show        = C.SHOW_PLOTS,
        )
        plot_silhouette_bars(
            analysis    = ac_analysis,
            results_dir = results_dir,
            save        = True,
            show        = C.SHOW_PLOTS,
        )
        plot_reconstructed_samples(
            mixed_dataset = mixed_dataset,
            dataset_info  = dataset_info,
            results_dir   = results_dir,
            n_per_pair    = 2,
            save          = True,
            show          = C.SHOW_PLOTS,
        )

        all_results[k] = (ac_result, raw_result)

    return all_results

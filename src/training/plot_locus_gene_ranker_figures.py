#!/usr/bin/env python3
"""Generate PNG visualizations for the locus-to-gene ranking proof-of-concept."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/locus_gene_ranker_poc"
)
DEFAULT_FIG_DIR = Path("/home/viguinijpv/200.18.99.75:8000/IC/results/figures")
TEXT_SCORE_CANDIDATE_COLUMNS = [
    "text_score",
    "gene_text_score",
    "text_similarity_score",
    "disease_text_score",
    "literature_score",
]
FAMILY_COLOR_MAP = {
    "distance": "#4e79a7",
    "coloc_qtl": "#f28e2b",
    "abundance": "#e15759",
    "network": "#76b7b2",
    "embedding_raw": "#edc948",
    "embedding_pca": "#59a14f",
    "other": "#9ca3af",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create interpretation figures for locus-gene ranker outputs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing mode_none/mode_full/mode_pca outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help="Directory to save PNG figures and figure_index.md",
    )
    parser.add_argument(
        "--ranking-mode",
        choices=["none", "full", "pca"],
        default="full",
        help="Mode used for per-locus ranking and biological inspection plots.",
    )
    parser.add_argument(
        "--top-k-label-annotations",
        type=int,
        default=5,
        help="Number of top-ranked genes to annotate in per-locus ranking plots.",
    )
    parser.add_argument(
        "--top-false-positive-n",
        type=int,
        default=15,
        help="Maximum number of false positives to plot after within-locus rank filtering.",
    )
    parser.add_argument(
        "--false-positive-rank-threshold",
        type=int,
        default=3,
        help="Keep false positives with rank_within_locus <= K before optional top-N truncation.",
    )
    parser.add_argument(
        "--top-contribution-features",
        type=int,
        default=20,
        help="Top-N features in contribution bar plots.",
    )
    parser.add_argument(
        "--decomposition-top-positives",
        type=int,
        default=3,
        help="Number of top true positives to include in decomposition plot.",
    )
    parser.add_argument(
        "--decomposition-top-false-positives",
        type=int,
        default=3,
        help="Number of top false positives (within rank threshold) to include in decomposition plot.",
    )
    parser.add_argument(
        "--include-raw-embedding-features-in-top-plot",
        action="store_true",
        help="Include raw embedding dimensions in top-feature contribution plot.",
    )
    return parser.parse_args()


def _load_pca_transformer_artifacts(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _as_str_list(arr: object) -> List[str]:
    if arr is None:
        return []
    values = np.asarray(arr)
    if values.ndim == 0:
        return [str(values.item())]
    return [str(x) for x in values.tolist()]


def _reconstruct_pca_features_from_raw_embeddings(
    pred_df: pd.DataFrame,
    pca_artifacts: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, Optional[str]]:
    required = [
        "embedding_feature_names",
        "pca_feature_names",
        "emb_scaler_mean",
        "emb_scaler_scale",
        "pca_components",
        "pca_mean",
    ]
    missing_keys = [k for k in required if k not in pca_artifacts]
    if missing_keys:
        return pd.DataFrame(index=pred_df.index), f"missing PCA artifact keys: {', '.join(missing_keys)}"

    embedding_feature_names = _as_str_list(pca_artifacts.get("embedding_feature_names"))
    pca_feature_names = _as_str_list(pca_artifacts.get("pca_feature_names"))
    if not embedding_feature_names or not pca_feature_names:
        return pd.DataFrame(index=pred_df.index), "empty embedding/pca feature names in PCA artifacts"

    missing_raw = [c for c in embedding_feature_names if c not in pred_df.columns]
    if missing_raw:
        preview = ", ".join(missing_raw[:5])
        return (
            pd.DataFrame(index=pred_df.index),
            f"missing raw embedding columns in all_ranked_predictions.csv (e.g. {preview})",
        )

    x_raw = pred_df[embedding_feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scaler_mean = np.asarray(pca_artifacts.get("emb_scaler_mean"), dtype=float).reshape(-1)
    scaler_scale = np.asarray(pca_artifacts.get("emb_scaler_scale"), dtype=float).reshape(-1)
    pca_components = np.asarray(pca_artifacts.get("pca_components"), dtype=float)
    pca_mean = np.asarray(pca_artifacts.get("pca_mean"), dtype=float).reshape(-1)

    n_embed = x_raw.shape[1]
    if scaler_mean.size != n_embed or scaler_scale.size != n_embed:
        return pd.DataFrame(index=pred_df.index), "embedding scaler stats shape mismatch"
    if pca_components.ndim != 2 or pca_components.shape[1] != n_embed:
        return pd.DataFrame(index=pred_df.index), "PCA components shape mismatch"
    if pca_mean.size != n_embed:
        return pd.DataFrame(index=pred_df.index), "PCA mean shape mismatch"
    if pca_components.shape[0] != len(pca_feature_names):
        return pd.DataFrame(index=pred_df.index), "PCA component count does not match pca_feature_names"

    valid_scale = np.isfinite(scaler_scale) & (np.abs(scaler_scale) > 0.0)
    x_scaled = np.zeros_like(x_raw, dtype=float)
    if np.any(valid_scale):
        x_scaled[:, valid_scale] = (x_raw[:, valid_scale] - scaler_mean[valid_scale]) / scaler_scale[valid_scale]

    x_centered = x_scaled - pca_mean
    x_pca = x_centered @ pca_components.T
    pca_df = pd.DataFrame(x_pca, index=pred_df.index, columns=pca_feature_names)
    return pca_df, None


def load_mode_tables(results_dir: Path, mode: str) -> Dict[str, object]:
    mode_dir = results_dir / f"mode_{mode}"
    if not mode_dir.exists():
        raise FileNotFoundError(f"Mode output directory not found: {mode_dir}")

    scaler_stats_path = mode_dir / "scaler_feature_stats.csv"
    if scaler_stats_path.exists():
        scaler_stats_df = pd.read_csv(scaler_stats_path)
    else:
        scaler_stats_df = pd.DataFrame(
            columns=[
                "validation_mode",
                "embedding_mode",
                "mode",
                "cv_mode",
                "feature",
                "scaler_mean",
                "scaler_scale",
                "scaler_var",
            ]
        )

    model_params_path = mode_dir / "model_parameters.json"
    if model_params_path.exists():
        with open(model_params_path, "r", encoding="utf-8") as f:
            model_params = json.load(f)
    else:
        model_params = {}
    pca_artifacts = _load_pca_transformer_artifacts(mode_dir / "pca_transformer_artifacts.npz")

    return {
        "summary": pd.read_json(mode_dir / "summary_metrics.json", typ="series").to_frame().T,
        "all_predictions": pd.read_csv(mode_dir / "all_ranked_predictions.csv"),
        "fold_metrics": pd.read_csv(mode_dir / "fold_metrics.csv"),
        "positive_ranks": pd.read_csv(mode_dir / "positive_gene_ranks.csv"),
        "coefficients": pd.read_csv(mode_dir / "coefficient_table.csv"),
        "scaler_stats": scaler_stats_df,
        "model_params": model_params,
        "pca_artifacts": pca_artifacts,
        "pca_evr": pd.read_csv(mode_dir / "pca_explained_variance.csv"),
        "false_positives": pd.read_csv(mode_dir / "high_rank_false_positives.csv"),
    }


def load_l1_benchmark_table(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "l1_benchmark_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def metric_comparison_plots(summary_df: pd.DataFrame, out_dir: Path) -> List[Tuple[str, str]]:
    metric_map = [
        ("mean_fold_pr_auc", "PR-AUC", "metric_comparison_pr_auc.png"),
        ("mean_fold_roc_auc", "ROC-AUC", "metric_comparison_roc_auc.png"),
        ("mean_recall_at_1", "Recall@1", "metric_comparison_recall_at1.png"),
        ("mean_recall_at_3", "Recall@3", "metric_comparison_recall_at3.png"),
        ("mean_mrr", "MRR", "metric_comparison_mrr.png"),
    ]
    mode_label = {
        "none": "Baseline",
        "full": "Baseline + Emb (Full)",
        "pca": "Baseline + Emb (PCA)",
    }
    generated: List[Tuple[str, str]] = []

    ordered = (
        summary_df.set_index("mode")
        .reindex(["none", "full", "pca"])
        .reset_index()
    )
    labels = [mode_label.get(m, m) for m in ordered["mode"].tolist()]

    for key, ylabel, filename in metric_map:
        if key not in ordered.columns:
            continue
        vals = ordered[key].astype(float).to_numpy()
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        bars = ax.bar(labels, vals, color=["#4e79a7", "#f28e2b", "#59a14f"])
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(f"Model Comparison: {ylabel}")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, min(1.05, max(1.0, np.nanmax(vals) + 0.12)))
        ax.grid(axis="y", alpha=0.25)
        out_path = out_dir / filename
        save_figure(fig, out_path)
        generated.append((filename, f"Bar plot comparing {ylabel} across baseline/full/PCA models."))
    return generated


def l1_benchmark_plots(
    bench_df: pd.DataFrame,
    out_dir: Path,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    generated: List[Tuple[str, str]] = []
    skipped: List[str] = []
    if bench_df.empty:
        skipped.append("L1 benchmark plots skipped: l1_benchmark_summary.csv not found.")
        return generated, skipped

    required = {
        "C",
        "mean_fold_pr_auc",
        "mean_fold_roc_auc",
        "mean_recall_at_1",
        "mean_mrr",
        "non_zero_total_coefficients",
        "non_zero_pca_coefficients",
        "selected",
    }
    if not required.issubset(bench_df.columns):
        missing = sorted(required.difference(set(bench_df.columns)))
        skipped.append(
            f"L1 benchmark plots skipped: missing columns in l1_benchmark_summary.csv: {missing}"
        )
        return generated, skipped

    dd = bench_df.copy()
    for col in [
        "C",
        "mean_fold_pr_auc",
        "mean_fold_roc_auc",
        "mean_recall_at_1",
        "mean_mrr",
        "non_zero_total_coefficients",
        "non_zero_pca_coefficients",
        "selected",
    ]:
        dd[col] = pd.to_numeric(dd[col], errors="coerce")
    dd = dd.dropna(subset=["C"]).sort_values("C", ascending=True, kind="stable").reset_index(drop=True)
    if dd.empty:
        skipped.append("L1 benchmark plots skipped: all C values invalid.")
        return generated, skipped

    sel = dd.loc[dd["selected"].fillna(0).astype(int) == 1].copy()
    selected_c = float(sel.iloc[0]["C"]) if not sel.empty else None

    metric_specs = [
        ("mean_fold_pr_auc", "Mean PR-AUC", "l1_benchmark_pr_auc.png"),
        ("mean_fold_roc_auc", "Mean ROC-AUC", "l1_benchmark_roc_auc.png"),
        ("mean_recall_at_1", "Recall@1", "l1_benchmark_recall_at1.png"),
        ("mean_mrr", "MRR", "l1_benchmark_mrr.png"),
    ]
    for col, ylabel, fname in metric_specs:
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        ax.plot(dd["C"], dd[col], marker="o", color="#4e79a7", linewidth=1.8)
        ax.set_xscale("log")
        ax.set_xlabel("C (inverse regularization strength, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"L1 Benchmark: {ylabel} vs C")
        ax.grid(alpha=0.25)
        if selected_c is not None:
            y_sel = float(sel.iloc[0][col])
            ax.axvline(selected_c, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.9)
            ax.scatter([selected_c], [y_sel], color="#dc2626", marker="*", s=180, zorder=4)
            ax.text(
                selected_c,
                y_sel,
                f" selected C={selected_c:g}",
                fontsize=8,
                color="#991b1b",
                ha="left",
                va="bottom",
            )
        save_figure(fig, out_dir / fname)
        generated.append((fname, f"L1 benchmark curve for {ylabel} across C values."))

    sparsity_specs = [
        ("non_zero_total_coefficients", "Non-zero coefficients (total)", "l1_benchmark_nonzero_total.png"),
        ("non_zero_pca_coefficients", "Non-zero PCA coefficients", "l1_benchmark_nonzero_pca.png"),
    ]
    for col, ylabel, fname in sparsity_specs:
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        ax.plot(dd["C"], dd[col], marker="o", color="#59a14f", linewidth=1.8)
        ax.set_xscale("log")
        ax.set_xlabel("C (inverse regularization strength, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"L1 Benchmark: {ylabel} vs C")
        ax.grid(alpha=0.25)
        if selected_c is not None:
            y_sel = float(sel.iloc[0][col])
            ax.axvline(selected_c, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.9)
            ax.scatter([selected_c], [y_sel], color="#dc2626", marker="*", s=180, zorder=4)
        save_figure(fig, out_dir / fname)
        generated.append((fname, f"L1 benchmark sparsity curve for {ylabel.lower()}."))

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.scatter(
        dd["non_zero_total_coefficients"],
        dd["mean_fold_pr_auc"],
        c=np.log10(dd["C"].astype(float)),
        cmap="viridis",
        s=70,
        alpha=0.9,
    )
    for row in dd.itertuples(index=False):
        ax.text(
            float(row.non_zero_total_coefficients) + 0.25,
            float(row.mean_fold_pr_auc) + 0.002,
            f"C={float(row.C):g}",
            fontsize=7,
            alpha=0.9,
        )
    if selected_c is not None:
        sel_row = sel.iloc[0]
        ax.scatter(
            [float(sel_row["non_zero_total_coefficients"])],
            [float(sel_row["mean_fold_pr_auc"])],
            color="#dc2626",
            marker="*",
            s=220,
            zorder=5,
            label=f"Selected C={selected_c:g}",
        )
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Non-zero coefficients (total)")
    ax.set_ylabel("Mean PR-AUC")
    ax.set_title("L1 Benchmark Tradeoff: Sparsity vs PR-AUC")
    ax.grid(alpha=0.25)
    fname = "l1_benchmark_tradeoff.png"
    save_figure(fig, out_dir / fname)
    generated.append((fname, "Tradeoff between coefficient sparsity and mean PR-AUC across L1 settings."))
    return generated, skipped


def per_locus_ranking_plots(
    pred_df: pd.DataFrame,
    out_dir: Path,
    top_k_annot: int,
) -> List[Tuple[str, str]]:
    generated: List[Tuple[str, str]] = []
    locus_ids = sorted(pred_df["gwas_study_locus_id"].astype(str).unique().tolist())
    subdir = out_dir / "per_locus_rankings"
    subdir.mkdir(parents=True, exist_ok=True)

    for locus in locus_ids:
        d = pred_df[pred_df["gwas_study_locus_id"].astype(str) == locus].copy()
        d = d.sort_values("predicted_score", ascending=False, kind="stable").reset_index(drop=True)
        d["rank"] = np.arange(1, len(d) + 1)
        top_gene = d.iloc[0]
        pos = d[d["label_positive"].astype(int) == 1]
        neg = d[d["label_positive"].astype(int) == 0]

        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        ax.plot(d["rank"], d["predicted_score"], color="#9ca3af", linewidth=1.2, alpha=0.7)
        ax.scatter(neg["rank"], neg["predicted_score"], color="#6b7280", s=18, alpha=0.65, label="Negative")
        if not pos.empty:
            ax.scatter(pos["rank"], pos["predicted_score"], color="#dc2626", s=48, marker="o", label="Positive")
        ax.scatter(
            [top_gene["rank"]],
            [top_gene["predicted_score"]],
            color="#1d4ed8",
            s=120,
            marker="*",
            label="Top-ranked gene",
            zorder=4,
        )

        # annotate top-k and positives
        ann = d.head(top_k_annot)
        for row in ann.itertuples(index=False):
            ax.annotate(
                str(row.gene_symbol),
                (row.rank, row.predicted_score),
                textcoords="offset points",
                xytext=(2, 4),
                fontsize=7,
            )
        for row in pos.itertuples(index=False):
            ax.annotate(
                str(row.gene_symbol),
                (row.rank, row.predicted_score),
                textcoords="offset points",
                xytext=(2, -10),
                fontsize=7,
                color="#b91c1c",
            )

        ax.set_title(f"Locus {locus}: Candidate Gene Ranking")
        ax.set_xlabel("Rank within locus")
        ax.set_ylabel("Predicted score")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        file_name = f"locus_{locus}_ranking.png"
        save_figure(fig, subdir / file_name)
        generated.append((f"per_locus_rankings/{file_name}", f"Ranking curve for locus {locus} with positives and top-ranked gene highlighted."))
    return generated


def positive_rank_distribution_plot(pos_rank_df: pd.DataFrame, out_dir: Path) -> List[Tuple[str, str]]:
    generated: List[Tuple[str, str]] = []
    if pos_rank_df.empty or "rank_within_locus" not in pos_rank_df.columns:
        return generated

    ranks = pos_rank_df["rank_within_locus"].astype(int)
    max_rank = int(max(3, ranks.max()))
    bins = np.arange(0.5, max_rank + 1.5, 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.hist(ranks, bins=bins, color="#4e79a7", edgecolor="black")
    ax.set_xticks(np.arange(1, max_rank + 1))
    ax.set_title("Rank Positions of Positive Genes")
    ax.set_xlabel("Rank within held-out locus")
    ax.set_ylabel("Count of positive genes")
    ax.grid(axis="y", alpha=0.25)
    out_name = "positive_rank_distribution.png"
    save_figure(fig, out_dir / out_name)
    generated.append((out_name, "Histogram of rank positions achieved by true positive genes."))
    return generated


def embedding_ablation_plot(
    pos_none: pd.DataFrame,
    pos_full: pd.DataFrame,
    out_dir: Path,
) -> List[Tuple[str, str]]:
    generated: List[Tuple[str, str]] = []
    if pos_none.empty or pos_full.empty:
        return generated

    none_best = (
        pos_none.groupby("gwas_study_locus_id", as_index=False)["rank_within_locus"]
        .min()
        .rename(columns={"rank_within_locus": "rank_none"})
    )
    full_best = (
        pos_full.groupby("gwas_study_locus_id", as_index=False)["rank_within_locus"]
        .min()
        .rename(columns={"rank_within_locus": "rank_full"})
    )
    merged = none_best.merge(full_best, on="gwas_study_locus_id", how="inner")
    if merged.empty:
        return generated

    merged = merged.sort_values("rank_none", kind="stable").reset_index(drop=True)
    x = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(x, merged["rank_none"], marker="o", label="No embeddings", color="#4e79a7")
    ax.plot(x, merged["rank_full"], marker="o", label="Full embeddings", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels([s[:8] for s in merged["gwas_study_locus_id"]], rotation=45, ha="right", fontsize=8)
    ax.invert_yaxis()
    ax.set_ylabel("Best positive rank (lower is better)")
    ax.set_title("Embedding Ablation per Locus")
    ax.legend()
    ax.grid(alpha=0.25)

    out_name = "embedding_ablation_per_locus.png"
    save_figure(fig, out_dir / out_name)
    generated.append((out_name, "Per-locus comparison of best positive rank without vs with embeddings."))
    return generated


def coefficient_plot(coef_df: pd.DataFrame, title: str, filename: str, out_dir: Path) -> List[Tuple[str, str]]:
    generated: List[Tuple[str, str]] = []
    nz = coef_df[coef_df["non_zero"].astype(int) == 1].copy()
    if nz.empty:
        return generated
    nz = nz.sort_values("abs_coefficient", ascending=True, kind="stable")
    color_map = {
        "baseline": "#4e79a7",
        "abundance": "#e15759",
        "network": "#76b7b2",
        "embedding_raw": "#f28e2b",
        "embedding_pca": "#59a14f",
        "other": "#9ca3af",
    }
    colors = [color_map.get(g, "#9ca3af") for g in nz["feature_group"].astype(str).tolist()]

    fig_h = max(4.0, 0.28 * len(nz))
    fig, ax = plt.subplots(figsize=(8.6, fig_h))
    ax.barh(nz["feature"], nz["coefficient"], color=colors, edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, out_dir / filename)
    generated.append((filename, f"Non-zero model coefficients for {title.lower()}."))
    return generated


def feature_weight_comparison_plot(
    coef_none: pd.DataFrame,
    coef_full: pd.DataFrame,
    coef_pca: pd.DataFrame,
    out_dir: Path,
) -> List[Tuple[str, str]]:
    """Compare absolute weights for shared non-embedding features across model modes."""
    generated: List[Tuple[str, str]] = []

    def _prepare(df: pd.DataFrame, mode_name: str) -> pd.DataFrame:
        dd = df.copy()
        dd["mode"] = mode_name
        return dd

    cat = pd.concat(
        [
            _prepare(coef_none, "none"),
            _prepare(coef_full, "full"),
            _prepare(coef_pca, "pca"),
        ],
        axis=0,
        ignore_index=True,
    )
    if cat.empty:
        return generated

    # Keep only biologically interpretable/shared non-embedding features.
    allowed_groups = {"baseline", "abundance", "network", "other"}
    cat = cat[cat["feature_group"].astype(str).isin(allowed_groups)].copy()
    if cat.empty:
        return generated

    cat["abs_coefficient"] = pd.to_numeric(cat["abs_coefficient"], errors="coerce").fillna(0.0)
    pivot = cat.pivot_table(
        index="feature",
        columns="mode",
        values="abs_coefficient",
        aggfunc="first",
        fill_value=0.0,
    )
    for col in ["none", "full", "pca"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    # Stable ordering by average weight.
    pivot["mean_abs"] = pivot[["none", "full", "pca"]].mean(axis=1)
    pivot = pivot.sort_values("mean_abs", ascending=False, kind="stable").drop(columns=["mean_abs"])
    if pivot.empty:
        return generated

    x = np.arange(len(pivot))
    w = 0.27
    fig_w = max(8.5, 0.55 * len(pivot))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    ax.bar(x - w, pivot["none"].to_numpy(), width=w, label="none", color="#4e79a7")
    ax.bar(x, pivot["full"].to_numpy(), width=w, label="full", color="#f28e2b")
    ax.bar(x + w, pivot["pca"].to_numpy(), width=w, label="pca", color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=45, ha="right")
    ax.set_ylabel("|Coefficient|")
    ax.set_title("Feature Weight Comparison Across Modes (Shared Features)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    out_name = "feature_weight_comparison_shared_features.png"
    save_figure(fig, out_dir / out_name)
    generated.append(
        (
            out_name,
            "Grouped bars comparing absolute coefficient weights of shared non-embedding features across none/full/pca modes.",
        )
    )
    return generated


def infer_feature_family(feature_name: str, fallback_group: Optional[str] = None) -> str:
    f = str(feature_name or "").strip().lower()
    fb = str(fallback_group).strip().lower() if fallback_group is not None else ""

    distance_exact = {"variant_inside_gene", "dist_score_500kb_log"}
    coloc_qtl_exact = {
        "has_qtl_evidence",
        "tissue_count",
        "coloc_score",
        "coloc_record_count",
        "any_trans_qtl",
        "n_trans_qtl",
    }

    if f.startswith("gene_emb_"):
        return "embedding_raw"
    if f.startswith("emb_pca_"):
        return "embedding_pca"
    if f.startswith("network_") or f == "has_network_prop_score":
        return "network"
    if f.startswith("hpa_") or f in {"has_hpa_expression_evidence", "has_expression_evidence"}:
        return "abundance"
    if f.startswith("dist_") or f in distance_exact:
        return "distance"
    if (
        f.startswith("coloc")
        or f.startswith("colocalisation")
        or f.startswith("qtl_")
        or "qtl" in f
        or f in coloc_qtl_exact
    ):
        return "coloc_qtl"

    if fb in {"embedding_raw", "embedding_pca", "network", "abundance", "distance", "coloc_qtl", "other"}:
        return fb
    if fb == "baseline":
        if f.startswith("dist_") or f in distance_exact:
            return "distance"
        return "coloc_qtl"
    return "other"


def _coefficient_column_name(coef_df: pd.DataFrame) -> Optional[str]:
    for cand in ["coefficient", "coef"]:
        if cand in coef_df.columns:
            return cand
    return None


def compute_contribution_tables(
    pred_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    scaler_stats_df: pd.DataFrame,
    model_params: Dict[str, object],
    pca_artifacts: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object], List[str]]:
    skipped: List[str] = []
    tables: Dict[str, pd.DataFrame] = {
        "feature_summary": pd.DataFrame(),
        "family_net_summary": pd.DataFrame(),
        "family_magnitude_summary": pd.DataFrame(),
        "family_raw_summary": pd.DataFrame(),
        "row_decomposition": pd.DataFrame(),
    }
    audit: Dict[str, object] = {
        "old_relative_percent_raw": {},
        "new_relative_percent_net": {},
        "new_relative_percent_magnitude": {},
        "intercept_available": False,
        "intercept_value": None,
        "decision_score_column": None,
        "pca_features_reconstructed": 0,
    }

    coeff_col = _coefficient_column_name(coef_df)
    if coeff_col is None:
        skipped.append(
            "Contribution plots skipped: coefficient_table.csv missing `coefficient` (or `coef`) column."
        )
        return tables, audit, skipped

    cols = ["feature", coeff_col]
    if "feature_group" in coef_df.columns:
        cols.append("feature_group")
    cdf = coef_df[cols].copy()
    cdf["feature"] = cdf["feature"].astype(str)
    cdf["coefficient"] = pd.to_numeric(cdf[coeff_col], errors="coerce")
    cdf = cdf.dropna(subset=["feature", "coefficient"])
    cdf = cdf.drop_duplicates(subset=["feature"], keep="first")

    pred_for_contrib = pred_df
    coeff_features = cdf["feature"].tolist()
    missing_pca_features = [f for f in coeff_features if f.startswith("emb_pca_") and f not in pred_for_contrib.columns]
    if missing_pca_features:
        if pca_artifacts:
            pca_df, pca_err = _reconstruct_pca_features_from_raw_embeddings(pred_for_contrib, pca_artifacts)
            if pca_err is not None:
                skipped.append(f"Contribution analysis: PCA feature reconstruction failed ({pca_err}).")
            else:
                add_cols = [c for c in pca_df.columns if c not in pred_for_contrib.columns]
                if add_cols:
                    pred_for_contrib = pred_for_contrib.join(pca_df[add_cols], how="left")
                    audit["pca_features_reconstructed"] = int(len(add_cols))
        else:
            skipped.append(
                "Contribution analysis: emb_pca_* features missing in all_ranked_predictions.csv and "
                "pca_transformer_artifacts.npz not found."
            )

    present_mask = cdf["feature"].isin(pred_for_contrib.columns)
    cdf_present = cdf.loc[present_mask].copy()
    n_missing_features = int((~present_mask).sum())
    if n_missing_features > 0:
        skipped.append(
            f"Contribution analysis: skipped {n_missing_features} coefficient features not present in all_ranked_predictions.csv."
        )
    if cdf_present.empty:
        skipped.append("Contribution plots skipped: no overlap between coefficient features and prediction columns.")
        return tables, audit, skipped

    cdf_present["feature_family"] = [
        infer_feature_family(f, fallback_group=g)
        for f, g in zip(cdf_present["feature"].tolist(), cdf_present.get("feature_group", pd.Series([None] * len(cdf_present))).tolist())
    ]
    cdf_present = cdf_present.drop_duplicates(subset=["feature"], keep="first").reset_index(drop=True)

    scaler_required = {"feature", "scaler_mean", "scaler_scale"}
    if scaler_stats_df.empty or not scaler_required.issubset(scaler_stats_df.columns):
        skipped.append(
            "Contribution analysis skipped: scaler_feature_stats.csv missing required columns (`feature`, `scaler_mean`, `scaler_scale`)."
        )
        return tables, audit, skipped
    sref = scaler_stats_df[["feature", "scaler_mean", "scaler_scale"]].copy()
    sref["feature"] = sref["feature"].astype(str)
    sref["scaler_mean"] = pd.to_numeric(sref["scaler_mean"], errors="coerce")
    sref["scaler_scale"] = pd.to_numeric(sref["scaler_scale"], errors="coerce")
    sref = sref.dropna(subset=["feature"]).drop_duplicates(subset=["feature"], keep="first")

    scaler_present_mask = cdf_present["feature"].isin(set(sref["feature"].tolist()))
    n_missing_scaler = int((~scaler_present_mask).sum())
    if n_missing_scaler > 0:
        skipped.append(
            f"Contribution analysis: skipped {n_missing_scaler} features missing scaler stats."
        )
    cdf_use = cdf_present.loc[scaler_present_mask].copy().reset_index(drop=True)
    if cdf_use.empty:
        skipped.append("Contribution analysis skipped: no features had both coefficients and scaler stats.")
        return tables, audit, skipped

    features = cdf_use["feature"].tolist()
    x_raw = pred_for_contrib[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    sref_map = sref.set_index("feature")
    means = np.array([float(sref_map.loc[f, "scaler_mean"]) for f in features], dtype=float)
    scales = np.array([float(sref_map.loc[f, "scaler_scale"]) for f in features], dtype=float)
    valid_scale = np.isfinite(scales) & (np.abs(scales) > 0.0)

    z = np.zeros_like(x_raw.to_numpy(dtype=float))
    x_vals = x_raw.to_numpy(dtype=float)
    if np.any(valid_scale):
        z[:, valid_scale] = (x_vals[:, valid_scale] - means[valid_scale]) / scales[valid_scale]
    z_df = pd.DataFrame(z, index=x_raw.index, columns=features)

    coef = cdf_use.set_index("feature")["coefficient"].astype(float)
    feature_family = cdf_use.set_index("feature")["feature_family"].astype(str)

    # Old (incorrect) raw-space contributions kept only for audit/comparison.
    raw_contrib_df = x_raw.mul(coef, axis=1)
    raw_family_df = raw_contrib_df.T.groupby(feature_family).sum().T
    raw_rows: List[Dict[str, object]] = []
    for fam in raw_family_df.columns:
        arr = pd.to_numeric(raw_family_df[fam], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        raw_rows.append(
            {
                "feature_family": fam,
                "mean_abs_raw_contribution": float(np.mean(np.abs(arr))),
                "mean_signed_raw_contribution": float(np.mean(arr)),
                "std_raw_contribution": float(np.std(arr)),
            }
        )
    family_raw_summary_df = pd.DataFrame(raw_rows).sort_values(
        "mean_abs_raw_contribution", ascending=False, kind="stable"
    )
    if not family_raw_summary_df.empty:
        denom = float(family_raw_summary_df["mean_abs_raw_contribution"].sum())
        family_raw_summary_df["relative_percent_raw"] = (
            100.0 * family_raw_summary_df["mean_abs_raw_contribution"] / denom if denom > 0 else 0.0
        )
        audit["old_relative_percent_raw"] = {
            str(r.feature_family): float(r.relative_percent_raw)
            for r in family_raw_summary_df.itertuples(index=False)
        }

    # Correct contributions in standardized model space.
    contrib_df = z_df.mul(coef, axis=1)
    family_net_df = contrib_df.T.groupby(feature_family).sum().T
    family_mag_df = contrib_df.abs().T.groupby(feature_family).sum().T

    feature_rows: List[Dict[str, object]] = []
    for feat in features:
        c = float(coef.loc[feat])
        x = x_raw[feat].to_numpy(dtype=float)
        zz = z_df[feat].to_numpy(dtype=float)
        cc = contrib_df[feat].to_numpy(dtype=float)
        feature_rows.append(
            {
                "feature": feat,
                "feature_family": str(feature_family.loc[feat]),
                "coefficient": c,
                "feature_mean_raw": float(np.mean(x)),
                "feature_std_raw": float(np.std(x)),
                "feature_mean_z": float(np.mean(zz)),
                "feature_std_z": float(np.std(zz)),
                "mean_abs_contribution": float(np.mean(np.abs(cc))),
                "mean_signed_contribution": float(np.mean(cc)),
                "std_contribution": float(np.std(cc)),
            }
        )
    feature_summary_df = pd.DataFrame(feature_rows).sort_values(
        "mean_abs_contribution", ascending=False, kind="stable"
    )

    meta_cols = [
        "gene_symbol",
        "gene_id",
        "gwas_study_locus_id",
        "label_positive",
        "rank_within_locus",
        "predicted_score",
        "fold_index",
        "fold_id",
        "validation_mode",
        "embedding_mode",
        "mode",
        "cv_mode",
    ]
    row_contrib_df = pred_for_contrib[[c for c in meta_cols if c in pred_for_contrib.columns]].copy()
    for fam in family_net_df.columns:
        row_contrib_df[f"contrib_net_{fam}"] = family_net_df[fam].to_numpy(dtype=float)
    for fam in family_mag_df.columns:
        row_contrib_df[f"contrib_magnitude_{fam}"] = family_mag_df[fam].to_numpy(dtype=float)

    intercept_raw = model_params.get("model_intercept", None) if isinstance(model_params, dict) else None
    intercept_val = float(intercept_raw) if intercept_raw is not None else 0.0
    intercept_available = intercept_raw is not None
    audit["intercept_available"] = bool(intercept_available)
    audit["intercept_value"] = float(intercept_val)
    row_contrib_df["intercept"] = float(intercept_val)

    net_cols = [c for c in row_contrib_df.columns if c.startswith("contrib_net_")]
    row_contrib_df["score_reconstructed_net"] = (
        row_contrib_df[net_cols].sum(axis=1) + row_contrib_df["intercept"] if net_cols else row_contrib_df["intercept"]
    )

    decision_score_col = None
    for cand in ["decision_score", "logit_score", "linear_score", "model_score"]:
        if cand in row_contrib_df.columns:
            decision_score_col = cand
            break
    audit["decision_score_column"] = decision_score_col
    if decision_score_col is not None:
        decision = pd.to_numeric(row_contrib_df[decision_score_col], errors="coerce").fillna(0.0)
        row_contrib_df["decision_score_reconstruction_error"] = decision - row_contrib_df["score_reconstructed_net"]

    net_rows: List[Dict[str, object]] = []
    for fam in family_net_df.columns:
        arr = pd.to_numeric(family_net_df[fam], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        net_rows.append(
            {
                "feature_family": str(fam),
                "mean_abs_net_contribution": float(np.mean(np.abs(arr))),
                "mean_signed_net_contribution": float(np.mean(arr)),
                "std_net_contribution": float(np.std(arr)),
            }
        )
    family_net_summary_df = pd.DataFrame(net_rows).sort_values(
        "mean_abs_net_contribution", ascending=False, kind="stable"
    )
    if not family_net_summary_df.empty:
        denom = float(family_net_summary_df["mean_abs_net_contribution"].sum())
        family_net_summary_df["relative_percent_net"] = (
            100.0 * family_net_summary_df["mean_abs_net_contribution"] / denom if denom > 0 else 0.0
        )
        audit["new_relative_percent_net"] = {
            str(r.feature_family): float(r.relative_percent_net)
            for r in family_net_summary_df.itertuples(index=False)
        }

    mag_rows: List[Dict[str, object]] = []
    for fam in family_mag_df.columns:
        arr = pd.to_numeric(family_mag_df[fam], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mag_rows.append(
            {
                "feature_family": str(fam),
                "mean_magnitude_contribution": float(np.mean(arr)),
                "std_magnitude_contribution": float(np.std(arr)),
            }
        )
    family_mag_summary_df = pd.DataFrame(mag_rows).sort_values(
        "mean_magnitude_contribution", ascending=False, kind="stable"
    )
    if not family_mag_summary_df.empty:
        denom = float(family_mag_summary_df["mean_magnitude_contribution"].sum())
        family_mag_summary_df["relative_percent_magnitude"] = (
            100.0 * family_mag_summary_df["mean_magnitude_contribution"] / denom if denom > 0 else 0.0
        )
        audit["new_relative_percent_magnitude"] = {
            str(r.feature_family): float(r.relative_percent_magnitude)
            for r in family_mag_summary_df.itertuples(index=False)
        }

    tables["feature_summary"] = feature_summary_df
    tables["family_net_summary"] = family_net_summary_df
    tables["family_magnitude_summary"] = family_mag_summary_df
    tables["family_raw_summary"] = family_raw_summary_df
    tables["row_decomposition"] = row_contrib_df
    return tables, audit, skipped


def family_contribution_plots(
    family_net_summary_df: pd.DataFrame,
    family_magnitude_summary_df: pd.DataFrame,
    out_dir: Path,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    generated: List[Tuple[str, str]] = []
    skipped: List[str] = []
    required_net_cols = {
        "feature_family",
        "mean_abs_net_contribution",
        "mean_signed_net_contribution",
        "std_net_contribution",
        "relative_percent_net",
    }
    required_mag_cols = {
        "feature_family",
        "mean_magnitude_contribution",
        "std_magnitude_contribution",
        "relative_percent_magnitude",
    }
    if family_net_summary_df.empty or not required_net_cols.issubset(family_net_summary_df.columns):
        skipped.append(
            "family_mean_abs_net_contribution.png: skipped (NET family contribution summary unavailable)."
        )
        skipped.append(
            "family_relative_contribution_net_percent.png: skipped (NET family contribution summary unavailable)."
        )
        skipped.append("family_contribution_summary_net.csv: skipped (NET family contribution summary unavailable).")
    else:
        family_net_summary_df.to_csv(out_dir / "family_contribution_summary_net.csv", index=False)
        generated.append(
            (
                "family_contribution_summary_net.csv",
                "Family NET contribution summary in standardized model space.",
            )
        )

        dd = family_net_summary_df.sort_values(
            "mean_abs_net_contribution", ascending=False, kind="stable"
        ).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        bars = ax.bar(
            dd["feature_family"],
            dd["mean_abs_net_contribution"],
            color=[FAMILY_COLOR_MAP.get(f, "#9ca3af") for f in dd["feature_family"].astype(str).tolist()],
        )
        for bar, val in zip(bars, dd["mean_abs_net_contribution"].to_numpy(dtype=float)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1e-9, 0.01 * float(dd["mean_abs_net_contribution"].max())),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title("NET Family Contribution (Standardized Space)")
        ax.set_ylabel("Mean |net contribution|")
        ax.grid(axis="y", alpha=0.25)
        out_name = "family_mean_abs_net_contribution.png"
        save_figure(fig, out_dir / out_name)
        generated.append((out_name, "NET family mean absolute contribution in standardized space."))

        dd_pct = family_net_summary_df.sort_values(
            "relative_percent_net", ascending=False, kind="stable"
        ).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        bars = ax.bar(
            dd_pct["feature_family"],
            dd_pct["relative_percent_net"],
            color=[FAMILY_COLOR_MAP.get(f, "#9ca3af") for f in dd_pct["feature_family"].astype(str).tolist()],
        )
        for bar, val in zip(bars, dd_pct["relative_percent_net"].to_numpy(dtype=float)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1e-9, 0.01 * float(dd_pct["relative_percent_net"].max())),
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title("NET Family Relative Contribution (%)")
        ax.set_ylabel("Relative contribution (%)")
        ax.grid(axis="y", alpha=0.25)
        out_name = "family_relative_contribution_net_percent.png"
        save_figure(fig, out_dir / out_name)
        generated.append((out_name, "NET family relative contribution percentage in standardized space."))

    if family_magnitude_summary_df.empty or not required_mag_cols.issubset(family_magnitude_summary_df.columns):
        skipped.append(
            "family_mean_magnitude_contribution.png: skipped (MAGNITUDE family contribution summary unavailable)."
        )
        skipped.append(
            "family_relative_contribution_magnitude_percent.png: skipped (MAGNITUDE family contribution summary unavailable)."
        )
        skipped.append(
            "family_contribution_summary_magnitude.csv: skipped (MAGNITUDE family contribution summary unavailable)."
        )
    else:
        family_magnitude_summary_df.to_csv(out_dir / "family_contribution_summary_magnitude.csv", index=False)
        generated.append(
            (
                "family_contribution_summary_magnitude.csv",
                "Family MAGNITUDE contribution summary in standardized model space (no cancellation).",
            )
        )

        dd = family_magnitude_summary_df.sort_values(
            "mean_magnitude_contribution", ascending=False, kind="stable"
        ).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        bars = ax.bar(
            dd["feature_family"],
            dd["mean_magnitude_contribution"],
            color=[FAMILY_COLOR_MAP.get(f, "#9ca3af") for f in dd["feature_family"].astype(str).tolist()],
        )
        for bar, val in zip(bars, dd["mean_magnitude_contribution"].to_numpy(dtype=float)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1e-9, 0.01 * float(dd["mean_magnitude_contribution"].max())),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title("MAGNITUDE Family Contribution (Standardized Space)")
        ax.set_ylabel("Mean magnitude contribution")
        ax.grid(axis="y", alpha=0.25)
        out_name = "family_mean_magnitude_contribution.png"
        save_figure(fig, out_dir / out_name)
        generated.append((out_name, "MAGNITUDE family mean contribution (sum of absolute feature contributions)."))

        dd_pct = family_magnitude_summary_df.sort_values(
            "relative_percent_magnitude", ascending=False, kind="stable"
        ).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        bars = ax.bar(
            dd_pct["feature_family"],
            dd_pct["relative_percent_magnitude"],
            color=[FAMILY_COLOR_MAP.get(f, "#9ca3af") for f in dd_pct["feature_family"].astype(str).tolist()],
        )
        for bar, val in zip(bars, dd_pct["relative_percent_magnitude"].to_numpy(dtype=float)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1e-9, 0.01 * float(dd_pct["relative_percent_magnitude"].max())),
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title("MAGNITUDE Family Relative Contribution (%)")
        ax.set_ylabel("Relative contribution (%)")
        ax.grid(axis="y", alpha=0.25)
        out_name = "family_relative_contribution_magnitude_percent.png"
        save_figure(fig, out_dir / out_name)
        generated.append((out_name, "MAGNITUDE family relative contribution percentage in standardized space."))

    return generated, skipped


def _plot_top_feature_contributions(
    df: pd.DataFrame,
    title: str,
    out_name: str,
    out_dir: Path,
) -> Optional[Tuple[str, str]]:
    if df.empty:
        return None
    plot_df = df.sort_values("mean_abs_contribution", ascending=True, kind="stable")
    fig_h = max(4.5, 0.33 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    colors = [FAMILY_COLOR_MAP.get(fam, "#9ca3af") for fam in plot_df["feature_family"].astype(str).tolist()]
    ax.barh(plot_df["feature"], plot_df["mean_abs_contribution"], color=colors)
    ax.set_title(title)
    ax.set_xlabel("Mean |coefficient * standardized_feature_value|")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, out_dir / out_name)
    return out_name, title


def top_feature_contribution_plots(
    feature_summary_df: pd.DataFrame,
    out_dir: Path,
    top_n: int,
    include_raw_embedding_features: bool,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    generated: List[Tuple[str, str]] = []
    skipped: List[str] = []
    req = {
        "feature",
        "feature_family",
        "coefficient",
        "feature_mean_raw",
        "feature_std_raw",
        "feature_mean_z",
        "feature_std_z",
        "mean_abs_contribution",
        "mean_signed_contribution",
        "std_contribution",
    }
    if feature_summary_df.empty or not req.issubset(feature_summary_df.columns):
        skipped.append(
            "top_feature_mean_abs_contribution_all.png: skipped (feature contribution summary unavailable)."
        )
        skipped.append(
            "top_feature_mean_abs_contribution_interpretable.png: skipped (feature contribution summary unavailable)."
        )
        skipped.append(
            "top_feature_contribution_summary_all.csv: skipped (feature contribution summary unavailable)."
        )
        skipped.append(
            "top_feature_contribution_summary_interpretable.csv: skipped (feature contribution summary unavailable)."
        )
        return generated, skipped

    feature_summary_df.to_csv(out_dir / "feature_contribution_summary_standardized.csv", index=False)
    generated.append(
        (
            "feature_contribution_summary_standardized.csv",
            "Per-feature contribution summary in standardized model space.",
        )
    )

    n = max(int(top_n), 1)

    all_source = feature_summary_df.copy()
    if not bool(include_raw_embedding_features):
        all_source = all_source.loc[all_source["feature_family"] != "embedding_raw"].copy()
        skipped.append(
            "Top-feature (all) plot excludes raw embedding dimensions by default; set --include-raw-embedding-features-in-top-plot to include."
        )
    all_top = all_source.sort_values("mean_abs_contribution", ascending=False, kind="stable").head(n).copy()
    if all_top.empty:
        skipped.append("top_feature_mean_abs_contribution_all.png: skipped (no features after filtering).")
        skipped.append("top_feature_contribution_summary_all.csv: skipped (no features after filtering).")
    else:
        all_top.to_csv(out_dir / "top_feature_contribution_summary_all.csv", index=False)
        generated.append(
            (
                "top_feature_contribution_summary_all.csv",
                "Top features ranked by mean absolute contribution (all-view filtering policy).",
            )
        )
        result = _plot_top_feature_contributions(
            all_top,
            title=f"Top {len(all_top)} Features by Mean Absolute Contribution (All)",
            out_name="top_feature_mean_abs_contribution_all.png",
            out_dir=out_dir,
        )
        if result is not None:
            generated.append((result[0], "Top-N feature contributions (all features view)."))

    interpretable_source = feature_summary_df.loc[
        ~feature_summary_df["feature_family"].isin(["embedding_raw", "embedding_pca"])
    ].copy()
    interpretable_top = (
        interpretable_source.sort_values("mean_abs_contribution", ascending=False, kind="stable").head(n).copy()
    )
    if interpretable_top.empty:
        skipped.append(
            "top_feature_mean_abs_contribution_interpretable.png: skipped (no non-embedding features available)."
        )
        skipped.append(
            "top_feature_contribution_summary_interpretable.csv: skipped (no non-embedding features available)."
        )
    else:
        interpretable_top.to_csv(out_dir / "top_feature_contribution_summary_interpretable.csv", index=False)
        generated.append(
            (
                "top_feature_contribution_summary_interpretable.csv",
                "Top non-embedding interpretable features ranked by mean absolute contribution.",
            )
        )
        result = _plot_top_feature_contributions(
            interpretable_top,
            title=f"Top {len(interpretable_top)} Interpretable Features by Mean Absolute Contribution",
            out_name="top_feature_mean_abs_contribution_interpretable.png",
            out_dir=out_dir,
        )
        if result is not None:
            generated.append(
                (
                    result[0],
                    "Top-N mean absolute contribution among non-embedding interpretable features.",
                )
            )

    return generated, skipped


def selected_gene_score_decomposition_plot(
    row_contrib_df: pd.DataFrame,
    family_net_summary_df: pd.DataFrame,
    out_dir: Path,
    top_positives: int,
    top_false_positives: int,
    false_positive_rank_threshold: int,
    intercept_available: bool,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    generated: List[Tuple[str, str]] = []
    skipped: List[str] = []
    needed = {"gene_symbol", "gwas_study_locus_id", "label_positive", "predicted_score", "rank_within_locus"}
    if row_contrib_df.empty or not needed.issubset(row_contrib_df.columns):
        skipped.append(
            "selected_gene_score_decomposition.png: skipped (required row-level columns missing for selection)."
        )
        skipped.append(
            "selected_gene_score_decomposition.csv: skipped (required row-level columns missing for selection)."
        )
        return generated, skipped

    family_net_cols = [c for c in row_contrib_df.columns if c.startswith("contrib_net_")]
    family_mag_cols = [c for c in row_contrib_df.columns if c.startswith("contrib_magnitude_")]
    if not family_net_cols:
        skipped.append("selected_gene_score_decomposition.png: skipped (no family contribution columns found).")
        skipped.append("selected_gene_score_decomposition.csv: skipped (no family contribution columns found).")
        return generated, skipped

    dd = row_contrib_df.copy()
    dd["label_positive"] = pd.to_numeric(dd["label_positive"], errors="coerce").fillna(0).astype(int)
    dd["predicted_score"] = pd.to_numeric(dd["predicted_score"], errors="coerce")
    dd["rank_within_locus"] = pd.to_numeric(dd["rank_within_locus"], errors="coerce")
    dd = dd.dropna(subset=["predicted_score", "rank_within_locus"])
    if dd.empty:
        skipped.append("selected_gene_score_decomposition.png: skipped (no valid rows after numeric cleaning).")
        skipped.append("selected_gene_score_decomposition.csv: skipped (no valid rows after numeric cleaning).")
        return generated, skipped

    top_pos = (
        dd.loc[dd["label_positive"] == 1]
        .sort_values("predicted_score", ascending=False, kind="stable")
        .head(max(int(top_positives), 0))
        .copy()
    )
    top_pos["selection_group"] = "top_true_positive"

    top_fp = (
        dd.loc[(dd["label_positive"] == 0) & (dd["rank_within_locus"] <= int(false_positive_rank_threshold))]
        .sort_values("predicted_score", ascending=False, kind="stable")
        .head(max(int(top_false_positives), 0))
        .copy()
    )
    top_fp["selection_group"] = "top_false_positive"

    selected = pd.concat([top_pos, top_fp], axis=0, ignore_index=True)
    if selected.empty:
        skipped.append("selected_gene_score_decomposition.png: skipped (no selected positive/false-positive rows).")
        skipped.append("selected_gene_score_decomposition.csv: skipped (no selected positive/false-positive rows).")
        return generated, skipped

    selected = selected.drop_duplicates(subset=["gwas_study_locus_id", "gene_symbol"], keep="first")

    csv_cols = [
        "selection_group",
        "gene_symbol",
        "gwas_study_locus_id",
        "label_positive",
        "rank_within_locus",
        "predicted_score",
    ] + family_net_cols + family_mag_cols + ["intercept", "score_reconstructed_net", "decision_score_reconstruction_error"]
    csv_cols = [c for c in csv_cols if c in selected.columns]
    selected[csv_cols].to_csv(out_dir / "selected_gene_score_decomposition.csv", index=False)
    generated.append(
        (
            "selected_gene_score_decomposition.csv",
            "Row-level score decomposition table for selected top positives and false positives.",
        )
    )

    if not family_net_summary_df.empty and "feature_family" in family_net_summary_df.columns:
        family_order = [
            f"contrib_net_{f}"
            for f in family_net_summary_df["feature_family"].astype(str).tolist()
            if f"contrib_net_{f}" in family_net_cols
        ]
        for col in family_net_cols:
            if col not in family_order:
                family_order.append(col)
    else:
        family_order = sorted(family_net_cols)

    fig_h = max(4.8, 0.7 * len(selected))
    fig, ax = plt.subplots(figsize=(12.0, fig_h))

    y_positions = np.arange(len(selected))
    for i, row in enumerate(selected.itertuples(index=False)):
        pos_left = 0.0
        neg_left = 0.0
        for fam_col in family_order:
            fam = fam_col.replace("contrib_net_", "")
            val = float(getattr(row, fam_col))
            color = FAMILY_COLOR_MAP.get(fam, "#9ca3af")
            if val >= 0:
                ax.barh(i, val, left=pos_left, color=color, edgecolor="white", linewidth=0.4)
                pos_left += val
            else:
                ax.barh(i, val, left=neg_left, color=color, edgecolor="white", linewidth=0.4)
                neg_left += val

    labels = [
        f"{str(r.gene_symbol)} | {str(r.gwas_study_locus_id)[:8]} | y={int(r.label_positive)} | r={int(r.rank_within_locus)}"
        for r in selected.itertuples(index=False)
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Approx. family contribution in standardized model space")
    if intercept_available:
        ax.set_title("Selected Gene Score Decomposition by Feature Family (Standardized, NET)")
    else:
        ax.set_title("Selected Gene Score Decomposition by Feature Family (Standardized, NET; intercept unavailable)")
    ax.grid(axis="x", alpha=0.25)

    for i, r in enumerate(selected.itertuples(index=False)):
        pred = float(getattr(r, "predicted_score", np.nan))
        recon = float(getattr(r, "score_reconstructed_net", np.nan))
        ax.text(
            ax.get_xlim()[1] * 0.98,
            i,
            f"pred_prob={pred:.3f} | net_linear={recon:.3f}",
            ha="right",
            va="center",
            fontsize=8,
            color="#111827",
        )

    legend_handles = [
        Patch(color=FAMILY_COLOR_MAP.get(col.replace("contrib_", ""), "#9ca3af"), label=col.replace("contrib_", ""))
        for col in family_order
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8, ncol=2)

    out_name = "selected_gene_score_decomposition.png"
    save_figure(fig, out_dir / out_name)
    generated.append((out_name, "Stacked family-contribution decomposition for selected genes."))
    return generated, skipped


def write_contribution_audit_report(
    out_dir: Path,
    ranking_mode: str,
    audit: Dict[str, object],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    generated: List[Tuple[str, str]] = []
    skipped: List[str] = []
    old_raw = audit.get("old_relative_percent_raw", {}) or {}
    new_net = audit.get("new_relative_percent_net", {}) or {}
    new_mag = audit.get("new_relative_percent_magnitude", {}) or {}
    if not old_raw or not new_net or not new_mag:
        skipped.append(
            "contribution_audit_report.md: skipped (old/new contribution percentages unavailable)."
        )
        return generated, skipped

    families = sorted(set(old_raw.keys()) | set(new_net.keys()) | set(new_mag.keys()))
    emb_raw_old = float(old_raw.get("embedding_raw", 0.0))
    emb_raw_net = float(new_net.get("embedding_raw", 0.0))
    emb_raw_mag = float(new_mag.get("embedding_raw", 0.0))
    emb_pca_old = float(old_raw.get("embedding_pca", 0.0))
    emb_pca_net = float(new_net.get("embedding_pca", 0.0))
    emb_pca_mag = float(new_mag.get("embedding_pca", 0.0))
    emb_old = emb_raw_old + emb_pca_old
    emb_net = emb_raw_net + emb_pca_net
    emb_mag = emb_raw_mag + emb_pca_mag
    dist_old = float(old_raw.get("distance", 0.0))
    dist_net = float(new_net.get("distance", 0.0))
    dist_mag = float(new_mag.get("distance", 0.0))

    lines: List[str] = []
    lines.append("# Contribution Audit Report")
    lines.append("")
    lines.append(f"Mode analyzed: `{ranking_mode}`")
    lines.append("")
    lines.append("## Why The Previous Method Was Wrong")
    lines.append("")
    lines.append("- Previous relative contribution used `coefficient * raw_feature_value`.")
    lines.append("- The model was fit with `StandardScaler`, so coefficients live in standardized space.")
    lines.append("- Correct feature contribution is `coefficient * z`, where `z = (x - mean) / scale` using saved scaler stats.")
    lines.append("")
    lines.append("## Methods Compared")
    lines.append("")
    lines.append("- Old (for audit only): family NET from raw-space contributions.")
    lines.append("- Correct NET: `abs(sum_j beta_j * z_ij)` per family, then mean across rows.")
    lines.append("- Correct MAGNITUDE: `sum_j abs(beta_j * z_ij)` per family, then mean across rows.")
    lines.append("")
    lines.append("## Relative Percentage Comparison")
    lines.append("")
    lines.append("| feature_family | old_raw_percent | new_net_percent | new_magnitude_percent |")
    lines.append("|---|---:|---:|---:|")
    for fam in families:
        lines.append(
            f"| {fam} | {float(old_raw.get(fam, 0.0)):.3f} | {float(new_net.get(fam, 0.0)):.3f} | {float(new_mag.get(fam, 0.0)):.3f} |"
        )
    lines.append("")
    lines.append("## Interpretation Change")
    lines.append("")
    lines.append(
        f"- `embedding_pca`: old={emb_pca_old:.3f}% -> net={emb_pca_net:.3f}% -> magnitude={emb_pca_mag:.3f}%."
    )
    lines.append(
        f"- `embedding_raw`: old={emb_raw_old:.3f}% -> net={emb_raw_net:.3f}% -> magnitude={emb_raw_mag:.3f}%."
    )
    lines.append(
        f"- `embedding_total` (raw+pca): old={emb_old:.3f}% -> net={emb_net:.3f}% -> magnitude={emb_mag:.3f}%."
    )
    lines.append(
        f"- `distance`: old={dist_old:.3f}% -> net={dist_net:.3f}% -> magnitude={dist_mag:.3f}%."
    )
    if emb_net > emb_old:
        lines.append("- Embedding contribution increased after correcting scaling, indicating prior underestimation.")
    else:
        lines.append("- Embedding contribution did not increase after correction.")
    if dist_net < dist_old:
        lines.append("- Distance contribution decreased after correction, consistent with raw-scale inflation before.")
    else:
        lines.append("- Distance contribution did not decrease after correction.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    decision_col = audit.get("decision_score_column", None)
    intercept_available = bool(audit.get("intercept_available", False))
    if decision_col is None:
        lines.append("- Model decision score column was not available in all_ranked_predictions.csv.")
        lines.append("- Decomposition is reported in standardized linear contribution units.")
    else:
        lines.append(f"- Decision score column used for reconstruction checks: `{decision_col}`.")
    if intercept_available:
        lines.append(f"- Intercept loaded from model parameters: {float(audit.get('intercept_value', 0.0)):.6f}.")
    else:
        lines.append("- Intercept was not available; assumed 0.0 for approximate decomposition.")
    pca_reconstructed = int(audit.get("pca_features_reconstructed", 0) or 0)
    if pca_reconstructed > 0:
        lines.append(f"- PCA embedding features reconstructed for contribution analysis: {pca_reconstructed}.")
    lines.append("")

    out_name = "contribution_audit_report.md"
    (out_dir / out_name).write_text("\n".join(lines), encoding="utf-8")
    generated.append((out_name, "Audit of old raw-space vs corrected standardized contribution percentages."))
    return generated, skipped


def feature_family_utility_summary_plot(
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Hook for family-level utility ablations.

    Current training outputs compare embedding modes (none/full/pca) but do not
    isolate dedicated '+HPA only' or '+network only' ablations. If those are
    added later, this function can be extended to plot explicit family utility.
    """
    generated: List[Tuple[str, str]] = []
    skipped: List[str] = []
    needed_cols = {"mode", "mean_fold_pr_auc", "mean_fold_roc_auc", "used_abundance_feature_count"}
    if not needed_cols.issubset(summary_df.columns):
        skipped.append(
            "feature_family_utility_ablation.png: skipped (summary table missing required columns for family utility hook)."
        )
        return generated, skipped

    has_abundance_toggle = summary_df["used_abundance_feature_count"].nunique() > 1
    if not has_abundance_toggle:
        skipped.append(
            "feature_family_utility_ablation.png: skipped (no explicit HPA-ablation rows in summary; TODO extend once +HPA/-HPA runs are exported)."
        )
        return generated, skipped

    dd = summary_df.sort_values("mode", kind="stable").copy()
    x = np.arange(len(dd))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(x, dd["mean_fold_pr_auc"].astype(float), marker="o", label="PR-AUC", color="#f28e2b")
    ax.plot(x, dd["mean_fold_roc_auc"].astype(float), marker="o", label="ROC-AUC", color="#4e79a7")
    ax.set_xticks(x)
    ax.set_xticklabels(dd["mode"].astype(str).tolist(), rotation=25, ha="right")
    ax.set_ylabel("Metric")
    ax.set_title("Family Utility Ablation Summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    out_name = "feature_family_utility_ablation.png"
    save_figure(fig, out_dir / out_name)
    generated.append((out_name, "Utility hook plot for future family-level ablation runs."))
    return generated, skipped


def score_distribution_plot(pred_df: pd.DataFrame, title: str, filename: str, out_dir: Path) -> List[Tuple[str, str]]:
    generated: List[Tuple[str, str]] = []
    if pred_df.empty:
        return generated
    pos = pred_df[pred_df["label_positive"].astype(int) == 1]["predicted_score"].astype(float)
    neg = pred_df[pred_df["label_positive"].astype(int) == 0]["predicted_score"].astype(float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(neg, bins=25, alpha=0.65, color="#6b7280", label="Negative", density=True)
    if not pos.empty:
        ax.hist(pos, bins=25, alpha=0.65, color="#dc2626", label="Positive", density=True)
    ax.set_title(title)
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, out_dir / filename)
    generated.append((filename, f"Predicted score distribution (positives vs negatives) for {title.lower()}."))
    return generated


def scatter_with_labels(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    out_name: str,
    out_dir: Path,
) -> List[Tuple[str, str]]:
    generated: List[Tuple[str, str]] = []
    if x_col not in df.columns or y_col not in df.columns:
        return generated
    dd = df[[x_col, y_col, "label_positive"]].copy()
    dd[x_col] = pd.to_numeric(dd[x_col], errors="coerce")
    dd[y_col] = pd.to_numeric(dd[y_col], errors="coerce")
    dd = dd.dropna(subset=[x_col, y_col])
    if dd.empty:
        return generated

    neg = dd[dd["label_positive"].astype(int) == 0]
    pos = dd[dd["label_positive"].astype(int) == 1]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.scatter(neg[x_col], neg[y_col], s=18, alpha=0.6, color="#6b7280", label="Negative")
    if not pos.empty:
        ax.scatter(pos[x_col], pos[y_col], s=40, alpha=0.85, color="#dc2626", label="Positive")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Predicted score")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir / out_name)
    generated.append((out_name, f"{x_label} versus predicted score, with positives highlighted."))
    return generated


def qtl_presence_boxplot(df: pd.DataFrame, out_dir: Path) -> List[Tuple[str, str]]:
    generated: List[Tuple[str, str]] = []
    if "has_qtl_evidence" not in df.columns:
        return generated
    dd = df[["has_qtl_evidence", "predicted_score", "label_positive"]].copy()
    dd["has_qtl_evidence"] = pd.to_numeric(dd["has_qtl_evidence"], errors="coerce").fillna(0).astype(int)
    dd["predicted_score"] = pd.to_numeric(dd["predicted_score"], errors="coerce")
    dd = dd.dropna(subset=["predicted_score"])
    if dd.empty:
        return generated
    groups = [dd.loc[dd["has_qtl_evidence"] == i, "predicted_score"].to_numpy() for i in [0, 1]]
    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    ax.boxplot(groups, labels=["No QTL", "Has QTL"], showfliers=False)
    # jitter points
    rng = np.random.default_rng(42)
    for i, grp in enumerate(groups, start=1):
        if grp.size == 0:
            continue
        x = rng.normal(loc=i, scale=0.04, size=grp.size)
        ax.scatter(x, grp, s=10, alpha=0.35, color="#6b7280")
    ax.set_title("QTL Evidence Presence vs Predicted Score")
    ax.set_ylabel("Predicted score")
    ax.grid(axis="y", alpha=0.25)
    out_name = "qtl_presence_vs_score.png"
    save_figure(fig, out_dir / out_name)
    generated.append((out_name, "Box plot of predicted scores by has_qtl_evidence (0/1)."))
    return generated


def embedding_coverage_plot(df: pd.DataFrame, out_dir: Path) -> List[Tuple[str, str]]:
    generated: List[Tuple[str, str]] = []
    if "has_gene_embedding" not in df.columns or "gene_symbol" not in df.columns:
        return generated

    has_emb = pd.to_numeric(df["has_gene_embedding"], errors="coerce").fillna(0).astype(int)
    row_with = int((has_emb == 1).sum())
    row_without = int((has_emb == 0).sum())
    gene_with = int(df.loc[has_emb == 1, "gene_symbol"].astype(str).nunique())
    gene_without = int(df.loc[has_emb == 0, "gene_symbol"].astype(str).nunique())

    categories = ["Rows", "Unique genes"]
    with_vals = [row_with, gene_with]
    without_vals = [row_without, gene_without]
    x = np.arange(len(categories))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    b1 = ax.bar(x - w / 2, with_vals, width=w, label="With embedding", color="#59a14f")
    b2 = ax.bar(x + w / 2, without_vals, width=w, label="Without embedding", color="#9ca3af")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1, 0.01 * max(with_vals + without_vals)),
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Count")
    ax.set_title("Embedding Coverage")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    out_name = "embedding_coverage.png"
    save_figure(fig, out_dir / out_name)
    generated.append((out_name, "Counts of rows and unique genes with vs without embeddings."))
    return generated


def top_false_positive_plot(
    df: pd.DataFrame,
    out_dir: Path,
    top_n: int,
    rank_threshold: int,
) -> List[Tuple[str, str]]:
    """
    Plot high-priority false positives using *within-locus* ranking.

    Old logic (incorrect for this project):
    - filter label_positive == 0
    - sort globally by predicted_score across all loci
    - keep top N
    This treats scores as directly comparable across loci and ignores the fact that
    the model is used for ranking genes *inside each locus*.

    New logic (aligned with interpretation):
    - filter label_positive == 0
    - filter rank_within_locus <= K
    - sort by (rank_within_locus asc, predicted_score desc)
    - optionally keep top N after this filter
    This preserves locus-level ranking semantics first, then uses score as tie-breaker.
    """
    generated: List[Tuple[str, str]] = []
    if df.empty:
        return generated
    required = [
        "gene_symbol",
        "gwas_study_locus_id",
        "predicted_score",
        "rank_within_locus",
        "has_qtl_evidence",
        "has_gene_embedding",
        "dist_score_500kb_log",
        "label_positive",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return generated

    fp = df.copy()
    fp["label_positive"] = pd.to_numeric(fp["label_positive"], errors="coerce").fillna(0).astype(int)
    fp["rank_within_locus"] = pd.to_numeric(fp["rank_within_locus"], errors="coerce")
    fp["predicted_score"] = pd.to_numeric(fp["predicted_score"], errors="coerce")
    fp["has_qtl_evidence"] = pd.to_numeric(fp["has_qtl_evidence"], errors="coerce").fillna(0).astype(int)
    fp["has_gene_embedding"] = pd.to_numeric(fp["has_gene_embedding"], errors="coerce").fillna(0).astype(int)
    fp["dist_score_500kb_log"] = pd.to_numeric(fp["dist_score_500kb_log"], errors="coerce")
    fp = fp.dropna(subset=["rank_within_locus", "predicted_score"])
    fp["rank_within_locus"] = fp["rank_within_locus"].astype(int)

    fp = fp.loc[(fp["label_positive"] == 0) & (fp["rank_within_locus"] <= int(rank_threshold))].copy()
    fp = fp.sort_values(
        by=["rank_within_locus", "predicted_score"],
        ascending=[True, False],
        kind="stable",
    )
    if int(top_n) > 0 and len(fp) > int(top_n):
        fp = fp.head(int(top_n)).copy()
    if fp.empty:
        return generated

    # Persist exactly the rows used in this figure.
    csv_name = "top_false_positives_by_locus_rank.csv"
    csv_cols = [
        "gene_symbol",
        "gwas_study_locus_id",
        "rank_within_locus",
        "predicted_score",
        "has_qtl_evidence",
        "has_gene_embedding",
        "dist_score_500kb_log",
        "label_positive",
    ]
    fp[csv_cols].to_csv(out_dir / csv_name, index=False)

    # Reverse for barh so highest priority appears on top.
    fp = fp.iloc[::-1].reset_index(drop=True)
    labels = [
        f"{g} | {str(l)[:8]} | r={int(r)}"
        for g, l, r in zip(fp["gene_symbol"], fp["gwas_study_locus_id"], fp["rank_within_locus"])
    ]

    fig_h = max(4.2, 0.38 * len(fp))
    fig, ax = plt.subplots(figsize=(12.0, fig_h))
    ax.barh(labels, fp["predicted_score"], color="#f28e2b")
    ax.set_xlabel("Predicted score")
    ax.set_title(
        f"False Positives by Within-Locus Rank (label=0, rank<= {int(rank_threshold)}, shown={len(fp)})"
    )
    ax.grid(axis="x", alpha=0.25)
    for i, row in enumerate(fp.itertuples(index=False)):
        ax.text(
            row.predicted_score + 0.005,
            i,
            (
                f"score={float(row.predicted_score):.3f} "
                f"rank={int(row.rank_within_locus)} "
                f"qtl={int(row.has_qtl_evidence)} "
                f"emb={int(row.has_gene_embedding)} "
                f"distScore={float(row.dist_score_500kb_log):.3f}"
            ),
            va="center",
            fontsize=8,
        )

    out_name = "top_false_positives_by_locus_rank.png"
    save_figure(fig, out_dir / out_name)
    generated.append(
        (
            out_name,
            (
                "False positives selected by within-locus rank (label=0, rank<=K), "
                "then ordered by rank asc and score desc; annotations include score/rank/qtl/embedding/distance."
            ),
        )
    )
    generated.append(
        (
            csv_name,
            "CSV with exactly the rows used in top_false_positives_by_locus_rank.png.",
        )
    )
    return generated


def optional_text_plots(df: pd.DataFrame, out_dir: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    generated: List[Tuple[str, str]] = []
    skipped: List[str] = []
    text_col = None
    for col in TEXT_SCORE_CANDIDATE_COLUMNS:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        skipped.append("text_score_vs_prediction.png: skipped (no text-score column found).")
        skipped.append("text_score_distribution.png: skipped (no text-score column found).")
        return generated, skipped

    dd = df[[text_col, "predicted_score", "label_positive"]].copy()
    dd[text_col] = pd.to_numeric(dd[text_col], errors="coerce")
    dd["predicted_score"] = pd.to_numeric(dd["predicted_score"], errors="coerce")
    dd = dd.dropna()
    if dd.empty:
        skipped.append("text_score_vs_prediction.png: skipped (text-score values are all missing).")
        skipped.append("text_score_distribution.png: skipped (text-score values are all missing).")
        return generated, skipped

    neg = dd[dd["label_positive"].astype(int) == 0]
    pos = dd[dd["label_positive"].astype(int) == 1]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.scatter(neg[text_col], neg["predicted_score"], s=16, alpha=0.55, color="#6b7280", label="Negative")
    if not pos.empty:
        ax.scatter(pos[text_col], pos["predicted_score"], s=38, alpha=0.85, color="#dc2626", label="Positive")
    ax.set_title(f"{text_col} vs Predicted Score")
    ax.set_xlabel(text_col)
    ax.set_ylabel("Predicted score")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir / "text_score_vs_prediction.png")
    generated.append(("text_score_vs_prediction.png", f"Scatter plot of {text_col} versus predicted score."))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(neg[text_col], bins=25, alpha=0.65, color="#6b7280", label="Negative", density=True)
    if not pos.empty:
        ax.hist(pos[text_col], bins=25, alpha=0.65, color="#dc2626", label="Positive", density=True)
    ax.set_title(f"{text_col} Distribution")
    ax.set_xlabel(text_col)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir / "text_score_distribution.png")
    generated.append(("text_score_distribution.png", f"Distribution of {text_col} for positives vs negatives."))

    return generated, skipped


def write_figure_index(
    figures_dir: Path,
    generated: List[Tuple[str, str]],
    skipped: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# Figure Index")
    lines.append("")
    lines.append(f"Output directory: `{figures_dir}`")
    lines.append("")
    lines.append("## Generated outputs")
    lines.append("")
    if not generated:
        lines.append("- None")
    else:
        for fname, desc in generated:
            lines.append(f"- `{fname}`: {desc}")
    lines.append("")
    lines.append("## Skipped figures")
    lines.append("")
    if not skipped:
        lines.append("- None")
    else:
        for item in skipped:
            lines.append(f"- {item}")
    lines.append("")
    (figures_dir / "figure_index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # load modes
    mode_none = load_mode_tables(args.results_dir, "none")
    mode_full = load_mode_tables(args.results_dir, "full")
    mode_pca = load_mode_tables(args.results_dir, "pca")
    l1_benchmark_df = load_l1_benchmark_table(args.results_dir)

    summary_df = pd.concat(
        [mode_none["summary"], mode_full["summary"], mode_pca["summary"]],
        axis=0,
        ignore_index=True,
    )
    generated: List[Tuple[str, str]] = []
    skipped: List[str] = []

    # 1) Model performance comparison
    generated.extend(metric_comparison_plots(summary_df, args.figures_dir))
    l1_generated, l1_skipped = l1_benchmark_plots(l1_benchmark_df, args.figures_dir)
    generated.extend(l1_generated)
    skipped.extend(l1_skipped)

    # choose mode for locus-specific inspection plots
    chosen_mode = {
        "none": mode_none,
        "full": mode_full,
        "pca": mode_pca,
    }[args.ranking_mode]
    pred_df = chosen_mode["all_predictions"].copy()

    # 2) Per-locus ranking visualization
    generated.extend(
        per_locus_ranking_plots(
            pred_df=pred_df,
            out_dir=args.figures_dir,
            top_k_annot=int(args.top_k_label_annotations),
        )
    )

    # 3) Rank position of true positives
    generated.extend(positive_rank_distribution_plot(chosen_mode["positive_ranks"], args.figures_dir))

    # 4) Embedding ablation effect (none vs full)
    generated.extend(
        embedding_ablation_plot(
            pos_none=mode_none["positive_ranks"],
            pos_full=mode_full["positive_ranks"],
            out_dir=args.figures_dir,
        )
    )

    # 5) Coefficients / feature importance
    generated.extend(
        coefficient_plot(
            mode_none["coefficients"],
            title="Baseline Model Coefficients (Non-zero)",
            filename="coefficients_baseline.png",
            out_dir=args.figures_dir,
        )
    )
    generated.extend(
        coefficient_plot(
            mode_full["coefficients"],
            title="Full Embedding Model Coefficients (Non-zero)",
            filename="coefficients_full_embeddings.png",
            out_dir=args.figures_dir,
        )
    )
    generated.extend(
        coefficient_plot(
            mode_pca["coefficients"],
            title="PCA Embedding Model Coefficients (Non-zero)",
            filename="coefficients_pca_embeddings.png",
            out_dir=args.figures_dir,
        )
    )
    generated.extend(
        feature_weight_comparison_plot(
            coef_none=mode_none["coefficients"],
            coef_full=mode_full["coefficients"],
            coef_pca=mode_pca["coefficients"],
            out_dir=args.figures_dir,
        )
    )

    # 6) Score distributions by model
    generated.extend(
        score_distribution_plot(
            mode_none["all_predictions"],
            title="Score Distribution: Baseline",
            filename="score_distribution_baseline.png",
            out_dir=args.figures_dir,
        )
    )
    generated.extend(
        score_distribution_plot(
            mode_full["all_predictions"],
            title="Score Distribution: Full Embeddings",
            filename="score_distribution_full_embeddings.png",
            out_dir=args.figures_dir,
        )
    )
    generated.extend(
        score_distribution_plot(
            mode_pca["all_predictions"],
            title="Score Distribution: PCA Embeddings",
            filename="score_distribution_pca_embeddings.png",
            out_dir=args.figures_dir,
        )
    )

    # 7) Distance vs score
    generated.extend(
        scatter_with_labels(
            pred_df,
            x_col="dist_variant_to_gene_kb",
            y_col="predicted_score",
            title="Distance to Gene vs Predicted Score",
            x_label="dist_variant_to_gene_kb",
            out_name="distance_to_gene_vs_score.png",
            out_dir=args.figures_dir,
        )
    )
    generated.extend(
        scatter_with_labels(
            pred_df,
            x_col="dist_variant_to_tss_kb",
            y_col="predicted_score",
            title="Distance to TSS vs Predicted Score",
            x_label="dist_variant_to_tss_kb",
            out_name="distance_to_tss_vs_score.png",
            out_dir=args.figures_dir,
        )
    )

    # 8) QTL/coloc signal vs score
    generated.extend(
        scatter_with_labels(
            pred_df,
            x_col="colocalisation_h4_max",
            y_col="predicted_score",
            title="colocalisation_h4_max vs Predicted Score",
            x_label="colocalisation_h4_max",
            out_name="h4_vs_score.png",
            out_dir=args.figures_dir,
        )
    )
    generated.extend(
        scatter_with_labels(
            pred_df,
            x_col="colocalisation_clpp_max",
            y_col="predicted_score",
            title="colocalisation_clpp_max vs Predicted Score",
            x_label="colocalisation_clpp_max",
            out_name="clpp_vs_score.png",
            out_dir=args.figures_dir,
        )
    )
    generated.extend(qtl_presence_boxplot(pred_df, args.figures_dir))

    # 9) Embedding coverage
    generated.extend(embedding_coverage_plot(pred_df, args.figures_dir))

    # 10) Top false positives
    generated.extend(
        top_false_positive_plot(
            df=pred_df,
            out_dir=args.figures_dir,
            top_n=int(args.top_false_positive_n),
            rank_threshold=int(args.false_positive_rank_threshold),
        )
    )

    # 11) Contribution analysis (family / feature / selected genes)
    contrib_tables, contrib_audit, contrib_skipped = compute_contribution_tables(
        pred_df=pred_df,
        coef_df=chosen_mode["coefficients"],
        scaler_stats_df=chosen_mode["scaler_stats"],
        model_params=chosen_mode["model_params"],
        pca_artifacts=chosen_mode.get("pca_artifacts"),
    )
    skipped.extend(contrib_skipped)

    feature_contrib_df = contrib_tables["feature_summary"]
    family_net_df = contrib_tables["family_net_summary"]
    family_mag_df = contrib_tables["family_magnitude_summary"]
    family_raw_df = contrib_tables["family_raw_summary"]
    row_contrib_df = contrib_tables["row_decomposition"]

    if not family_raw_df.empty:
        family_raw_df.to_csv(args.figures_dir / "family_contribution_summary_raw_for_audit.csv", index=False)
        generated.append(
            (
                "family_contribution_summary_raw_for_audit.csv",
                "Raw-space family contribution summary kept only for audit comparison with standardized-space results.",
            )
        )

    contrib_generated, contrib_skipped = family_contribution_plots(
        family_net_summary_df=family_net_df,
        family_magnitude_summary_df=family_mag_df,
        out_dir=args.figures_dir,
    )
    generated.extend(contrib_generated)
    skipped.extend(contrib_skipped)

    contrib_generated, contrib_skipped = top_feature_contribution_plots(
        feature_summary_df=feature_contrib_df,
        out_dir=args.figures_dir,
        top_n=int(args.top_contribution_features),
        include_raw_embedding_features=bool(args.include_raw_embedding_features_in_top_plot),
    )
    generated.extend(contrib_generated)
    skipped.extend(contrib_skipped)

    contrib_generated, contrib_skipped = selected_gene_score_decomposition_plot(
        row_contrib_df=row_contrib_df,
        family_net_summary_df=family_net_df,
        out_dir=args.figures_dir,
        top_positives=int(args.decomposition_top_positives),
        top_false_positives=int(args.decomposition_top_false_positives),
        false_positive_rank_threshold=int(args.false_positive_rank_threshold),
        intercept_available=bool(contrib_audit.get("intercept_available", False)),
    )
    generated.extend(contrib_generated)
    skipped.extend(contrib_skipped)

    contrib_generated, contrib_skipped = write_contribution_audit_report(
        out_dir=args.figures_dir,
        ranking_mode=str(args.ranking_mode),
        audit=contrib_audit,
    )
    generated.extend(contrib_generated)
    skipped.extend(contrib_skipped)

    # 12) Optional family utility ablation hook
    utility_generated, utility_skipped = feature_family_utility_summary_plot(summary_df, args.figures_dir)
    generated.extend(utility_generated)
    skipped.extend(utility_skipped)

    # 13) Optional text-score plots
    text_generated, text_skipped = optional_text_plots(pred_df, args.figures_dir)
    generated.extend(text_generated)
    skipped.extend(text_skipped)

    # mark skipped required plots when data absent
    if chosen_mode["positive_ranks"].empty:
        skipped.append("positive_rank_distribution.png: skipped (positive rank table is empty).")
    if mode_none["positive_ranks"].empty or mode_full["positive_ranks"].empty:
        skipped.append("embedding_ablation_per_locus.png: skipped (missing positive-rank rows in none/full mode).")

    write_figure_index(args.figures_dir, generated, skipped)
    print(f"[done] Generated {len(generated)} figure files in {args.figures_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate PNG visualizations for the locus-to-gene ranking proof-of-concept."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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
        help="Number of highest-scoring false positives to plot.",
    )
    return parser.parse_args()


def load_mode_tables(results_dir: Path, mode: str) -> Dict[str, pd.DataFrame]:
    mode_dir = results_dir / f"mode_{mode}"
    if not mode_dir.exists():
        raise FileNotFoundError(f"Mode output directory not found: {mode_dir}")

    return {
        "summary": pd.read_json(mode_dir / "summary_metrics.json", typ="series").to_frame().T,
        "all_predictions": pd.read_csv(mode_dir / "all_ranked_predictions.csv"),
        "fold_metrics": pd.read_csv(mode_dir / "fold_metrics.csv"),
        "positive_ranks": pd.read_csv(mode_dir / "positive_gene_ranks.csv"),
        "coefficients": pd.read_csv(mode_dir / "coefficient_table.csv"),
        "pca_evr": pd.read_csv(mode_dir / "pca_explained_variance.csv"),
        "false_positives": pd.read_csv(mode_dir / "high_rank_false_positives.csv"),
    }


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


def top_false_positive_plot(df: pd.DataFrame, out_dir: Path, top_n: int) -> List[Tuple[str, str]]:
    generated: List[Tuple[str, str]] = []
    if df.empty:
        return generated
    required = ["gene_symbol", "gwas_study_locus_id", "predicted_score", "has_qtl_evidence", "has_gene_embedding", "dist_score_500kb_log", "label_positive"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return generated

    fp = df[df["label_positive"].astype(int) == 0].copy()
    fp = fp.sort_values("predicted_score", ascending=False, kind="stable").head(top_n).copy()
    if fp.empty:
        return generated
    fp = fp.iloc[::-1]  # for top at top in barh
    labels = [f"{g} | {l[:8]}" for g, l in zip(fp["gene_symbol"], fp["gwas_study_locus_id"])]

    fig_h = max(4.2, 0.38 * len(fp))
    fig, ax = plt.subplots(figsize=(10.8, fig_h))
    ax.barh(labels, fp["predicted_score"], color="#f28e2b")
    ax.set_xlabel("Predicted score")
    ax.set_title(f"Top {len(fp)} High-scoring False Positives")
    ax.grid(axis="x", alpha=0.25)
    for i, row in enumerate(fp.itertuples(index=False)):
        ax.text(
            row.predicted_score + 0.005,
            i,
            f"qtl={int(row.has_qtl_evidence)} emb={int(row.has_gene_embedding)} distScore={float(row.dist_score_500kb_log):.3f}",
            va="center",
            fontsize=8,
        )

    out_name = "top_false_positives.png"
    save_figure(fig, out_dir / out_name)
    generated.append((out_name, "Top-scoring false positives with locus/gene and key biological context annotations."))
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
    lines.append("## Generated figures")
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

    summary_df = pd.concat(
        [mode_none["summary"], mode_full["summary"], mode_pca["summary"]],
        axis=0,
        ignore_index=True,
    )
    generated: List[Tuple[str, str]] = []
    skipped: List[str] = []

    # 1) Model performance comparison
    generated.extend(metric_comparison_plots(summary_df, args.figures_dir))

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
        )
    )

    # 11) Optional text-score plots
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

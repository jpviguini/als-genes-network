#!/usr/bin/env python3
"""Generate no-network supervisor plots for reduced-baseline full comparison runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reduced-baseline no-network supervisor plots and summary."
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        required=True,
        help="Root containing runs/{noreg,l1,l2}/cv_lolo_gene_exclusion/mode_* outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for summary plots and markdown.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Top N rows by |delta_score| to export per regularization (l1/l2).",
    )
    return parser.parse_args()


def _load_summary_metrics(mode_dir: Path) -> Dict[str, float]:
    p = mode_dir / "summary_metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _load_predictions(mode_dir: Path) -> pd.DataFrame:
    return pd.read_csv(mode_dir / "all_ranked_predictions.csv")


def _safe_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _build_delta_table(baseline_pred: pd.DataFrame, pca_pred: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["fold_id", "gwas_study_locus_id", "gene_symbol"]
    optional_key_cols = ["gene_id", "gwas_lead_variant_id"]
    for c in optional_key_cols:
        if c in baseline_pred.columns and c in pca_pred.columns:
            key_cols.append(c)

    use_cols = key_cols + ["predicted_score", "rank_within_locus", "has_gene_embedding"]
    b = baseline_pred[use_cols].copy().rename(
        columns={
            "predicted_score": "baseline_score",
            "rank_within_locus": "baseline_rank",
            "has_gene_embedding": "has_embedding",
        }
    )
    p = pca_pred[use_cols].copy().rename(
        columns={
            "predicted_score": "pca_score",
            "rank_within_locus": "pca_rank",
        }
    )
    merged = b.merge(
        p[key_cols + ["pca_score", "pca_rank"]],
        on=key_cols,
        how="inner",
    )
    merged = _safe_to_numeric(
        merged,
        ["baseline_score", "pca_score", "baseline_rank", "pca_rank", "has_embedding"],
    )
    merged["has_embedding"] = merged["has_embedding"].fillna(0).astype(int)
    merged["delta_score_pca_minus_baseline"] = merged["pca_score"] - merged["baseline_score"]
    merged["abs_delta_score"] = merged["delta_score_pca_minus_baseline"].abs()
    return merged


def _plot_roc_auc_bar(roc_values: Dict[str, float], out_path: Path) -> None:
    labels = list(roc_values.keys())
    vals = [roc_values[k] for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    bars = ax.bar(x, vals, color=["#6baed6", "#3182bd", "#31a354", "#756bb1", "#2ca25f"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Reduced Baseline: ROC-AUC comparison")
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_delta_bar(selected: pd.DataFrame, title: str, out_path: Path) -> None:
    if selected.empty:
        return
    dd = selected.copy()
    label_col = "label"
    dd[label_col] = (
        dd["gwas_lead_variant_id"].fillna("").astype(str)
        + " | "
        + dd["gene_symbol"].fillna("").astype(str)
    )
    dd = dd.sort_values("delta_score_pca_minus_baseline", ascending=True, kind="stable")
    colors = ["#d95f0e" if v < 0 else "#238b45" for v in dd["delta_score_pca_minus_baseline"].to_numpy()]

    fig_h = max(4.0, 0.35 * len(dd))
    fig, ax = plt.subplots(figsize=(10.0, fig_h))
    y = np.arange(len(dd))
    ax.barh(y, dd["delta_score_pca_minus_baseline"].to_numpy(), color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(dd[label_col].tolist(), fontsize=8)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("delta_score = pca_score - baseline_score")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_baseline_feature_set(comparison_root: Path) -> List[str]:
    p = (
        comparison_root
        / "runs"
        / "l1"
        / "cv_lolo_gene_exclusion"
        / "mode_none"
        / "feature_lists.json"
    )
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [str(x) for x in data.get("baseline_feature_columns", [])]


def main() -> None:
    args = parse_args()
    root = args.comparison_root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load run summaries.
    noreg_none = _load_summary_metrics(root / "runs" / "noreg" / "cv_lolo_gene_exclusion" / "mode_none")
    l1_none = _load_summary_metrics(root / "runs" / "l1" / "cv_lolo_gene_exclusion" / "mode_none")
    l1_pca = _load_summary_metrics(root / "runs" / "l1" / "cv_lolo_gene_exclusion" / "mode_pca")
    l2_none = _load_summary_metrics(root / "runs" / "l2" / "cv_lolo_gene_exclusion" / "mode_none")
    l2_pca = _load_summary_metrics(root / "runs" / "l2" / "cv_lolo_gene_exclusion" / "mode_pca")

    roc_values = {
        "Baseline (No Reg)": float(noreg_none.get("mean_fold_roc_auc", float("nan"))),
        "Baseline L1": float(l1_none.get("mean_fold_roc_auc", float("nan"))),
        "Baseline + PCA L1": float(l1_pca.get("mean_fold_roc_auc", float("nan"))),
        "Baseline L2": float(l2_none.get("mean_fold_roc_auc", float("nan"))),
        "Baseline + PCA L2": float(l2_pca.get("mean_fold_roc_auc", float("nan"))),
    }
    _plot_roc_auc_bar(roc_values, out_dir / "roc_auc_baseline_vs_pca_l1_l2.png")

    selected_tables: Dict[str, pd.DataFrame] = {}
    for reg in ["l1", "l2"]:
        base_pred = _load_predictions(root / "runs" / reg / "cv_lolo_gene_exclusion" / "mode_none")
        pca_pred = _load_predictions(root / "runs" / reg / "cv_lolo_gene_exclusion" / "mode_pca")
        merged = _build_delta_table(base_pred, pca_pred)
        selected = (
            merged.sort_values("abs_delta_score", ascending=False, kind="stable")
            .head(int(args.top_n))
            .copy()
        )
        keep_cols = [
            "gwas_lead_variant_id",
            "gene_symbol",
            "has_embedding",
            "baseline_score",
            "pca_score",
            "delta_score_pca_minus_baseline",
            "baseline_rank",
            "pca_rank",
        ]
        for c in keep_cols:
            if c not in selected.columns:
                selected[c] = np.nan
        selected = selected[keep_cols]
        selected.to_csv(out_dir / f"score_difference_selected_genes_{reg}.csv", index=False)
        _plot_delta_bar(
            selected=selected,
            title=f"{reg.upper()}: score delta (PCA - baseline), top |delta| genes",
            out_path=out_dir / f"score_difference_baseline_vs_pca_{reg}.png",
        )
        selected_tables[reg] = selected

    baseline_features = _load_baseline_feature_set(root)

    lines: List[str] = []
    lines.append("# Reduced-Baseline Full Comparison Plot Summary")
    lines.append("")
    lines.append("## Baseline feature set used")
    lines.append("")
    if baseline_features:
        for feat in baseline_features:
            lines.append(f"- {feat}")
    else:
        lines.append("- (feature list unavailable)")
    lines.append("")
    lines.append("## Runs used")
    lines.append("")
    lines.append(f"- baseline_noreg: `{root / 'runs' / 'noreg' / 'cv_lolo_gene_exclusion' / 'mode_none'}`")
    lines.append(f"- baseline_l1: `{root / 'runs' / 'l1' / 'cv_lolo_gene_exclusion' / 'mode_none'}`")
    lines.append(f"- pca_l1: `{root / 'runs' / 'l1' / 'cv_lolo_gene_exclusion' / 'mode_pca'}`")
    lines.append(f"- baseline_l2: `{root / 'runs' / 'l2' / 'cv_lolo_gene_exclusion' / 'mode_none'}`")
    lines.append(f"- pca_l2: `{root / 'runs' / 'l2' / 'cv_lolo_gene_exclusion' / 'mode_pca'}`")
    lines.append("")
    lines.append("## ROC-AUC values used in bar plot")
    lines.append("")
    for k, v in roc_values.items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("## Gene selection method")
    lines.append("")
    lines.append("- Merged locus-gene rows by fold+locus+gene between baseline and PCA runs.")
    lines.append("- Computed `delta_score = pca_score - baseline_score`.")
    lines.append(f"- Selected top {int(args.top_n)} rows by largest absolute score difference (`|delta_score|`) per regularization.")
    lines.append("- Added `emb=1/0` using `has_gene_embedding` for interpretability.")
    lines.append("")
    for reg in ["l1", "l2"]:
        lines.append(f"## Selected genes ({reg.upper()})")
        lines.append("")
        tbl = selected_tables.get(reg, pd.DataFrame())
        if tbl.empty:
            lines.append("_(empty)_")
            lines.append("")
            continue
        lines.append(tbl.to_string(index=False))
        lines.append("")
    lines.append("## Key takeaway")
    lines.append("")
    lines.append(
        "- This rerun keeps embedding features and baseline setup unchanged and only refreshes outputs under the current root."
    )

    (out_dir / "no_network_plot_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] Wrote: {out_dir / 'no_network_plot_summary.md'}")


if __name__ == "__main__":
    main()


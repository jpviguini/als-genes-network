#!/usr/bin/env python3
"""Interpretability follow-up for no-network baseline vs baseline+PCA experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COMPARISON_ROOT = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/full_comparison/"
    "reduced_baseline_full_comparison_20260406"
)
DEFAULT_NEW_MODE_ROOT = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/locus_gene_ranker_baseline_then_pca_20260407"
)
DEFAULT_OUT_DIR = DEFAULT_COMPARISON_ROOT / "interpretation_followup_20260407"
DEFAULT_FOCUS_GENES = ["DDIT3", "B4GALNT1", "ERGIC1", "KIF5A"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate supervisor-focused interpretation outputs for no-network runs."
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=DEFAULT_COMPARISON_ROOT,
        help="Folder containing reduced_baseline_full_comparison_20260406 outputs.",
    )
    parser.add_argument(
        "--new-mode-root",
        type=Path,
        default=DEFAULT_NEW_MODE_ROOT,
        help="Folder containing mode_baseline_then_pca outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output folder for interpretation follow-up artifacts.",
    )
    parser.add_argument(
        "--focus-genes",
        type=str,
        default=",".join(DEFAULT_FOCUS_GENES),
        help="Comma-separated genes to prioritize in locus-level report.",
    )
    parser.add_argument(
        "--max-auto-loci",
        type=int,
        default=8,
        help="Max automatically selected loci (focus loci are always included).",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def df_to_markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(empty)_"
    dd = df.copy()
    headers = [str(c) for c in dd.columns.tolist()]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in dd.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                if np.isfinite(val):
                    sval = f"{val:.6g}"
                else:
                    sval = "nan"
            else:
                sval = str(val)
            sval = sval.replace("\n", " ").replace("|", "/")
            vals.append(sval)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def sanitize_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def load_mode_bundle(root: Path, penalty: str, mode: str) -> Dict[str, object]:
    mode_dir = root / "runs" / penalty / "cv_lolo_gene_exclusion" / f"mode_{mode}"
    if not mode_dir.exists():
        raise FileNotFoundError(f"Missing mode directory: {mode_dir}")

    summary_path = mode_dir / "summary_metrics.json"
    feature_lists_path = mode_dir / "feature_lists.json"
    coef_path = mode_dir / "coefficient_table.csv"
    pred_path = mode_dir / "all_ranked_predictions.csv"
    pca_artifacts_path = mode_dir / "pca_transformer_artifacts.npz"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(feature_lists_path, "r", encoding="utf-8") as f:
        feature_lists = json.load(f)
    coef_df = pd.read_csv(coef_path)
    pred_df = pd.read_csv(pred_path)
    pca_artifacts: Dict[str, np.ndarray] = {}
    if pca_artifacts_path.exists():
        with np.load(pca_artifacts_path, allow_pickle=True) as data:
            pca_artifacts = {k: data[k] for k in data.files}

    return {
        "mode_dir": mode_dir,
        "summary": summary,
        "feature_lists": feature_lists,
        "coefficients": coef_df,
        "predictions": pred_df,
        "pca_artifacts": pca_artifacts,
    }


def load_new_mode_bundle(root: Path) -> Dict[str, object]:
    mode_dir = root / "cv_lolo_gene_exclusion" / "mode_baseline_then_pca"
    if not mode_dir.exists():
        raise FileNotFoundError(f"Missing new-mode directory: {mode_dir}")

    with open(mode_dir / "summary_metrics.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(mode_dir / "feature_lists.json", "r", encoding="utf-8") as f:
        feature_lists = json.load(f)
    coef_df = pd.read_csv(mode_dir / "coefficient_table.csv")
    pred_df = pd.read_csv(mode_dir / "all_ranked_predictions.csv")
    model_params = {}
    with open(mode_dir / "model_parameters.json", "r", encoding="utf-8") as f:
        model_params = json.load(f)

    return {
        "mode_dir": mode_dir,
        "summary": summary,
        "feature_lists": feature_lists,
        "coefficients": coef_df,
        "predictions": pred_df,
        "model_params": model_params,
    }


def make_gene_key(df: pd.DataFrame) -> pd.Series:
    gid = df.get("gene_id", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.strip()
    gsym = df.get("gene_symbol", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.strip()
    key = gid.where(gid != "", gsym)
    missing = key == ""
    if missing.any():
        key.loc[missing] = [f"row_{idx}" for idx in key.index[missing]]
    return key


def build_locus_comparison_tables(
    baseline_pred: pd.DataFrame,
    pca_pred: pd.DataFrame,
    focus_genes: Sequence[str],
    max_auto_loci: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, List[str]]]:
    base = baseline_pred.copy()
    pca = pca_pred.copy()
    base["gene_key"] = make_gene_key(base)
    pca["gene_key"] = make_gene_key(pca)

    key_cols = ["gwas_study_locus_id", "gene_key"]
    use_cols = [
        "gwas_study_locus_id",
        "gwas_lead_variant_id",
        "gene_symbol",
        "gene_key",
        "rank_within_locus",
        "predicted_score",
        "has_gene_embedding",
    ]
    for col in use_cols:
        if col not in base.columns:
            base[col] = np.nan
        if col not in pca.columns:
            pca[col] = np.nan

    bb = base[use_cols].rename(
        columns={
            "rank_within_locus": "rank_baseline",
            "predicted_score": "score_baseline",
            "has_gene_embedding": "has_embedding",
        }
    )
    pp = pca[use_cols].rename(
        columns={
            "rank_within_locus": "rank_pca",
            "predicted_score": "score_pca",
        }
    )
    merged = bb.merge(
        pp[["gwas_study_locus_id", "gene_key", "rank_pca", "score_pca"]],
        on=key_cols,
        how="inner",
    )
    merged["rank_baseline"] = pd.to_numeric(merged["rank_baseline"], errors="coerce")
    merged["rank_pca"] = pd.to_numeric(merged["rank_pca"], errors="coerce")
    merged["score_baseline"] = pd.to_numeric(merged["score_baseline"], errors="coerce")
    merged["score_pca"] = pd.to_numeric(merged["score_pca"], errors="coerce")
    merged["has_embedding"] = pd.to_numeric(merged["has_embedding"], errors="coerce").fillna(0).astype(int)
    merged["delta_rank"] = merged["rank_pca"] - merged["rank_baseline"]
    merged["delta_score"] = merged["score_pca"] - merged["score_baseline"]
    merged["rank_gain"] = merged["rank_baseline"] - merged["rank_pca"]

    focus_upper = {g.strip().upper() for g in focus_genes if g.strip()}
    focus_presence: Dict[str, List[str]] = {g: [] for g in sorted(focus_upper)}

    locus_rows: List[Dict[str, object]] = []
    candidate_loci: List[Tuple[str, float]] = []
    focus_loci: List[str] = []

    for locus_id, d in merged.groupby("gwas_study_locus_id", sort=True):
        d = d.copy()
        d["gene_symbol"] = d["gene_symbol"].fillna("").astype(str)

        gene_symbols_upper = set(d["gene_symbol"].str.upper().tolist())
        locus_has_focus = False
        for fg in focus_upper:
            if fg in gene_symbols_upper:
                locus_has_focus = True
                focus_presence[fg].append(str(locus_id))
        if locus_has_focus:
            focus_loci.append(str(locus_id))

        d_base_top = d.sort_values("rank_baseline", ascending=True, kind="stable")
        d_pca_top = d.sort_values("rank_pca", ascending=True, kind="stable")
        baseline_top5 = d_base_top.head(5)["gene_symbol"].tolist()
        pca_top5 = d_pca_top.head(5)["gene_symbol"].tolist()

        if d_base_top.empty:
            continue
        lead_row = d_base_top.iloc[0]
        lead_gene = str(lead_row["gene_symbol"])
        lead_rank_delta = float(lead_row["delta_rank"])
        lead_score_delta = float(lead_row["delta_score"])

        promoted_candidates = d[d["gene_symbol"] != lead_gene].copy()
        promoted_candidates = promoted_candidates.sort_values(
            ["rank_gain", "delta_score"],
            ascending=[False, False],
            kind="stable",
        )
        promoted = promoted_candidates.iloc[0] if not promoted_candidates.empty else None

        if promoted is not None:
            promoted_gene = str(promoted["gene_symbol"])
            promoted_rank_gain = float(promoted["rank_gain"])
            promoted_delta_score = float(promoted["delta_score"])
            promoted_rank_baseline = float(promoted["rank_baseline"])
            promoted_rank_pca = float(promoted["rank_pca"])
            promoted_has_embedding = int(promoted["has_embedding"])
            promoted_score_baseline = float(promoted["score_baseline"])
            promoted_score_pca = float(promoted["score_pca"])
        else:
            promoted_gene = ""
            promoted_rank_gain = 0.0
            promoted_delta_score = 0.0
            promoted_rank_baseline = float("nan")
            promoted_rank_pca = float("nan")
            promoted_has_embedding = 0
            promoted_score_baseline = float("nan")
            promoted_score_pca = float("nan")

        moved_up_genes = (
            d[d["delta_rank"] < 0]
            .sort_values("delta_rank", ascending=True, kind="stable")["gene_symbol"]
            .tolist()
        )
        moved_down_genes = (
            d[d["delta_rank"] > 0]
            .sort_values("delta_rank", ascending=False, kind="stable")["gene_symbol"]
            .tolist()
        )

        qualifies = bool(
            (lead_rank_delta >= 1.0 or lead_score_delta <= -0.01)
            and (promoted_rank_gain >= 1.0)
        )
        effect_strength = (
            max(0.0, lead_rank_delta)
            + max(0.0, promoted_rank_gain)
            + max(0.0, -lead_score_delta * 20.0)
        )
        if qualifies:
            candidate_loci.append((str(locus_id), float(effect_strength)))

        locus_rows.append(
            {
                "locus_id": str(locus_id),
                "lead_variant_id": str(lead_row.get("gwas_lead_variant_id", "")),
                "baseline_lead_gene": lead_gene,
                "baseline_lead_has_embedding": int(lead_row["has_embedding"]),
                "baseline_lead_rank_baseline": float(lead_row["rank_baseline"]),
                "baseline_lead_rank_pca": float(lead_row["rank_pca"]),
                "baseline_lead_delta_rank": lead_rank_delta,
                "baseline_lead_score_baseline": float(lead_row["score_baseline"]),
                "baseline_lead_score_pca": float(lead_row["score_pca"]),
                "baseline_lead_delta_score": lead_score_delta,
                "promoted_gene": promoted_gene,
                "promoted_gene_has_embedding": promoted_has_embedding,
                "promoted_rank_baseline": promoted_rank_baseline,
                "promoted_rank_pca": promoted_rank_pca,
                "promoted_delta_rank": -promoted_rank_gain if np.isfinite(promoted_rank_gain) else float("nan"),
                "promoted_rank_gain": promoted_rank_gain,
                "promoted_score_baseline": promoted_score_baseline,
                "promoted_score_pca": promoted_score_pca,
                "promoted_delta_score": promoted_delta_score,
                "baseline_top5": " | ".join(baseline_top5),
                "pca_top5": " | ".join(pca_top5),
                "moved_up_genes": " | ".join(moved_up_genes),
                "moved_down_genes": " | ".join(moved_down_genes),
                "focus_gene_present": int(locus_has_focus),
                "qualifies_alternative_story": int(qualifies),
                "effect_strength": float(effect_strength),
            }
        )

    locus_summary = pd.DataFrame(locus_rows).sort_values(
        ["qualifies_alternative_story", "effect_strength"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)

    selected: List[str] = []
    for locus in focus_loci:
        if locus not in selected:
            selected.append(locus)
    sorted_candidates = sorted(candidate_loci, key=lambda t: t[1], reverse=True)
    for locus, _ in sorted_candidates:
        if locus not in selected:
            selected.append(locus)
        if len([x for x in selected if x not in focus_loci]) >= int(max_auto_loci):
            break
    if not selected and not locus_summary.empty:
        selected = locus_summary.head(min(3, len(locus_summary)))["locus_id"].tolist()

    all_long = merged.copy()
    all_long = all_long.rename(columns={"gwas_study_locus_id": "locus_id", "gwas_lead_variant_id": "lead_variant_id"})
    all_long = all_long[
        [
            "locus_id",
            "lead_variant_id",
            "gene_symbol",
            "gene_key",
            "has_embedding",
            "rank_baseline",
            "rank_pca",
            "delta_rank",
            "score_baseline",
            "score_pca",
            "delta_score",
        ]
    ].copy()
    all_long["movement"] = np.where(
        all_long["delta_rank"] < 0,
        "up",
        np.where(all_long["delta_rank"] > 0, "down", "unchanged"),
    )

    return locus_summary, all_long, selected, focus_presence


def build_top1_replacement_summary(locus_long: pd.DataFrame, selected_loci: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    selected_set = {str(x) for x in selected_loci}
    dsel = locus_long[locus_long["locus_id"].astype(str).isin(selected_set)].copy()
    for locus_id, d in dsel.groupby("locus_id", sort=True):
        d = d.copy()
        if d.empty:
            continue
        d_base = d.sort_values(["rank_baseline", "score_baseline"], ascending=[True, False], kind="stable")
        d_pca = d.sort_values(["rank_pca", "score_pca"], ascending=[True, False], kind="stable")
        top_base = d_base.iloc[0]
        top_pca = d_pca.iloc[0]
        baseline_gene = str(top_base.get("gene_symbol", ""))
        pca_gene = str(top_pca.get("gene_symbol", ""))
        rows.append(
            {
                "locus_id": str(locus_id),
                "lead_variant_id": str(top_base.get("lead_variant_id", "")),
                "baseline_top1_gene": baseline_gene,
                "pca_top1_gene": pca_gene,
                "top1_changed": int(baseline_gene != pca_gene),
                "baseline_top1_has_embedding": int(pd.to_numeric(top_base.get("has_embedding", 0), errors="coerce")),
                "pca_top1_has_embedding": int(pd.to_numeric(top_pca.get("has_embedding", 0), errors="coerce")),
            }
        )
    out = pd.DataFrame(rows).sort_values(["top1_changed", "locus_id"], ascending=[False, True], kind="stable")
    return out.reset_index(drop=True)


def plot_top1_replacement_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    d = summary_df.copy().reset_index(drop=True)
    y = np.arange(len(d))
    fig_h = max(4.0, 0.6 * len(d) + 1.5)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))

    x_baseline = 0.0
    x_pca = 1.0
    for i, row in enumerate(d.itertuples(index=False)):
        changed = int(row.top1_changed) == 1
        color = "#dc2626" if changed else "#6b7280"
        ax.plot([x_baseline, x_pca], [y[i], y[i]], color=color, linewidth=2.0, alpha=0.9)

        marker_base = "o" if int(row.baseline_top1_has_embedding) == 1 else "s"
        marker_pca = "o" if int(row.pca_top1_has_embedding) == 1 else "s"
        ax.scatter([x_baseline], [y[i]], marker=marker_base, s=70, facecolors="white", edgecolors="#111827", linewidth=1.2, zorder=3)
        ax.scatter([x_pca], [y[i]], marker=marker_pca, s=70, facecolors=color, edgecolors="#111827", linewidth=0.8, zorder=4)

        ax.text(x_baseline - 0.02, y[i], str(row.baseline_top1_gene), ha="right", va="center", fontsize=8)
        ax.text(x_pca + 0.02, y[i], str(row.pca_top1_gene), ha="left", va="center", fontsize=8)
        ax.text(1.23, y[i], "changed" if changed else "same", color=color, ha="left", va="center", fontsize=8)

    ylabels = [
        f"{str(r.locus_id)} | {str(r.lead_variant_id)[:20]}"
        for r in d.itertuples(index=False)
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xticks([x_baseline, x_pca])
    ax.set_xticklabels(["Baseline top-1", "PCA top-1"], fontsize=9)
    ax.set_xlim(-0.35, 1.6)
    ax.set_title("Top-1 Gene Replacement Across Loci")
    ax.grid(axis="x", alpha=0.22)
    ax.invert_yaxis()

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#dc2626", lw=2.0, label="Top-1 changed"),
        Line2D([0], [0], color="#6b7280", lw=2.0, label="Top-1 unchanged"),
        Line2D([0], [0], marker="o", color="#374151", markerfacecolor="white", lw=0, label="emb=1"),
        Line2D([0], [0], marker="s", color="#374151", markerfacecolor="white", lw=0, label="emb=0"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def choose_locus_plot_genes(d: pd.DataFrame) -> pd.DataFrame:
    a = d.sort_values("rank_baseline", ascending=True, kind="stable").head(5)
    b = d.sort_values("rank_pca", ascending=True, kind="stable").head(5)
    gene_set = set(a["gene_key"].tolist()) | set(b["gene_key"].tolist())
    out = d[d["gene_key"].isin(gene_set)].copy()
    out = out.sort_values(["rank_baseline", "rank_pca"], ascending=[True, True], kind="stable").reset_index(drop=True)
    return out


def plot_locus_rank_comparison(locus_df: pd.DataFrame, locus_id: str, lead_variant: str, out_path: Path) -> None:
    d = choose_locus_plot_genes(locus_df)
    d = d.copy()
    d["label"] = d["gene_symbol"].astype(str) + " (emb=" + d["has_embedding"].astype(int).astype(str) + ")"
    y = np.arange(len(d))[::-1]

    fig_h = max(4.0, 0.45 * len(d) + 1.6)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))

    for i, row in enumerate(d.itertuples(index=False)):
        yy = y[i]
        r0 = float(row.rank_baseline)
        r1 = float(row.rank_pca)
        if r1 < r0:
            color = "#16a34a"
        elif r1 > r0:
            color = "#dc2626"
        else:
            color = "#6b7280"
        marker = "o" if int(row.has_embedding) == 1 else "s"
        ax.plot([r0, r1], [yy, yy], color=color, linewidth=2.1, alpha=0.95)
        ax.scatter([r0], [yy], marker=marker, s=62, facecolors="white", edgecolors=color, linewidth=1.7, zorder=3)
        ax.scatter([r1], [yy], marker=marker, s=62, facecolors=color, edgecolors="#111827", linewidth=0.8, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(d["label"].tolist(), fontsize=9)
    ax.set_xlabel("Rank within locus (lower is better)")
    ax.set_title(f"Locus {locus_id} | lead variant: {lead_variant}\nBaseline vs Baseline+PCA rank comparison")
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlim(0.5, max(float(d["rank_baseline"].max()), float(d["rank_pca"].max())) + 0.75)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#16a34a", lw=2.1, label="Moved up in PCA"),
        Line2D([0], [0], color="#dc2626", lw=2.1, label="Moved down in PCA"),
        Line2D([0], [0], marker="o", color="#374151", markerfacecolor="white", lw=0, label="emb=1"),
        Line2D([0], [0], marker="s", color="#374151", markerfacecolor="white", lw=0, label="emb=0"),
    ]
    ax.legend(handles=legend_elems, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_locus_score_comparison(locus_df: pd.DataFrame, locus_id: str, lead_variant: str, out_path: Path) -> None:
    d = choose_locus_plot_genes(locus_df)
    d = d.copy()
    d["label"] = d["gene_symbol"].astype(str) + " (emb=" + d["has_embedding"].astype(int).astype(str) + ")"
    y = np.arange(len(d))[::-1]

    fig_h = max(4.0, 0.45 * len(d) + 1.6)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    for i, row in enumerate(d.itertuples(index=False)):
        yy = y[i]
        s0 = float(row.score_baseline)
        s1 = float(row.score_pca)
        if s1 > s0:
            color = "#16a34a"
        elif s1 < s0:
            color = "#dc2626"
        else:
            color = "#6b7280"
        marker = "o" if int(row.has_embedding) == 1 else "s"
        ax.plot([s0, s1], [yy, yy], color=color, linewidth=2.1, alpha=0.95)
        ax.scatter([s0], [yy], marker=marker, s=62, facecolors="white", edgecolors=color, linewidth=1.7, zorder=3)
        ax.scatter([s1], [yy], marker=marker, s=62, facecolors=color, edgecolors="#111827", linewidth=0.8, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(d["label"].tolist(), fontsize=9)
    ax.set_xlabel("Predicted score")
    ax.set_title(f"Locus {locus_id} | lead variant: {lead_variant}\nBaseline vs Baseline+PCA score comparison")
    ax.grid(axis="x", alpha=0.22)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#16a34a", lw=2.1, label="Upweighted in PCA"),
        Line2D([0], [0], color="#dc2626", lw=2.1, label="Downweighted in PCA"),
        Line2D([0], [0], marker="o", color="#374151", markerfacecolor="white", lw=0, label="emb=1"),
        Line2D([0], [0], marker="s", color="#374151", markerfacecolor="white", lw=0, label="emb=0"),
    ]
    ax.legend(handles=legend_elems, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_locus_summary_across_loci(summary_df: pd.DataFrame, out_path: Path) -> None:
    d = summary_df.copy()
    d = d[d["promoted_gene"].fillna("").astype(str) != ""].copy()
    if d.empty:
        return
    d = d.sort_values("effect_strength", ascending=False, kind="stable").reset_index(drop=True)

    y = np.arange(len(d))
    fig_h = max(4.6, 0.55 * len(d) + 1.4)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))

    for i, row in enumerate(d.itertuples(index=False)):
        y_lead = y[i] + 0.15
        y_prom = y[i] - 0.15
        lb0 = float(row.baseline_lead_score_baseline)
        lb1 = float(row.baseline_lead_score_pca)
        pr0 = float(row.promoted_score_baseline)
        pr1 = float(row.promoted_score_pca)

        ax.plot([lb0, lb1], [y_lead, y_lead], color="#b91c1c", linewidth=2.0, alpha=0.95)
        ax.scatter([lb0], [y_lead], color="white", edgecolors="#b91c1c", s=42, zorder=3)
        ax.scatter([lb1], [y_lead], color="#b91c1c", edgecolors="#111827", s=42, zorder=4)

        ax.plot([pr0, pr1], [y_prom, y_prom], color="#15803d", linewidth=2.0, alpha=0.95)
        ax.scatter([pr0], [y_prom], color="white", edgecolors="#15803d", s=42, zorder=3)
        ax.scatter([pr1], [y_prom], color="#15803d", edgecolors="#111827", s=42, zorder=4)

        ax.text(
            max(lb0, lb1, pr0, pr1) + 0.01,
            y[i],
            f"{row.baseline_lead_gene} -> {row.promoted_gene}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ylabels = [
        f"{str(x.locus_id)} | {str(x.lead_variant_id)[:20]}"
        for x in d.itertuples(index=False)
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Predicted score")
    ax.set_title("Alternative-Gene Summary Across Loci\nBaseline-leading downweighted gene vs promoted alternative")
    ax.grid(axis="x", alpha=0.22)
    ax.invert_yaxis()

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#b91c1c", lw=2.0, label="Baseline-leading gene"),
        Line2D([0], [0], color="#15803d", lw=2.0, label="Promoted alternative"),
        Line2D([0], [0], marker="o", color="#374151", markerfacecolor="white", lw=0, label="Baseline score"),
        Line2D([0], [0], marker="o", color="#374151", markerfacecolor="#374151", lw=0, label="PCA score"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_locus_markdown_report(
    out_path: Path,
    locus_summary: pd.DataFrame,
    selected_loci: Sequence[str],
    focus_presence: Dict[str, List[str]],
    top1_summary: pd.DataFrame,
    top1_summary_plot: Optional[Path],
    rank_plot_map: Dict[str, Path],
    score_plot_map: Dict[str, Path],
    summary_plot: Optional[Path],
    relative_root: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Locus-Level Alternative Gene Analysis")
    lines.append("")
    lines.append("## Focus genes presence")
    lines.append("")
    for gene, loci in sorted(focus_presence.items()):
        if loci:
            lines.append(f"- `{gene}` present in loci: {', '.join(sorted(loci))}")
        else:
            lines.append(f"- `{gene}` not found in current output table.")
    lines.append("")

    if locus_summary.empty:
        lines.append("No loci available for comparison.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    cols_show = [
        "locus_id",
        "lead_variant_id",
        "baseline_lead_gene",
        "promoted_gene",
        "baseline_lead_delta_rank",
        "baseline_lead_delta_score",
        "promoted_rank_gain",
        "promoted_delta_score",
        "focus_gene_present",
    ]
    lines.append("## Selected locus summary")
    lines.append("")
    lines.append(df_to_markdown_table(locus_summary[locus_summary["locus_id"].isin(selected_loci)][cols_show]))
    lines.append("")

    if top1_summary is not None and not top1_summary.empty:
        lines.append("## Top-1 Replacement Summary")
        lines.append("")
        lines.append(df_to_markdown_table(top1_summary))
        lines.append("")
        if top1_summary_plot is not None and top1_summary_plot.exists():
            lines.append(f"![Top-1 replacement summary]({top1_summary_plot.relative_to(relative_root).as_posix()})")
            lines.append("")

    if summary_plot is not None and summary_plot.exists():
        rel = summary_plot.relative_to(relative_root)
        lines.append("## Cross-locus summary plot")
        lines.append("")
        lines.append(f"![Alternative gene summary]({rel.as_posix()})")
        lines.append("")

    lines.append("## Per-locus plots")
    lines.append("")
    lines.append(
        "Short note: genes shown in per-locus rank/score plots are the union of top-5 baseline genes and top-5 baseline+PCA genes."
    )
    lines.append("")
    for locus in selected_loci:
        row = locus_summary[locus_summary["locus_id"] == locus]
        if row.empty:
            continue
        rr = row.iloc[0]
        lines.append(f"### Locus `{locus}`")
        lines.append("")
        lines.append(f"- Lead variant: `{rr['lead_variant_id']}`")
        lines.append(f"- Baseline top5: {rr['baseline_top5']}")
        lines.append(f"- Baseline+PCA top5: {rr['pca_top5']}")
        lines.append(f"- Moved up: {rr['moved_up_genes']}")
        lines.append(f"- Moved down: {rr['moved_down_genes']}")
        top1_row = top1_summary[top1_summary["locus_id"] == locus] if top1_summary is not None else pd.DataFrame()
        if top1_row is not None and not top1_row.empty:
            tr = top1_row.iloc[0]
            changed = int(tr["top1_changed"]) == 1
            lines.append(
                f"- Top-1 changed: `{int(tr['top1_changed'])}` ({tr['baseline_top1_gene']} -> {tr['pca_top1_gene']})"
                if changed
                else f"- Top-1 changed: `{int(tr['top1_changed'])}` (same top gene: {tr['baseline_top1_gene']})"
            )
        rpath = rank_plot_map.get(locus)
        spath = score_plot_map.get(locus)
        if rpath is not None and rpath.exists():
            lines.append(f"![Rank comparison]({rpath.relative_to(relative_root).as_posix()})")
        if spath is not None and spath.exists():
            lines.append(f"![Score comparison]({spath.relative_to(relative_root).as_posix()})")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def reconstruct_pca_features(pred_df: pd.DataFrame, pca_artifacts: Dict[str, np.ndarray]) -> pd.DataFrame:
    required = [
        "embedding_feature_names",
        "pca_feature_names",
        "emb_scaler_mean",
        "emb_scaler_scale",
        "pca_components",
        "pca_mean",
    ]
    missing = [k for k in required if k not in pca_artifacts]
    if missing:
        raise ValueError(f"PCA artifacts missing keys: {missing}")

    emb_cols = [str(x) for x in np.asarray(pca_artifacts["embedding_feature_names"]).tolist()]
    pca_cols = [str(x) for x in np.asarray(pca_artifacts["pca_feature_names"]).tolist()]
    x_raw = pred_df[emb_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    emb_mean = np.asarray(pca_artifacts["emb_scaler_mean"], dtype=float).reshape(-1)
    emb_scale = np.asarray(pca_artifacts["emb_scaler_scale"], dtype=float).reshape(-1)
    pca_components = np.asarray(pca_artifacts["pca_components"], dtype=float)
    pca_mean = np.asarray(pca_artifacts["pca_mean"], dtype=float).reshape(-1)

    safe_scale = np.where(np.abs(emb_scale) > 0, emb_scale, 1.0)
    x_scaled = (x_raw - emb_mean) / safe_scale
    x_centered = x_scaled - pca_mean
    x_pca = x_centered @ pca_components.T
    return pd.DataFrame(x_pca, columns=pca_cols, index=pred_df.index)


def correlation_pairs_table(corr_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    cols = corr_df.columns.tolist()
    values = corr_df.to_numpy(dtype=float)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = float(values[i, j])
            rows.append(
                {
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                }
            )
    out = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False, kind="stable").reset_index(drop=True)
    return out


def plot_correlation_heatmap(corr_df: pd.DataFrame, out_path: Path) -> None:
    labels = []
    for c in corr_df.columns.tolist():
        if c.startswith("emb_pca_"):
            labels.append(c.replace("emb_pca_", "PC"))
        elif c == "dist_variant_to_gene_kb":
            labels.append("dist_to_gene_kb")
        elif c == "colocalisation_h4_max":
            labels.append("coloc_h4_max")
        elif c == "hpa_brain_expression_value":
            labels.append("hpa_brain_expr")
        elif c == "hpa_muscle_expression_value":
            labels.append("hpa_muscle_expr")
        else:
            labels.append(c)

    fig, ax = plt.subplots(figsize=(12.5, 10.5))
    im = ax.imshow(corr_df.to_numpy(dtype=float), cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Feature Correlation Matrix (Baseline + PCA Inputs)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_correlation_markdown(
    out_path: Path,
    corr_pairs: pd.DataFrame,
    surviving_pca: Sequence[str],
) -> None:
    lines: List[str] = []
    lines.append("# Feature Correlation Analysis")
    lines.append("")
    lines.append("## Surviving L1 PCA components")
    lines.append("")
    if surviving_pca:
        lines.append("- " + ", ".join(sorted(surviving_pca)))
    else:
        lines.append("- No non-zero PCA components in L1 coefficient table.")
    lines.append("")

    target_features = [
        "dist_variant_to_gene_kb",
        "colocalisation_h4_max",
        "hpa_brain_expression_value",
        "hpa_muscle_expression_value",
    ]
    lines.append("## Strongest correlations involving key features")
    lines.append("")
    for feat in target_features:
        sub = corr_pairs[
            (corr_pairs["feature_1"] == feat) | (corr_pairs["feature_2"] == feat)
        ].copy()
        sub = sub.sort_values("abs_correlation", ascending=False, kind="stable").head(8)
        if sub.empty:
            lines.append(f"- `{feat}`: no pairs found in matrix.")
            continue
        lines.append(f"- `{feat}`:")
        for row in sub.itertuples(index=False):
            other = row.feature_2 if row.feature_1 == feat else row.feature_1
            lines.append(f"  - with `{other}`: corr={float(row.correlation):.4f}")
    lines.append("")

    dist_sub = corr_pairs[
        (corr_pairs["feature_1"] == "dist_variant_to_gene_kb")
        | (corr_pairs["feature_2"] == "dist_variant_to_gene_kb")
    ].sort_values("abs_correlation", ascending=False, kind="stable")
    max_abs_dist = float(dist_sub["abs_correlation"].max()) if not dist_sub.empty else float("nan")
    lines.append("## Interpretation note")
    lines.append("")
    lines.append(
        f"- Maximum |correlation| involving `dist_variant_to_gene_kb` is {max_abs_dist:.4f}. "
        "High values suggest potential redundancy in multivariable L1 fitting."
    )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def sign_of(value: float, tol: float = 1e-12) -> str:
    if value > tol:
        return "+"
    if value < -tol:
        return "-"
    return "0"


def build_sign_consistency_table(
    coef_baseline_l1: pd.DataFrame,
    coef_pca_l1: pd.DataFrame,
    coef_baseline_l2: pd.DataFrame,
    coef_pca_l2: pd.DataFrame,
    baseline_features: Sequence[str],
) -> pd.DataFrame:
    def coef_map(df: pd.DataFrame) -> Dict[str, float]:
        dd = df.copy()
        dd["feature"] = dd["feature"].astype(str)
        dd["coefficient"] = pd.to_numeric(dd["coefficient"], errors="coerce")
        return {str(r.feature): float(r.coefficient) for r in dd.itertuples(index=False)}

    m_bl1 = coef_map(coef_baseline_l1)
    m_pl1 = coef_map(coef_pca_l1)
    m_bl2 = coef_map(coef_baseline_l2)
    m_pl2 = coef_map(coef_pca_l2)

    rows: List[Dict[str, object]] = []
    for feat in baseline_features:
        c1 = float(m_bl1.get(feat, 0.0))
        c2 = float(m_pl1.get(feat, 0.0))
        c3 = float(m_bl2.get(feat, 0.0))
        c4 = float(m_pl2.get(feat, 0.0))
        s1 = sign_of(c1)
        s2 = sign_of(c2)
        s3 = sign_of(c3)
        s4 = sign_of(c4)
        non_zero_signs = [s for s in [s1, s2, s3, s4] if s != "0"]
        sign_stable = int(len(set(non_zero_signs)) <= 1 and len(non_zero_signs) == 4)
        sign_flip_any = int(len(set(non_zero_signs)) > 1)
        rows.append(
            {
                "feature": feat,
                "coef_baseline_l1": c1,
                "sign_baseline_l1": s1,
                "coef_baseline_pca_l1": c2,
                "sign_baseline_pca_l1": s2,
                "coef_baseline_l2": c3,
                "sign_baseline_l2": s3,
                "coef_baseline_pca_l2": c4,
                "sign_baseline_pca_l2": s4,
                "sign_stable_all_four": sign_stable,
                "sign_flip_any": sign_flip_any,
                "flip_within_l1_none_vs_pca": int(s1 != s2),
                "flip_within_l2_none_vs_pca": int(s3 != s4),
            }
        )
    return pd.DataFrame(rows)


def plot_coefficient_sign_comparison(sign_df: pd.DataFrame, out_path: Path) -> None:
    if sign_df.empty:
        return
    model_cols = [
        ("coef_baseline_l1", "baseline_l1", "#1d4ed8"),
        ("coef_baseline_pca_l1", "baseline_pca_l1", "#059669"),
        ("coef_baseline_l2", "baseline_l2", "#dc2626"),
        ("coef_baseline_pca_l2", "baseline_pca_l2", "#7c3aed"),
    ]
    y_base = np.arange(len(sign_df))[::-1]
    offsets = [-0.24, -0.08, 0.08, 0.24]

    fig_h = max(3.6, 0.8 * len(sign_df) + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    for (col, label, color), off in zip(model_cols, offsets):
        vals = pd.to_numeric(sign_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.scatter(vals, y_base + off, s=60, color=color, alpha=0.9, label=label)
    ax.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.8)
    ax.set_yticks(y_base)
    ax.set_yticklabels(sign_df["feature"].tolist(), fontsize=9)
    ax.set_xlabel("Coefficient value")
    ax.set_title("Coefficient Comparison Across Models (Shared Baseline Features)")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_sign_markdown(out_path: Path, sign_df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Coefficient Sign Consistency")
    lines.append("")
    if sign_df.empty:
        lines.append("No shared baseline features available.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    stable = sign_df[sign_df["sign_stable_all_four"] == 1]["feature"].tolist()
    flipped = sign_df[sign_df["sign_flip_any"] == 1]["feature"].tolist()

    lines.append("## Sign-stable features across all four models")
    lines.append("")
    if stable:
        lines.append("- " + ", ".join(stable))
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Features with at least one sign flip")
    lines.append("")
    if flipped:
        lines.append("- " + ", ".join(flipped))
    else:
        lines.append("- None")
    lines.append("")

    show_cols = [
        "feature",
        "coef_baseline_l1",
        "coef_baseline_pca_l1",
        "coef_baseline_l2",
        "coef_baseline_pca_l2",
        "sign_baseline_l1",
        "sign_baseline_pca_l1",
        "sign_baseline_l2",
        "sign_baseline_pca_l2",
        "sign_stable_all_four",
    ]
    lines.append("## Coefficient/sign table")
    lines.append("")
    lines.append(df_to_markdown_table(sign_df[show_cols]))
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_model_comparison_summary(
    out_csv: Path,
    out_md: Path,
    baseline_l1_summary: Dict[str, object],
    pca_l1_summary: Dict[str, object],
    new_mode_summary: Dict[str, object],
    new_mode_coef: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "model": "baseline_l1",
            "mean_fold_pr_auc": float(baseline_l1_summary.get("mean_fold_pr_auc", float("nan"))),
            "mean_fold_roc_auc": float(baseline_l1_summary.get("mean_fold_roc_auc", float("nan"))),
            "mean_recall_at_1": float(baseline_l1_summary.get("mean_recall_at_1", float("nan"))),
            "mean_recall_at_3": float(baseline_l1_summary.get("mean_recall_at_3", float("nan"))),
            "mean_mrr": float(baseline_l1_summary.get("mean_mrr", float("nan"))),
        },
        {
            "model": "baseline_pca_l1",
            "mean_fold_pr_auc": float(pca_l1_summary.get("mean_fold_pr_auc", float("nan"))),
            "mean_fold_roc_auc": float(pca_l1_summary.get("mean_fold_roc_auc", float("nan"))),
            "mean_recall_at_1": float(pca_l1_summary.get("mean_recall_at_1", float("nan"))),
            "mean_recall_at_3": float(pca_l1_summary.get("mean_recall_at_3", float("nan"))),
            "mean_mrr": float(pca_l1_summary.get("mean_mrr", float("nan"))),
        },
        {
            "model": "baseline_then_pca_l1",
            "mean_fold_pr_auc": float(new_mode_summary.get("mean_fold_pr_auc", float("nan"))),
            "mean_fold_roc_auc": float(new_mode_summary.get("mean_fold_roc_auc", float("nan"))),
            "mean_recall_at_1": float(new_mode_summary.get("mean_recall_at_1", float("nan"))),
            "mean_recall_at_3": float(new_mode_summary.get("mean_recall_at_3", float("nan"))),
            "mean_mrr": float(new_mode_summary.get("mean_mrr", float("nan"))),
        },
    ]
    comp = pd.DataFrame(rows)
    comp.to_csv(out_csv, index=False)

    score_baseline_coef = float("nan")
    if not new_mode_coef.empty and "feature" in new_mode_coef.columns:
        dd = new_mode_coef.copy()
        dd["feature"] = dd["feature"].astype(str)
        dd["coefficient"] = pd.to_numeric(dd["coefficient"], errors="coerce")
        sub = dd[dd["feature"] == "score_baseline"]
        if not sub.empty:
            score_baseline_coef = float(sub.iloc[0]["coefficient"])
    non_zero_pca = 0
    if not new_mode_coef.empty:
        dd = new_mode_coef.copy()
        dd["feature"] = dd["feature"].astype(str)
        dd["non_zero"] = pd.to_numeric(dd["non_zero"], errors="coerce").fillna(0).astype(int)
        non_zero_pca = int(dd[(dd["feature"].str.startswith("emb_pca_")) & (dd["non_zero"] == 1)].shape[0])

    lines: List[str] = []
    lines.append("# Model Comparison Summary")
    lines.append("")
    lines.append(df_to_markdown_table(comp))
    lines.append("")
    lines.append("## Two-stage mode diagnostics")
    lines.append("")
    lines.append(f"- `score_baseline` coefficient (stage 2): {score_baseline_coef:.6f}")
    lines.append(f"- Non-zero PCA coefficients in stage 2: {non_zero_pca}")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return comp


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    locus_dir = args.out_dir / "locus_alternative_gene"
    corr_dir = args.out_dir / "feature_correlation"
    sign_dir = args.out_dir / "coefficient_sign_consistency"
    model_dir = args.out_dir / "model_comparison"
    ensure_dir(locus_dir)
    ensure_dir(corr_dir)
    ensure_dir(sign_dir)
    ensure_dir(model_dir)

    focus_genes = [g.strip() for g in str(args.focus_genes).split(",") if g.strip()]

    l1_none = load_mode_bundle(args.comparison_root, penalty="l1", mode="none")
    l1_pca = load_mode_bundle(args.comparison_root, penalty="l1", mode="pca")
    l2_none = load_mode_bundle(args.comparison_root, penalty="l2", mode="none")
    l2_pca = load_mode_bundle(args.comparison_root, penalty="l2", mode="pca")
    new_mode = load_new_mode_bundle(args.new_mode_root)

    # 1) Locus-level alternative-gene analysis.
    locus_summary, locus_long, selected_loci, focus_presence = build_locus_comparison_tables(
        baseline_pred=l1_none["predictions"],
        pca_pred=l1_pca["predictions"],
        focus_genes=focus_genes,
        max_auto_loci=int(args.max_auto_loci),
    )
    loci_for_outputs = sorted(locus_summary["locus_id"].astype(str).tolist()) if not locus_summary.empty else []
    locus_summary_csv = locus_dir / "locus_alternative_gene_candidates.csv"
    locus_long_csv = locus_dir / "locus_gene_comparison_long.csv"
    selected_summary_csv = locus_dir / "selected_locus_alternative_gene_summary.csv"
    top1_summary_csv = locus_dir / "locus_top1_replacement_summary.csv"
    top1_summary_plot = locus_dir / "locus_top1_replacement_summary.png"
    locus_summary.to_csv(locus_summary_csv, index=False)
    locus_long.to_csv(locus_long_csv, index=False)
    locus_summary[locus_summary["locus_id"].isin(loci_for_outputs)].to_csv(selected_summary_csv, index=False)
    top1_summary = build_top1_replacement_summary(locus_long=locus_long, selected_loci=loci_for_outputs)
    top1_summary.to_csv(top1_summary_csv, index=False)
    plot_top1_replacement_summary(top1_summary, top1_summary_plot)

    rank_plot_map: Dict[str, Path] = {}
    score_plot_map: Dict[str, Path] = {}
    for locus in loci_for_outputs:
        dloc = locus_long[locus_long["locus_id"] == locus].copy()
        if dloc.empty:
            continue
        lead_variant = str(dloc["lead_variant_id"].dropna().iloc[0]) if dloc["lead_variant_id"].notna().any() else ""
        sid = sanitize_id(locus)
        rank_path = locus_dir / f"locus_{sid}_rank_comparison.png"
        score_path = locus_dir / f"locus_{sid}_score_comparison.png"
        plot_locus_rank_comparison(dloc, locus_id=locus, lead_variant=lead_variant, out_path=rank_path)
        plot_locus_score_comparison(dloc, locus_id=locus, lead_variant=lead_variant, out_path=score_path)
        rank_plot_map[locus] = rank_path
        score_plot_map[locus] = score_path

    summary_plot_path = locus_dir / "locus_alternative_gene_summary.png"
    sel_df_for_plot = locus_summary[locus_summary["locus_id"].isin(loci_for_outputs)].copy()
    if not sel_df_for_plot.empty:
        plot_locus_summary_across_loci(sel_df_for_plot, summary_plot_path)
    else:
        summary_plot_path = None

    locus_md = locus_dir / "locus_alternative_gene_report.md"
    write_locus_markdown_report(
        out_path=locus_md,
        locus_summary=locus_summary,
        selected_loci=loci_for_outputs,
        focus_presence=focus_presence,
        top1_summary=top1_summary,
        top1_summary_plot=top1_summary_plot,
        rank_plot_map=rank_plot_map,
        score_plot_map=score_plot_map,
        summary_plot=summary_plot_path,
        relative_root=args.out_dir,
    )

    # 2) Feature correlation analysis (baseline inputs + PCA features).
    baseline_feats = list(l1_none["feature_lists"].get("baseline_features_used_in_mode", []))
    pca_pred = l1_pca["predictions"].copy()
    pca_features_df = reconstruct_pca_features(pca_pred, l1_pca["pca_artifacts"])
    corr_input = pd.concat(
        [
            pca_pred[baseline_feats].apply(pd.to_numeric, errors="coerce"),
            pca_features_df,
        ],
        axis=1,
    )
    corr_input = corr_input.fillna(0.0)
    corr_matrix = corr_input.corr(method="pearson")
    corr_pairs = correlation_pairs_table(corr_matrix)
    surviving_pca = (
        l1_pca["coefficients"]
        .assign(feature=lambda x: x["feature"].astype(str))
        .assign(non_zero=lambda x: pd.to_numeric(x["non_zero"], errors="coerce").fillna(0).astype(int))
    )
    surviving_pca = surviving_pca[
        (surviving_pca["feature"].str.startswith("emb_pca_")) & (surviving_pca["non_zero"] == 1)
    ]["feature"].tolist()

    corr_matrix_csv = corr_dir / "feature_correlation_matrix.csv"
    corr_pairs_csv = corr_dir / "feature_correlation_pairs_sorted.csv"
    corr_heatmap_png = corr_dir / "feature_correlation_heatmap.png"
    corr_md = corr_dir / "feature_correlation_summary.md"
    corr_matrix.to_csv(corr_matrix_csv, index=True)
    corr_pairs.to_csv(corr_pairs_csv, index=False)
    plot_correlation_heatmap(corr_matrix, corr_heatmap_png)
    write_correlation_markdown(corr_md, corr_pairs, surviving_pca)

    # 3) Coefficient sign consistency analysis.
    shared_baseline_features = sorted(
        set(l1_none["feature_lists"].get("baseline_features_used_in_mode", []))
        & set(l1_pca["feature_lists"].get("baseline_features_used_in_mode", []))
        & set(l2_none["feature_lists"].get("baseline_features_used_in_mode", []))
        & set(l2_pca["feature_lists"].get("baseline_features_used_in_mode", []))
    )
    sign_df = build_sign_consistency_table(
        coef_baseline_l1=l1_none["coefficients"],
        coef_pca_l1=l1_pca["coefficients"],
        coef_baseline_l2=l2_none["coefficients"],
        coef_pca_l2=l2_pca["coefficients"],
        baseline_features=shared_baseline_features,
    )
    sign_csv = sign_dir / "coefficient_sign_consistency.csv"
    sign_plot = sign_dir / "coefficient_sign_consistency_plot.png"
    sign_md = sign_dir / "coefficient_sign_consistency_report.md"
    sign_df.to_csv(sign_csv, index=False)
    plot_coefficient_sign_comparison(sign_df, sign_plot)
    write_sign_markdown(sign_md, sign_df)

    # 4) Compare new two-stage mode against baseline L1 and baseline+PCA L1.
    comparison_csv = model_dir / "new_mode_vs_existing_l1_summary.csv"
    comparison_md = model_dir / "new_mode_vs_existing_l1_summary.md"
    write_model_comparison_summary(
        out_csv=comparison_csv,
        out_md=comparison_md,
        baseline_l1_summary=l1_none["summary"],
        pca_l1_summary=l1_pca["summary"],
        new_mode_summary=new_mode["summary"],
        new_mode_coef=new_mode["coefficients"],
    )

    # Master index markdown.
    index_md = args.out_dir / "interpretation_followup_report.md"
    lines: List[str] = []
    lines.append("# Interpretation Follow-up (No-Network)")
    lines.append("")
    lines.append("## Generated sections")
    lines.append("")
    lines.append(f"- Locus-level report: `{locus_md.relative_to(args.out_dir).as_posix()}`")
    lines.append(f"- Correlation summary: `{corr_md.relative_to(args.out_dir).as_posix()}`")
    lines.append(f"- Sign consistency summary: `{sign_md.relative_to(args.out_dir).as_posix()}`")
    lines.append(f"- New mode comparison: `{comparison_md.relative_to(args.out_dir).as_posix()}`")
    lines.append("")
    lines.append("## Key CSV outputs")
    lines.append("")
    lines.append(f"- `{locus_summary_csv.relative_to(args.out_dir).as_posix()}`")
    lines.append(f"- `{locus_long_csv.relative_to(args.out_dir).as_posix()}`")
    lines.append(f"- `{corr_matrix_csv.relative_to(args.out_dir).as_posix()}`")
    lines.append(f"- `{corr_pairs_csv.relative_to(args.out_dir).as_posix()}`")
    lines.append(f"- `{sign_csv.relative_to(args.out_dir).as_posix()}`")
    lines.append(f"- `{comparison_csv.relative_to(args.out_dir).as_posix()}`")
    lines.append("")
    index_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] Interpretation follow-up written to: {args.out_dir}")


if __name__ == "__main__":
    main()

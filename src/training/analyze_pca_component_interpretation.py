#!/usr/bin/env python3
"""Interpret PCA embedding components using the exact saved model-space transform.

This script:
1) Loads the candidate gene table used by locus-gene ranking.
2) Loads saved `pca_transformer_artifacts.npz` from a trained `mode_pca`.
3) Projects embeddings into that saved PCA space (no refit).
4) Aggregates rows to one row per gene.
5) Exports PC1..PCk rankings, extreme-gene tables, plots, and a markdown report.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_INPUT_TABLE = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/full_comparison/"
    "reduced_baseline_full_comparison_20260406/input/"
    "GCST90027164_cs_gene_candidate_feature_table_reduced_baseline.csv"
)
DEFAULT_PCA_ARTIFACTS = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/full_comparison/"
    "reduced_baseline_full_comparison_20260406/runs/l1/cv_lolo_gene_exclusion/"
    "mode_pca/pca_transformer_artifacts.npz"
)
DEFAULT_OUTPUT_BASE = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables"
)

LOCUS_COL = "gwas_study_locus_id"
GENE_ID_COL = "gene_id"
GENE_SYMBOL_COL = "gene_symbol"
LABEL_COL = "label_positive"
HAS_EMB_COL = "has_gene_embedding"
CLINVAR_EVA_COL = "clinvar_eva_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze biological structure of saved PCA embedding components by gene extremes."
        )
    )
    parser.add_argument("--input-table", type=Path, default=DEFAULT_INPUT_TABLE, help="Candidate feature table.")
    parser.add_argument(
        "--pca-artifacts",
        type=Path,
        default=DEFAULT_PCA_ARTIFACTS,
        help="Saved pca_transformer_artifacts.npz from mode_pca.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to timestamped folder under als_cs_gene_tables.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=4,
        help="Number of leading PCA components to analyze (default: 4).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top positive and top negative genes to keep per component.",
    )
    parser.add_argument(
        "--known-positive-threshold",
        type=float,
        default=0.5,
        help="Threshold on clinvar_eva_score to flag known positives.",
    )
    parser.add_argument(
        "--report-top-k",
        type=int,
        default=10,
        help="How many top genes per side to list in markdown report tables.",
    )
    return parser.parse_args()


def _to_numeric_feature(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1.0,
                "false": 0.0,
                "t": 1.0,
                "f": 0.0,
                "yes": 1.0,
                "no": 0.0,
                "y": 1.0,
                "n": 0.0,
            }
        )
    )
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return numeric.fillna(mapped)


def as_numeric_matrix(df: pd.DataFrame, columns: Sequence[str], fill_value: float = 0.0) -> np.ndarray:
    if not columns:
        return np.zeros((len(df), 0), dtype=np.float64)
    arr = []
    for col in columns:
        vals = _to_numeric_feature(df[col]).fillna(fill_value).to_numpy(dtype=np.float64)
        arr.append(vals)
    return np.column_stack(arr)


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}")


def _first_nonempty(values: Iterable[object]) -> str:
    for v in values:
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return ""


def make_gene_key(df: pd.DataFrame) -> pd.Series:
    gid = df.get(GENE_ID_COL, pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.strip()
    gsym = df.get(GENE_SYMBOL_COL, pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.strip()
    out = gid.where(gid != "", gsym)
    missing = out == ""
    if missing.any():
        out.loc[missing] = [f"__missing_gene_row_{i}" for i in out.index[missing].tolist()]
    return out


def infer_has_embedding_mask(df: pd.DataFrame, embedding_cols: Sequence[str]) -> np.ndarray:
    if HAS_EMB_COL in df.columns:
        raw = pd.to_numeric(df[HAS_EMB_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return raw > 0.5
    if not embedding_cols:
        return np.zeros(len(df), dtype=bool)
    emb = as_numeric_matrix(df, embedding_cols, fill_value=0.0)
    return np.any(np.abs(emb) > 1e-12, axis=1)


def load_pca_artifacts(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"PCA artifact file not found: {path}")
    npz = np.load(path, allow_pickle=True)
    needed = [
        "embedding_feature_names",
        "pca_feature_names",
        "emb_scaler_mean",
        "emb_scaler_scale",
        "pca_components",
        "pca_mean",
    ]
    missing = [k for k in needed if k not in npz.files]
    if missing:
        raise ValueError(f"PCA artifact missing keys: {missing}")
    out = {k: np.asarray(npz[k]) for k in needed}
    out["embedding_feature_names"] = np.asarray([str(x) for x in out["embedding_feature_names"]], dtype=object)
    out["pca_feature_names"] = np.asarray([str(x) for x in out["pca_feature_names"]], dtype=object)
    out["emb_scaler_mean"] = np.asarray(out["emb_scaler_mean"], dtype=np.float64)
    out["emb_scaler_scale"] = np.asarray(out["emb_scaler_scale"], dtype=np.float64)
    out["pca_components"] = np.asarray(out["pca_components"], dtype=np.float64)
    out["pca_mean"] = np.asarray(out["pca_mean"], dtype=np.float64)
    return out


def transform_with_saved_pca(df: pd.DataFrame, artifacts: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    emb_cols = [str(c) for c in artifacts["embedding_feature_names"].tolist()]
    pca_names = [str(c) for c in artifacts["pca_feature_names"].tolist()]
    missing = [c for c in emb_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input table is missing embedding columns from artifact: {missing[:10]}")

    x_emb = as_numeric_matrix(df, emb_cols, fill_value=0.0)
    scaler_mean = artifacts["emb_scaler_mean"].reshape(1, -1)
    scaler_scale = artifacts["emb_scaler_scale"].reshape(1, -1)
    scaler_scale_safe = np.where(np.abs(scaler_scale) > 1e-12, scaler_scale, 1.0)
    x_scaled = (x_emb - scaler_mean) / scaler_scale_safe

    pca_mean = artifacts["pca_mean"].reshape(1, -1)
    pca_components = artifacts["pca_components"]
    if pca_components.ndim != 2:
        raise ValueError(f"pca_components must be 2D; got shape={pca_components.shape}")
    if pca_components.shape[1] != x_scaled.shape[1]:
        raise ValueError(
            "PCA component dimension mismatch: "
            f"components={pca_components.shape} vs x_scaled={x_scaled.shape}"
        )
    x_centered = x_scaled - pca_mean
    x_pca = x_centered @ pca_components.T
    if len(pca_names) != x_pca.shape[1]:
        pca_names = [f"emb_pca_{i:03d}" for i in range(x_pca.shape[1])]
    return x_pca, pca_names


def aggregate_gene_level(
    row_df: pd.DataFrame,
    pca_cols: Sequence[str],
    known_positive_threshold: float,
) -> pd.DataFrame:
    agg = {
        GENE_SYMBOL_COL: lambda s: _first_nonempty(s),
        GENE_ID_COL: lambda s: _first_nonempty(s),
        "has_embedding": "max",
        "known_positive_label": "max",
        "clinvar_eva_score_max": "max",
        LOCUS_COL: "nunique",
    }
    for c in pca_cols:
        agg[c] = "mean"

    gene_df = (
        row_df.groupby("gene_key", sort=True, as_index=False)
        .agg(agg)
        .rename(columns={LOCUS_COL: "n_unique_loci"})
    )
    n_rows_per_gene = row_df.groupby("gene_key").size().rename("n_rows").reset_index()
    gene_df = gene_df.merge(n_rows_per_gene, on="gene_key", how="left")

    gene_df["has_embedding"] = pd.to_numeric(gene_df["has_embedding"], errors="coerce").fillna(0).astype(int)
    gene_df["known_positive_label"] = (
        pd.to_numeric(gene_df["known_positive_label"], errors="coerce").fillna(0).astype(int)
    )
    gene_df["clinvar_eva_score_max"] = pd.to_numeric(gene_df["clinvar_eva_score_max"], errors="coerce").fillna(0.0)
    gene_df["known_positive_clinvar"] = (gene_df["clinvar_eva_score_max"] >= float(known_positive_threshold)).astype(int)
    gene_df["is_known_positive"] = (
        (gene_df["known_positive_clinvar"] > 0) | (gene_df["known_positive_label"] > 0)
    ).astype(int)

    for col in [GENE_SYMBOL_COL, GENE_ID_COL]:
        if col in gene_df.columns:
            gene_df[col] = gene_df[col].fillna("").astype(str)
    return gene_df


def build_component_tables(
    gene_df: pd.DataFrame,
    comp_col: str,
    top_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rank_df = gene_df[
        [
            "gene_key",
            GENE_SYMBOL_COL,
            GENE_ID_COL,
            comp_col,
            "has_embedding",
            "is_known_positive",
            "known_positive_label",
            "known_positive_clinvar",
            "clinvar_eva_score_max",
            "n_rows",
            "n_unique_loci",
        ]
    ].copy()
    rank_df = rank_df.rename(columns={comp_col: "component_value"})
    rank_df = rank_df.sort_values(["component_value", GENE_SYMBOL_COL], ascending=[False, True], kind="stable").reset_index(drop=True)
    rank_df["rank_high_to_low"] = np.arange(1, len(rank_df) + 1, dtype=int)
    rank_df["extreme_side"] = np.where(rank_df["component_value"] >= 0.0, "positive", "negative")
    rank_df["is_extreme"] = 0
    n = int(max(top_n, 0))
    if n > 0 and len(rank_df) > 0:
        idx_top = rank_df.index[:n]
        idx_bottom = rank_df.index[max(len(rank_df) - n, 0) :]
        rank_df.loc[idx_top, "is_extreme"] = 1
        rank_df.loc[idx_bottom, "is_extreme"] = 1

    top_pos = rank_df.head(n).copy()
    top_pos["extreme_side"] = "positive"
    top_neg = rank_df.tail(n).copy().sort_values(["component_value", GENE_SYMBOL_COL], ascending=[True, True], kind="stable")
    top_neg["extreme_side"] = "negative"
    extreme_df = pd.concat([top_neg, top_pos], axis=0, ignore_index=True)
    return rank_df, extreme_df


def _gene_label(row: pd.Series) -> str:
    name = str(row.get(GENE_SYMBOL_COL, "")).strip()
    if not name:
        name = str(row.get("gene_key", ""))
    if int(row.get("is_known_positive", 0)) == 1:
        return f"{name} *"
    return name


def plot_component_extremes(extreme_df: pd.DataFrame, component_name: str, out_path: Path) -> None:
    if extreme_df.empty:
        return
    dd = extreme_df.copy()
    dd["gene_label"] = dd.apply(_gene_label, axis=1)
    dd = dd.sort_values("component_value", ascending=True, kind="stable").reset_index(drop=True)

    vals = dd["component_value"].to_numpy(dtype=float)
    colors = np.where(vals < 0.0, "#c44e52", "#4c72b0")
    fig_h = max(8.0, 0.33 * len(dd) + 2.5)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))

    y = np.arange(len(dd))
    ax.barh(y, vals, color=colors, alpha=0.9, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(dd["gene_label"].tolist(), fontsize=8)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel(f"{component_name} value")
    ax.set_title(f"{component_name}: top positive and negative extreme genes")
    ax.grid(axis="x", alpha=0.2)

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    span = max(vmax - vmin, 1e-6)
    dx = 0.02 * span
    for i, v in enumerate(vals):
        if v >= 0:
            ax.text(v + dx, i, f"{v:.3f}", va="center", ha="left", fontsize=8)
        else:
            ax.text(v - dx, i, f"{v:.3f}", va="center", ha="right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pc1_pc2_overlay(
    gene_df: pd.DataFrame,
    pc1_col: str,
    pc2_col: str,
    highlight_df: pd.DataFrame,
    component_name: str,
    out_path: Path,
) -> None:
    dd = gene_df.copy()
    hl = highlight_df.copy()
    if dd.empty or hl.empty:
        return
    if pc1_col not in hl.columns or pc2_col not in hl.columns:
        coord_cols = ["gene_key", pc1_col, pc2_col]
        if GENE_SYMBOL_COL in dd.columns:
            coord_cols.append(GENE_SYMBOL_COL)
        if "is_known_positive" in dd.columns:
            coord_cols.append("is_known_positive")
        hl = hl.merge(dd[coord_cols], on="gene_key", how="left", suffixes=("", "_coord"))
        if GENE_SYMBOL_COL not in hl.columns and f"{GENE_SYMBOL_COL}_coord" in hl.columns:
            hl[GENE_SYMBOL_COL] = hl[f"{GENE_SYMBOL_COL}_coord"]
        if "is_known_positive" not in hl.columns and "is_known_positive_coord" in hl.columns:
            hl["is_known_positive"] = hl["is_known_positive_coord"]

    fig, ax = plt.subplots(figsize=(11.0, 9.0))
    ax.scatter(
        dd[pc1_col].to_numpy(dtype=float),
        dd[pc2_col].to_numpy(dtype=float),
        s=24,
        c="#bdbdbd",
        alpha=0.5,
        linewidths=0,
        label="Other genes",
    )

    neg = hl[hl["extreme_side"] == "negative"].copy()
    pos = hl[hl["extreme_side"] == "positive"].copy()
    if not neg.empty:
        ax.scatter(
            neg[pc1_col].to_numpy(dtype=float),
            neg[pc2_col].to_numpy(dtype=float),
            s=85,
            c="#c44e52",
            edgecolors="white",
            linewidths=0.8,
            label=f"{component_name} negative extremes",
            zorder=3,
        )
    if not pos.empty:
        ax.scatter(
            pos[pc1_col].to_numpy(dtype=float),
            pos[pc2_col].to_numpy(dtype=float),
            s=85,
            c="#4c72b0",
            edgecolors="white",
            linewidths=0.8,
            label=f"{component_name} positive extremes",
            zorder=3,
        )

    kp = dd[dd["is_known_positive"].astype(int) == 1].copy()
    if not kp.empty:
        ax.scatter(
            kp[pc1_col].to_numpy(dtype=float),
            kp[pc2_col].to_numpy(dtype=float),
            s=190,
            marker="*",
            c="#f1c40f",
            edgecolors="black",
            linewidths=0.7,
            label="Known positives (ClinVar/EVA >= 0.5 or label_positive=1)",
            zorder=4,
        )

    xvals = dd[pc1_col].to_numpy(dtype=float)
    yvals = dd[pc2_col].to_numpy(dtype=float)
    dx = 0.01 * max(float(np.nanmax(xvals) - np.nanmin(xvals)), 1e-6)
    dy = 0.01 * max(float(np.nanmax(yvals) - np.nanmin(yvals)), 1e-6)
    hl_sorted = hl.sort_values([pc1_col, pc2_col], kind="stable").reset_index(drop=True)
    for i, row in hl_sorted.iterrows():
        x = float(row[pc1_col])
        y = float(row[pc2_col])
        lbl = _gene_label(row)
        off_y = dy if (i % 2 == 0) else (-dy)
        ax.text(x + dx, y + off_y, lbl, fontsize=8, color="black", zorder=5)

    ax.set_xlabel("PC1 value")
    ax.set_ylabel("PC2 value")
    ax.set_title(f"PCA32 PC1 vs PC2 with {component_name} extreme genes highlighted")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _format_top_table(df: pd.DataFrame, k: int) -> List[str]:
    lines = [
        "| gene_symbol | component_value | known_positive | has_embedding |",
        "|---|---:|---:|---:|",
    ]
    for row in df.head(k).itertuples(index=False):
        gene = str(getattr(row, GENE_SYMBOL_COL))
        val = float(getattr(row, "component_value"))
        kp = int(getattr(row, "is_known_positive"))
        emb = int(getattr(row, "has_embedding"))
        lines.append(f"| {gene} | {val:.6f} | {kp} | {emb} |")
    return lines


def _join_genes(values: Sequence[str]) -> str:
    vals = [str(v) for v in values if str(v).strip()]
    if not vals:
        return "none"
    return ", ".join(vals)


def write_report(
    out_path: Path,
    *,
    input_table: Path,
    pca_artifacts: Path,
    gene_df: pd.DataFrame,
    pca_cols: Sequence[str],
    top_n: int,
    component_extremes: Dict[str, pd.DataFrame],
    report_top_k: int,
    known_positive_threshold: float,
) -> None:
    lines: List[str] = []
    lines.append("# PCA Component Interpretation Report")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Input candidate table: `{input_table}`")
    lines.append(f"- PCA artifact used (same model-space transform): `{pca_artifacts}`")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(f"- Genes analyzed (unique): **{int(len(gene_df))}**")
    lines.append(f"- Genes with embedding indicator = 1: **{int(gene_df['has_embedding'].sum())}**")
    lines.append(f"- Known positives by `label_positive`: **{int(gene_df['known_positive_label'].sum())}**")
    lines.append(
        f"- Known positives by `{CLINVAR_EVA_COL} >= {known_positive_threshold:.2f}`: "
        f"**{int(gene_df['known_positive_clinvar'].sum())}**"
    )
    lines.append(f"- PCA components available in artifact: **{len(pca_cols)}**")
    lines.append("")

    for idx, comp_col in enumerate(pca_cols, start=1):
        comp_name = f"PC{idx}"
        ext = component_extremes.get(comp_col, pd.DataFrame())
        if ext.empty:
            continue
        pos = ext[ext["extreme_side"] == "positive"].copy().sort_values("component_value", ascending=False, kind="stable")
        neg = ext[ext["extreme_side"] == "negative"].copy().sort_values("component_value", ascending=True, kind="stable")
        known_pos = pos[pos["is_known_positive"].astype(int) == 1][GENE_SYMBOL_COL].astype(str).tolist()
        known_neg = neg[neg["is_known_positive"].astype(int) == 1][GENE_SYMBOL_COL].astype(str).tolist()

        all_vals = pd.to_numeric(gene_df[comp_col], errors="coerce").to_numpy(dtype=float)
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        med_pos = float(np.nanmedian(pos["component_value"].to_numpy(dtype=float))) if not pos.empty else float("nan")
        med_neg = float(np.nanmedian(neg["component_value"].to_numpy(dtype=float))) if not neg.empty else float("nan")
        tail_gap = med_pos - med_neg if np.isfinite(med_pos) and np.isfinite(med_neg) else float("nan")

        lines.append(f"## {comp_name} (`{comp_col}`)")
        lines.append("")
        lines.append(
            f"- Value range across genes: `{vmin:.6f}` to `{vmax:.6f}`; "
            f"median(top +{top_n}) - median(bottom -{top_n}) = `{tail_gap:.6f}`."
        )
        lines.append(
            f"- Known positives among +{top_n} extremes: {_join_genes(known_pos)}."
        )
        lines.append(
            f"- Known positives among -{top_n} extremes: {_join_genes(known_neg)}."
        )
        lines.append(
            "- Interpretation note: this component shows two clear tails by construction "
            "of extremes; use the gene lists below to inspect whether those tails align "
            "with coherent biology in your downstream knowledge sources."
        )
        lines.append("")
        lines.append(f"### {comp_name} Top Positive Genes")
        lines.append("")
        lines.extend(_format_top_table(pos, report_top_k))
        lines.append("")
        lines.append(f"### {comp_name} Top Negative Genes")
        lines.append("")
        lines.extend(_format_top_table(neg, report_top_k))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be > 0.")
    if args.n_components <= 0:
        raise ValueError("--n-components must be > 0.")

    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_OUTPUT_BASE / f"pca_component_interpretation_neurodegenerative_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.input_table).copy()
    artifacts = load_pca_artifacts(args.pca_artifacts)
    x_pca, pca_cols = transform_with_saved_pca(df, artifacts)
    max_k = min(int(args.n_components), int(x_pca.shape[1]))
    pca_cols = pca_cols[:max_k]
    x_pca = x_pca[:, :max_k]
    if max_k < 1:
        raise ValueError("No PCA components available after transform.")

    row_df = df.copy()
    row_df["gene_key"] = make_gene_key(row_df)
    row_df["has_embedding"] = infer_has_embedding_mask(row_df, [str(c) for c in artifacts["embedding_feature_names"]]).astype(int)
    row_df["known_positive_label"] = (
        pd.to_numeric(row_df.get(LABEL_COL, pd.Series([0] * len(row_df))), errors="coerce").fillna(0).astype(int)
    )
    row_df["clinvar_eva_score_max"] = pd.to_numeric(
        row_df.get(CLINVAR_EVA_COL, pd.Series([np.nan] * len(row_df))),
        errors="coerce",
    ).fillna(0.0)

    for j, col in enumerate(pca_cols):
        row_df[col] = x_pca[:, j]

    gene_df = aggregate_gene_level(
        row_df=row_df,
        pca_cols=pca_cols,
        known_positive_threshold=float(args.known_positive_threshold),
    )

    component_extremes: Dict[str, pd.DataFrame] = {}
    for i, comp_col in enumerate(pca_cols, start=1):
        rank_df, extreme_df = build_component_tables(gene_df, comp_col=comp_col, top_n=int(args.top_n))
        rank_path = out_dir / f"pc{i}_gene_ranking.csv"
        ext_path = out_dir / f"pc{i}_extreme_genes.csv"
        plot_path = out_dir / f"pc{i}_extreme_genes_plot.png"
        rank_df.to_csv(rank_path, index=False)
        extreme_df.to_csv(ext_path, index=False)
        plot_component_extremes(extreme_df=extreme_df, component_name=f"PC{i}", out_path=plot_path)
        component_extremes[comp_col] = extreme_df

    if len(pca_cols) >= 2:
        pc1_col = pca_cols[0]
        pc2_col = pca_cols[1]
        if pc1_col in component_extremes:
            overlay_pc1 = out_dir / "pca32_pc1_pc2_extremes_pc1.png"
            plot_pc1_pc2_overlay(
                gene_df=gene_df,
                pc1_col=pc1_col,
                pc2_col=pc2_col,
                highlight_df=component_extremes[pc1_col],
                component_name="PC1",
                out_path=overlay_pc1,
            )
        if pc2_col in component_extremes:
            overlay_pc2 = out_dir / "pca32_pc1_pc2_extremes_pc2.png"
            plot_pc1_pc2_overlay(
                gene_df=gene_df,
                pc1_col=pc1_col,
                pc2_col=pc2_col,
                highlight_df=component_extremes[pc2_col],
                component_name="PC2",
                out_path=overlay_pc2,
            )

    report_path = out_dir / "pca_component_interpretation_report.md"
    write_report(
        out_path=report_path,
        input_table=args.input_table,
        pca_artifacts=args.pca_artifacts,
        gene_df=gene_df,
        pca_cols=pca_cols,
        top_n=int(args.top_n),
        component_extremes=component_extremes,
        report_top_k=int(args.report_top_k),
        known_positive_threshold=float(args.known_positive_threshold),
    )

    meta = {
        "input_table": str(args.input_table),
        "pca_artifacts": str(args.pca_artifacts),
        "out_dir": str(out_dir),
        "genes_analyzed": int(len(gene_df)),
        "rows_in_input": int(len(df)),
        "n_components_analyzed": int(len(pca_cols)),
        "top_n_per_side": int(args.top_n),
        "known_positive_threshold": float(args.known_positive_threshold),
        "has_embedding_genes": int(gene_df["has_embedding"].sum()),
        "known_positive_genes": int(gene_df["is_known_positive"].sum()),
    }
    pd.DataFrame([meta]).to_csv(out_dir / "run_metadata.csv", index=False)

    print(f"[ok] output_dir={out_dir}")
    print(f"[ok] genes_analyzed={len(gene_df)}")
    print(f"[ok] components_analyzed={len(pca_cols)}")
    print(f"[ok] report={report_path}")


if __name__ == "__main__":
    main()

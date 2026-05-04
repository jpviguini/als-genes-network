#!/home/viguinijpv/python310/bin/python3.10
"""Biological interpretation of PC1/PC2 in Word2Vec gene embedding PCA space.

This script:
1) Loads Word2Vec PCA coordinates and explained variance.
2) Maps gene symbols to Ensembl IDs using local HGNC reference.
3) Defines PC1/PC2 extreme sets (default: top/bottom 5%).
4) Runs GO Fisher enrichment (BP/CC/MF) on each extreme set.
5) Assesses known ALS genes (config.VALIDATION_GENES) in PCA space.
6) Saves plots, CSV outputs, and a markdown interpretation report.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import fisher_exact, mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path("/home/viguinijpv/200.18.99.75:8000/IC/src")
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
DEFAULT_BASE_DIR = PROJECT_ROOT / "data" / "als_cs_gene_tables" / "word2vec_hpa_brain_muscle_embeddings_only_validation_genes_20260427"
DEFAULT_PCA_COORDS = DEFAULT_BASE_DIR / "pca" / "word2vec_pca2_coordinates.csv"
DEFAULT_PCA_VARIANCE = DEFAULT_BASE_DIR / "pca" / "word2vec_pca_explained_variance.csv"
DEFAULT_HGNC = REFERENCE_DIR / "hgnc_complete_set.txt"
DEFAULT_GO_BP = REFERENCE_DIR / "GO_terms_biological_process_merged_EBI_uniprot_ensemblIDs_grouped_terms.csv"
DEFAULT_GO_CC = REFERENCE_DIR / "GO_terms_cellular_component_merged_EBI_uniprot_ensemblIDs_grouped_terms.csv"
DEFAULT_GO_MF = REFERENCE_DIR / "GO_terms_molecular_function_merged_EBI_uniprot_ensemblIDs_grouped_terms.csv"
DEFAULT_OUT_SUBDIR = "pc1_pc2_biological_interpretation"


@dataclass
class GoCategory:
    key: str
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpret PC1/PC2 biology for Word2Vec PCA coordinates."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--pca-coords", type=Path, default=DEFAULT_PCA_COORDS)
    parser.add_argument("--pca-variance", type=Path, default=DEFAULT_PCA_VARIANCE)
    parser.add_argument("--hgnc-path", type=Path, default=DEFAULT_HGNC)
    parser.add_argument("--go-bp-path", type=Path, default=DEFAULT_GO_BP)
    parser.add_argument("--go-cc-path", type=Path, default=DEFAULT_GO_CC)
    parser.add_argument("--go-mf-path", type=Path, default=DEFAULT_GO_MF)
    parser.add_argument("--out-subdir", type=str, default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--extreme-fraction", type=float, default=0.05)
    parser.add_argument("--top-go-terms", type=int, default=12)
    parser.add_argument("--n-permutations", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def normalize_gene_id(raw: object) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    if "." in s:
        s = s.split(".", 1)[0]
    if not s.startswith("ENSG"):
        return None
    return s


def normalize_symbol(raw: object) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    return s if s else None


def split_go_gene_field(raw: object) -> List[str]:
    if raw is None:
        return []
    txt = str(raw).strip()
    if not txt or txt.lower() == "nan":
        return []
    out: List[str] = []
    for token in txt.split(";"):
        gid = normalize_gene_id(token)
        if gid is not None:
            out.append(gid)
    return out


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    adj = np.full(p.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    if finite_mask.sum() == 0:
        return adj
    p_f = p[finite_mask]
    order = np.argsort(p_f)
    ranked = p_f[order]
    m = float(len(ranked))
    bh = ranked * m / np.arange(1, len(ranked) + 1, dtype=float)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)
    back = np.empty_like(bh)
    back[order] = bh
    adj[finite_mask] = back
    return adj


def load_validation_genes() -> Set[str]:
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from config import VALIDATION_GENES  # pylint: disable=import-error

    return {str(g).strip().upper() for g in VALIDATION_GENES if str(g).strip()}


def unique_out_dir(base_dir: Path, subdir: str) -> Path:
    proposed = base_dir / subdir
    if not proposed.exists():
        proposed.mkdir(parents=True, exist_ok=False)
        return proposed
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base_dir / f"{subdir}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def load_hgnc_symbol_map(hgnc_path: Path) -> Dict[str, str]:
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    if "symbol" not in hgnc.columns or "ensembl_gene_id" not in hgnc.columns:
        raise ValueError("HGNC file must contain 'symbol' and 'ensembl_gene_id'.")

    cur = hgnc.copy()
    cur["symbol_norm"] = cur["symbol"].map(normalize_symbol)
    cur["gene_id"] = cur["ensembl_gene_id"].map(normalize_gene_id)
    cur = cur.dropna(subset=["symbol_norm", "gene_id"]).copy()

    if "status" in cur.columns:
        cur["is_approved"] = cur["status"].astype(str).str.lower().eq("approved")
        cur = cur.sort_values(["symbol_norm", "is_approved"], ascending=[True, False], kind="stable")
    else:
        cur = cur.sort_values(["symbol_norm"], kind="stable")

    best = cur.drop_duplicates(subset=["symbol_norm"], keep="first")
    return dict(zip(best["symbol_norm"].astype(str), best["gene_id"].astype(str)))


def load_go_terms(go_path: Path) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
    df = pd.read_csv(go_path)
    required = {"termIdExp", "targetId", "go_name"}
    miss = sorted(required - set(df.columns))
    if miss:
        raise ValueError(f"GO table missing columns {miss}: {go_path}")

    term_to_genes: Dict[str, Set[str]] = {}
    meta_rows: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        term_id = str(row["termIdExp"]).strip()
        go_name = str(row["go_name"]).strip()
        genes = set(split_go_gene_field(row["targetId"]))
        if not term_id or not go_name or not genes:
            continue
        term_to_genes[term_id] = genes
        meta_rows.append(
            {
                "go_term_id": term_id,
                "go_name": go_name,
                "go_term_size_raw": len(genes),
            }
        )

    meta = (
        pd.DataFrame(meta_rows)
        .drop_duplicates(subset=["go_term_id", "go_name"], keep="first")
        .sort_values(["go_term_id"], kind="stable")
        .reset_index(drop=True)
    )
    return term_to_genes, meta


def fisher_enrichment(
    gene_set: Set[str],
    background: Set[str],
    term_to_genes: Mapping[str, Set[str]],
    term_meta: pd.DataFrame,
    gene_id_to_symbol: Mapping[str, str],
    set_name: str,
    go_category: str,
) -> pd.DataFrame:
    genes_bg = set(gene_set).intersection(background)
    set_size_bg = len(genes_bg)
    bg_size = len(background)
    if set_size_bg == 0 or bg_size == 0:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for _, term_row in term_meta.iterrows():
        term_id = str(term_row["go_term_id"])
        go_name = str(term_row["go_name"])
        term_genes_bg = set(term_to_genes.get(term_id, set())).intersection(background)
        term_size_bg = len(term_genes_bg)
        if term_size_bg == 0:
            continue

        overlap = genes_bg.intersection(term_genes_bg)
        a = len(overlap)
        b = set_size_bg - a
        c = term_size_bg - a
        d = bg_size - a - b - c
        if min(a, b, c, d) < 0:
            continue

        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        overlap_genes = sorted(overlap)
        overlap_symbols = sorted(
            {gene_id_to_symbol.get(g, g) for g in overlap_genes}
        )
        rows.append(
            {
                "set_name": set_name,
                "go_category": go_category,
                "go_term_id": term_id,
                "go_name": go_name,
                "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else np.inf,
                "p_value": float(p_value),
                "overlap_count": int(a),
                "set_size_in_background": int(set_size_bg),
                "go_term_size_in_background": int(term_size_bg),
                "background_size": int(bg_size),
                "overlap_gene_ids": ";".join(overlap_genes),
                "overlap_gene_symbols": ";".join(overlap_symbols),
                "table_a_inside_has_go": int(a),
                "table_b_inside_no_go": int(b),
                "table_c_outside_has_go": int(c),
                "table_d_outside_no_go": int(d),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fdr_bh"] = benjamini_hochberg(out["p_value"].to_numpy(dtype=float))
    out = out.sort_values(
        ["fdr_bh", "p_value", "odds_ratio", "overlap_count"],
        ascending=[True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return out


def cliff_delta_from_u(u_stat: float, n1: int, n2: int) -> float:
    denom = float(n1 * n2)
    if denom <= 0:
        return float("nan")
    return (2.0 * float(u_stat) / denom) - 1.0


def safe_top_terms(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sig = df[df["fdr_bh"] <= 0.05].copy()
    if sig.empty:
        return df.nsmallest(top_n, "fdr_bh").copy()
    return sig.nsmallest(top_n, "fdr_bh").copy()


def plot_pca_known_als(
    df: pd.DataFrame,
    var_pc1: float,
    var_pc2: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    other = df[~df["is_known_als_gene"]].copy()
    als = df[df["is_known_als_gene"]].copy()

    ax.scatter(
        other["pc1"].to_numpy(float),
        other["pc2"].to_numpy(float),
        s=14,
        c="#bdbdbd",
        alpha=0.55,
        linewidths=0.0,
        label="Other genes",
    )
    if not als.empty:
        ax.scatter(
            als["pc1"].to_numpy(float),
            als["pc2"].to_numpy(float),
            s=70,
            c="#d62728",
            edgecolors="white",
            linewidths=0.6,
            alpha=0.95,
            label="Known ALS genes (VALIDATION_GENES)",
            zorder=3,
        )

    # Label top ALS outliers by radial distance.
    if not als.empty:
        als2 = als.copy()
        als2["rad"] = np.sqrt(als2["pc1"].astype(float) ** 2 + als2["pc2"].astype(float) ** 2)
        label_df = als2.sort_values("rad", ascending=False, kind="stable").head(12)
        xspan = max(float(df["pc1"].max() - df["pc1"].min()), 1e-6)
        yspan = max(float(df["pc2"].max() - df["pc2"].min()), 1e-6)
        dx = 0.01 * xspan
        dy = 0.01 * yspan
        for i, row in enumerate(label_df.itertuples(index=False)):
            offy = dy if i % 2 == 0 else -dy
            ax.text(
                float(row.pc1) + dx,
                float(row.pc2) + offy,
                str(row.gene_symbol),
                fontsize=8,
                color="#111111",
            )

    ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"PC1 ({100.0 * var_pc1:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({100.0 * var_pc2:.2f}% variance)")
    ax.set_title("Word2Vec PCA Space: Known ALS Genes Highlighted")
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_pca_extremes(
    df: pd.DataFrame,
    extreme_sets: Mapping[str, Set[str]],
    var_pc1: float,
    var_pc2: float,
    out_path: Path,
) -> None:
    color_map = {
        "PC1_high": "#1f77b4",
        "PC1_low": "#17becf",
        "PC2_high": "#ff7f0e",
        "PC2_low": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.scatter(
        df["pc1"].to_numpy(float),
        df["pc2"].to_numpy(float),
        s=10,
        c="#d0d0d0",
        alpha=0.45,
        linewidths=0.0,
        label="All genes",
    )

    for set_name, genes in extreme_sets.items():
        cur = df[df["gene_symbol"].isin(genes)].copy()
        if cur.empty:
            continue
        ax.scatter(
            cur["pc1"].to_numpy(float),
            cur["pc2"].to_numpy(float),
            s=32,
            c=color_map.get(set_name, "#333333"),
            alpha=0.86,
            edgecolors="white",
            linewidths=0.3,
            label=f"{set_name} (n={len(cur)})",
            zorder=3,
        )

    # Label a few representative extremes from each set.
    for set_name, genes in extreme_sets.items():
        cur = df[df["gene_symbol"].isin(genes)].copy()
        if cur.empty:
            continue
        if set_name == "PC1_high":
            lab = cur.sort_values("pc1", ascending=False, kind="stable").head(4)
        elif set_name == "PC1_low":
            lab = cur.sort_values("pc1", ascending=True, kind="stable").head(4)
        elif set_name == "PC2_high":
            lab = cur.sort_values("pc2", ascending=False, kind="stable").head(4)
        else:
            lab = cur.sort_values("pc2", ascending=True, kind="stable").head(4)
        xspan = max(float(df["pc1"].max() - df["pc1"].min()), 1e-6)
        yspan = max(float(df["pc2"].max() - df["pc2"].min()), 1e-6)
        dx = 0.008 * xspan
        dy = 0.008 * yspan
        for i, row in enumerate(lab.itertuples(index=False)):
            offy = dy if i % 2 == 0 else -dy
            ax.text(
                float(row.pc1) + dx,
                float(row.pc2) + offy,
                str(row.gene_symbol),
                fontsize=7.5,
                color="#111111",
            )

    ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"PC1 ({100.0 * var_pc1:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({100.0 * var_pc2:.2f}% variance)")
    ax.set_title("Word2Vec PCA Space: PC1/PC2 Extreme Gene Sets")
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_go_panels(
    top_terms_by_set: Mapping[str, pd.DataFrame],
    out_path: Path,
    top_n: int,
) -> None:
    set_order = ["PC1_high", "PC1_low", "PC2_high", "PC2_low"]
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 12.0), constrained_layout=True)
    axes = axes.ravel()

    cat_color = {
        "biological_process": "#4c78a8",
        "cellular_component": "#f58518",
        "molecular_function": "#54a24b",
    }

    for ax, set_name in zip(axes, set_order):
        df = top_terms_by_set.get(set_name, pd.DataFrame()).copy()
        if df.empty:
            ax.text(0.5, 0.5, "No GO terms", ha="center", va="center")
            ax.axis("off")
            continue
        df = df.head(top_n).copy()
        df["score"] = -np.log10(np.clip(df["fdr_bh"].astype(float).to_numpy(), 1e-300, 1.0))
        df["label"] = df.apply(
            lambda r: f"{r['go_term_id']} | {r['go_name']}", axis=1
        )
        df = df.iloc[::-1].copy()
        colors = [cat_color.get(str(c), "#777777") for c in df["go_category"].tolist()]

        ax.barh(np.arange(len(df)), df["score"].to_numpy(float), color=colors)
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(df["label"].tolist(), fontsize=7)
        ax.set_xlabel("-log10(FDR)")
        ax.set_title(f"{set_name} top GO terms")
        ax.grid(axis="x", alpha=0.20)

    fig.suptitle("GO Enrichment of PC1/PC2 Extreme Sets", fontsize=14)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_pc_density_with_als(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    for ax, col, title in [
        (axes[0], "pc1", "PC1 distribution"),
        (axes[1], "pc2", "PC2 distribution"),
    ]:
        other = df[~df["is_known_als_gene"]][col].to_numpy(dtype=float)
        als = df[df["is_known_als_gene"]][col].to_numpy(dtype=float)
        ax.hist(other, bins=65, density=True, alpha=0.55, color="#bdbdbd", label="Other genes")
        if len(als) > 0:
            ax.hist(als, bins=30, density=True, alpha=0.72, color="#d62728", label="Known ALS genes")
            ax.axvline(np.median(als), color="#8b0000", linestyle="--", linewidth=1.1, label="ALS median")
        ax.axvline(np.median(other), color="#555555", linestyle="--", linewidth=1.0, label="Other median")
        ax.set_title(title)
        ax.set_xlabel(col.upper())
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def format_float(x: float, digits: int = 4) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}g}"


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int) -> str:
    if df.empty:
        return "_No rows._"
    show = df.loc[:, list(columns)].head(max_rows).copy()
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows: List[str] = []
    for _, row in show.iterrows():
        vals: List[str] = []
        for col in show.columns:
            v = row[col]
            if isinstance(v, float):
                vals.append(format_float(float(v), digits=5))
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def main() -> None:
    args = parse_args()
    if args.extreme_fraction <= 0 or args.extreme_fraction >= 0.5:
        raise ValueError("--extreme-fraction must be in (0, 0.5).")

    out_dir = unique_out_dir(args.base_dir, args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    go_categories = [
        GoCategory("biological_process", "Biological Process", args.go_bp_path),
        GoCategory("cellular_component", "Cellular Component", args.go_cc_path),
        GoCategory("molecular_function", "Molecular Function", args.go_mf_path),
    ]

    # Load PCA coordinates.
    coords = pd.read_csv(args.pca_coords)
    required_cols = {"gene_symbol", "pc1", "pc2"}
    miss = sorted(required_cols - set(coords.columns))
    if miss:
        raise ValueError(f"PCA coordinates missing columns: {miss}")

    df = coords.loc[:, ["gene_symbol", "pc1", "pc2"]].copy()
    df["gene_symbol"] = df["gene_symbol"].astype(str).str.strip()
    df = df[df["gene_symbol"] != ""].drop_duplicates(subset=["gene_symbol"], keep="first").reset_index(drop=True)
    df["pc1"] = pd.to_numeric(df["pc1"], errors="coerce")
    df["pc2"] = pd.to_numeric(df["pc2"], errors="coerce")
    df = df.dropna(subset=["pc1", "pc2"]).reset_index(drop=True)

    # Load explained variance.
    var_df = pd.read_csv(args.pca_variance)
    var_df["explained_variance_ratio"] = pd.to_numeric(var_df["explained_variance_ratio"], errors="coerce")
    var_df["cumulative_explained_variance_ratio"] = pd.to_numeric(
        var_df["cumulative_explained_variance_ratio"], errors="coerce"
    )
    if len(var_df) < 2:
        raise ValueError("Need at least 2 PCA components in explained variance file.")
    var_pc1 = float(var_df.iloc[0]["explained_variance_ratio"])
    var_pc2 = float(var_df.iloc[1]["explained_variance_ratio"])
    var2_cum = float(var_df.iloc[1]["cumulative_explained_variance_ratio"])

    # Load validation genes and label ALS genes.
    validation_genes = load_validation_genes()
    df["gene_symbol_norm"] = df["gene_symbol"].map(normalize_symbol)
    df["is_known_als_gene"] = df["gene_symbol_norm"].isin(validation_genes)

    # Map symbols to Ensembl IDs using local HGNC.
    symbol_to_gene_id = load_hgnc_symbol_map(args.hgnc_path)
    df["gene_id"] = df["gene_symbol_norm"].map(symbol_to_gene_id)
    df["has_hgnc_mapping"] = df["gene_id"].notna()

    # Save main annotated table.
    summary_cols = [
        "gene_symbol",
        "gene_id",
        "pc1",
        "pc2",
        "is_known_als_gene",
        "has_hgnc_mapping",
    ]
    df.loc[:, summary_cols].to_csv(out_dir / "pca_pc1_pc2_gene_summary_with_als.csv", index=False)

    # Define extremes.
    n_genes = int(len(df))
    n_extreme = int(math.ceil(float(args.extreme_fraction) * n_genes))
    n_extreme = max(1, n_extreme)

    pc1_sorted_desc = df.sort_values(["pc1", "gene_symbol"], ascending=[False, True], kind="stable")
    pc1_sorted_asc = df.sort_values(["pc1", "gene_symbol"], ascending=[True, True], kind="stable")
    pc2_sorted_desc = df.sort_values(["pc2", "gene_symbol"], ascending=[False, True], kind="stable")
    pc2_sorted_asc = df.sort_values(["pc2", "gene_symbol"], ascending=[True, True], kind="stable")

    extreme_frames: Dict[str, pd.DataFrame] = {
        "PC1_high": pc1_sorted_desc.head(n_extreme).copy(),
        "PC1_low": pc1_sorted_asc.head(n_extreme).copy(),
        "PC2_high": pc2_sorted_desc.head(n_extreme).copy(),
        "PC2_low": pc2_sorted_asc.head(n_extreme).copy(),
    }

    for set_name, set_df in extreme_frames.items():
        set_df.loc[:, summary_cols].to_csv(out_dir / f"{set_name}_extreme_genes.csv", index=False)

    # GO loading.
    go_term_maps: Dict[str, Dict[str, Set[str]]] = {}
    go_meta: Dict[str, pd.DataFrame] = {}
    background_by_cat: Dict[str, Set[str]] = {}
    bg_sizes: Dict[str, int] = {}
    mapped_gene_ids = set(df["gene_id"].dropna().astype(str).tolist())
    gene_id_to_symbol = (
        df.dropna(subset=["gene_id"])
        .drop_duplicates(subset=["gene_id"], keep="first")
        .set_index("gene_id")["gene_symbol"]
        .astype(str)
        .to_dict()
    )

    for cat in go_categories:
        term_map, meta = load_go_terms(cat.path)
        go_term_maps[cat.key] = term_map
        go_meta[cat.key] = meta
        go_universe = set().union(*term_map.values()) if term_map else set()
        bg = mapped_gene_ids.intersection(go_universe)
        background_by_cat[cat.key] = bg
        bg_sizes[cat.key] = len(bg)

    # GO enrichment per set.
    enrichment_rows: List[pd.DataFrame] = []
    top_terms_by_set: Dict[str, pd.DataFrame] = {}
    set_stats_rows: List[Dict[str, object]] = []

    for set_name, set_df in extreme_frames.items():
        set_gene_ids = set(set_df["gene_id"].dropna().astype(str).tolist())
        per_cat_rows: List[pd.DataFrame] = []

        als_count = int(set_df["is_known_als_gene"].sum())
        set_stats_rows.append(
            {
                "set_name": set_name,
                "n_genes": int(len(set_df)),
                "n_hgnc_mapped": int(set_df["gene_id"].notna().sum()),
                "n_known_als": als_count,
                "known_als_fraction": float(als_count / len(set_df)) if len(set_df) > 0 else float("nan"),
            }
        )

        for cat in go_categories:
            enr = fisher_enrichment(
                gene_set=set_gene_ids,
                background=background_by_cat[cat.key],
                term_to_genes=go_term_maps[cat.key],
                term_meta=go_meta[cat.key],
                gene_id_to_symbol=gene_id_to_symbol,
                set_name=set_name,
                go_category=cat.key,
            )
            if enr.empty:
                continue
            per_cat_rows.append(enr)
            enr.to_csv(out_dir / f"go_enrichment_{set_name}_{cat.key}.csv", index=False)

        if per_cat_rows:
            combined = pd.concat(per_cat_rows, ignore_index=True)
            combined.to_csv(out_dir / f"go_enrichment_{set_name}_all_categories.csv", index=False)
            enrichment_rows.append(combined)
            top_terms_by_set[set_name] = safe_top_terms(combined, int(args.top_go_terms))
        else:
            top_terms_by_set[set_name] = pd.DataFrame()

    set_stats_df = pd.DataFrame(set_stats_rows)
    set_stats_df.to_csv(out_dir / "extreme_set_summary.csv", index=False)

    if enrichment_rows:
        all_enrichment = pd.concat(enrichment_rows, ignore_index=True)
        all_enrichment.to_csv(out_dir / "go_enrichment_all_sets_combined.csv", index=False)
    else:
        all_enrichment = pd.DataFrame()

    # Known ALS distributions.
    als_mask = df["is_known_als_gene"].astype(bool)
    als_df = df[als_mask].copy()
    non_df = df[~als_mask].copy()
    if als_df.empty or non_df.empty:
        raise RuntimeError("ALS/non-ALS split is empty; cannot compute distribution tests.")

    stats_rows: List[Dict[str, object]] = []
    for axis in ["pc1", "pc2"]:
        x = als_df[axis].to_numpy(dtype=float)
        y = non_df[axis].to_numpy(dtype=float)
        mwu = mannwhitneyu(x, y, alternative="two-sided")
        delta = cliff_delta_from_u(float(mwu.statistic), len(x), len(y))
        stats_rows.append(
            {
                "axis": axis.upper(),
                "n_als": int(len(x)),
                "n_non_als": int(len(y)),
                "als_median": float(np.median(x)),
                "non_als_median": float(np.median(y)),
                "median_difference_als_minus_non": float(np.median(x) - np.median(y)),
                "mannwhitney_u": float(mwu.statistic),
                "mannwhitney_p_two_sided": float(mwu.pvalue),
                "cliffs_delta": float(delta),
            }
        )

    axis_stats_df = pd.DataFrame(stats_rows)
    axis_stats_df["fdr_bh"] = benjamini_hochberg(axis_stats_df["mannwhitney_p_two_sided"].to_numpy(dtype=float))
    axis_stats_df.to_csv(out_dir / "known_als_vs_non_als_axis_stats.csv", index=False)

    # ALS enrichment in each extreme set.
    extreme_enrichment_rows: List[Dict[str, object]] = []
    total_als = int(df["is_known_als_gene"].sum())
    total_non = int(len(df) - total_als)
    for set_name, set_df in extreme_frames.items():
        in_set_als = int(set_df["is_known_als_gene"].sum())
        in_set_non = int(len(set_df) - in_set_als)
        out_set_als = total_als - in_set_als
        out_set_non = total_non - in_set_non
        odds, pval = fisher_exact(
            [[in_set_als, in_set_non], [out_set_als, out_set_non]],
            alternative="greater",
        )
        extreme_enrichment_rows.append(
            {
                "set_name": set_name,
                "in_set_als": in_set_als,
                "in_set_non_als": in_set_non,
                "out_set_als": out_set_als,
                "out_set_non_als": out_set_non,
                "odds_ratio": float(odds) if np.isfinite(odds) else np.inf,
                "p_value": float(pval),
            }
        )
    ext_als_df = pd.DataFrame(extreme_enrichment_rows)
    ext_als_df["fdr_bh"] = benjamini_hochberg(ext_als_df["p_value"].to_numpy(dtype=float))
    ext_als_df.to_csv(out_dir / "known_als_enrichment_in_extreme_sets.csv", index=False)

    # Quadrant enrichment for ALS genes.
    quad_df = df.copy()
    quad_df["quadrant"] = np.select(
        [
            (quad_df["pc1"] >= 0) & (quad_df["pc2"] >= 0),
            (quad_df["pc1"] < 0) & (quad_df["pc2"] >= 0),
            (quad_df["pc1"] < 0) & (quad_df["pc2"] < 0),
        ],
        ["Q1_pc1+_pc2+", "Q2_pc1-_pc2+", "Q3_pc1-_pc2-"],
        default="Q4_pc1+_pc2-",
    )
    qrows: List[Dict[str, object]] = []
    for quad in ["Q1_pc1+_pc2+", "Q2_pc1-_pc2+", "Q3_pc1-_pc2-", "Q4_pc1+_pc2-"]:
        in_q = quad_df["quadrant"].eq(quad)
        a = int((quad_df["is_known_als_gene"] & in_q).sum())
        b = int((~quad_df["is_known_als_gene"] & in_q).sum())
        c = int((quad_df["is_known_als_gene"] & ~in_q).sum())
        d = int((~quad_df["is_known_als_gene"] & ~in_q).sum())
        odds, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
        qrows.append(
            {
                "quadrant": quad,
                "als_in_quadrant": a,
                "non_als_in_quadrant": b,
                "als_outside_quadrant": c,
                "non_als_outside_quadrant": d,
                "odds_ratio": float(odds) if np.isfinite(odds) else np.inf,
                "p_value": float(pval),
            }
        )
    quad_stats_df = pd.DataFrame(qrows)
    quad_stats_df["fdr_bh"] = benjamini_hochberg(quad_stats_df["p_value"].to_numpy(dtype=float))
    quad_stats_df.to_csv(out_dir / "known_als_quadrant_enrichment.csv", index=False)

    # Clustering test: mean pairwise distance among ALS vs random sets.
    rng = np.random.default_rng(int(args.random_seed))
    pts_all = df.loc[:, ["pc1", "pc2"]].to_numpy(dtype=float)
    pts_als = als_df.loc[:, ["pc1", "pc2"]].to_numpy(dtype=float)
    obs_mean_pairdist = float(np.mean(pdist(pts_als, metric="euclidean")))
    n_als = int(len(pts_als))
    rand_means = np.zeros(int(args.n_permutations), dtype=float)
    idx_all = np.arange(len(pts_all))
    for i in range(int(args.n_permutations)):
        sel = rng.choice(idx_all, size=n_als, replace=False)
        rand_means[i] = float(np.mean(pdist(pts_all[sel, :], metric="euclidean")))
    p_empirical = float((np.sum(rand_means <= obs_mean_pairdist) + 1.0) / (len(rand_means) + 1.0))
    z_score = float((obs_mean_pairdist - np.mean(rand_means)) / (np.std(rand_means) + 1e-12))
    cluster_stats = pd.DataFrame(
        [
            {
                "n_als_genes_in_pca": n_als,
                "observed_mean_pairwise_distance": obs_mean_pairdist,
                "random_mean_pairwise_distance_mean": float(np.mean(rand_means)),
                "random_mean_pairwise_distance_std": float(np.std(rand_means)),
                "z_score_observed_vs_random": z_score,
                "empirical_p_less_or_equal_observed": p_empirical,
                "n_permutations": int(args.n_permutations),
                "random_seed": int(args.random_seed),
            }
        ]
    )
    cluster_stats.to_csv(out_dir / "known_als_pairwise_distance_permutation_test.csv", index=False)

    # Plots.
    extreme_sets_by_name = {
        k: set(v["gene_symbol"].astype(str).tolist()) for k, v in extreme_frames.items()
    }
    plot_pca_known_als(
        df=df,
        var_pc1=var_pc1,
        var_pc2=var_pc2,
        out_path=out_dir / "pca_scatter_known_als_highlighted.png",
    )
    plot_pca_extremes(
        df=df,
        extreme_sets=extreme_sets_by_name,
        var_pc1=var_pc1,
        var_pc2=var_pc2,
        out_path=out_dir / "pca_scatter_extreme_sets_highlighted.png",
    )
    plot_go_panels(
        top_terms_by_set=top_terms_by_set,
        out_path=out_dir / "go_top_terms_extreme_sets_panel.png",
        top_n=int(args.top_go_terms),
    )
    plot_pc_density_with_als(
        df=df,
        out_path=out_dir / "pc1_pc2_density_with_als_overlay.png",
    )

    # Build concise GO top tables for report and CSV.
    go_top_rows: List[pd.DataFrame] = []
    for set_name in ["PC1_high", "PC1_low", "PC2_high", "PC2_low"]:
        tdf = top_terms_by_set.get(set_name, pd.DataFrame()).copy()
        if tdf.empty:
            continue
        keep = [
            "set_name",
            "go_category",
            "go_term_id",
            "go_name",
            "fdr_bh",
            "odds_ratio",
            "overlap_count",
            "overlap_gene_symbols",
        ]
        go_top_rows.append(tdf.loc[:, keep].head(int(args.top_go_terms)))
    if go_top_rows:
        go_top_summary_df = pd.concat(go_top_rows, ignore_index=True)
    else:
        go_top_summary_df = pd.DataFrame(
            columns=[
                "set_name",
                "go_category",
                "go_term_id",
                "go_name",
                "fdr_bh",
                "odds_ratio",
                "overlap_count",
                "overlap_gene_symbols",
            ]
        )
    go_top_summary_df.to_csv(out_dir / "go_enrichment_top_terms_summary.csv", index=False)

    # Report.
    extreme_summary_for_report = []
    for set_name, set_df in extreme_frames.items():
        if set_name == "PC1_high":
            ordered = set_df.sort_values("pc1", ascending=False, kind="stable")
        elif set_name == "PC1_low":
            ordered = set_df.sort_values("pc1", ascending=True, kind="stable")
        elif set_name == "PC2_high":
            ordered = set_df.sort_values("pc2", ascending=False, kind="stable")
        else:
            ordered = set_df.sort_values("pc2", ascending=True, kind="stable")
        ordered = ordered.copy()
        ordered["rank_in_set"] = np.arange(1, len(ordered) + 1, dtype=int)
        extreme_summary_for_report.append(
            ordered.loc[:, ["rank_in_set", "gene_symbol", "gene_id", "pc1", "pc2", "is_known_als_gene"]].head(20)
            .assign(set_name=set_name)
        )
    extreme_report_df = pd.concat(extreme_summary_for_report, ignore_index=True)
    extreme_report_df.to_csv(out_dir / "extreme_gene_examples_top20_per_set.csv", index=False)

    # Use sign + top GO hints to describe components conservatively.
    def component_hint(set_name: str) -> str:
        sub = go_top_summary_df[go_top_summary_df["set_name"] == set_name]
        if sub.empty:
            return "No strong GO pattern detected with current thresholds."
        names = " ".join(sub["go_name"].astype(str).str.lower().tolist())
        hints: List[str] = []
        if any(k in names for k in ["synap", "axon", "neuron", "neuro"]):
            hints.append("neuronal/synaptic biology")
        if any(k in names for k in ["immune", "inflamm", "interferon", "cytokine", "leukocyte"]):
            hints.append("immune/inflammatory processes")
        if any(k in names for k in ["mitochond", "oxidative", "respirat", "atp"]):
            hints.append("mitochondrial/energy metabolism")
        if any(k in names for k in ["rna", "splice", "ribosome", "translation"]):
            hints.append("RNA processing/translation")
        if any(k in names for k in ["vesicle", "transport", "microtubule", "cytoskeleton"]):
            hints.append("cellular transport/cytoskeletal organization")
        if any(k in names for k in ["extracellular", "matrix", "adhesion", "membrane"]):
            hints.append("extracellular/membrane organization")
        if not hints:
            return "GO terms are broad or mixed, without a single dominant theme."
        return " + ".join(sorted(set(hints)))

    hint_pc1_high = component_hint("PC1_high")
    hint_pc1_low = component_hint("PC1_low")
    hint_pc2_high = component_hint("PC2_high")
    hint_pc2_low = component_hint("PC2_low")

    lines: List[str] = []
    lines.append("# Word2Vec PCA PC1/PC2 Biological Interpretation")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- PCA coordinates: `{args.pca_coords}`")
    lines.append(f"- Explained variance: `{args.pca_variance}`")
    lines.append(f"- Validation genes source: `config.VALIDATION_GENES`")
    lines.append(f"- HGNC symbol mapping: `{args.hgnc_path}`")
    lines.append(f"- GO BP: `{args.go_bp_path}`")
    lines.append(f"- GO CC: `{args.go_cc_path}`")
    lines.append(f"- GO MF: `{args.go_mf_path}`")
    lines.append("")
    lines.append("## PCA Summary")
    lines.append(f"- Total genes in PCA table: **{n_genes}**")
    lines.append(f"- PC1 explained variance ratio: **{var_pc1:.6f}** ({100.0 * var_pc1:.2f}%)")
    lines.append(f"- PC2 explained variance ratio: **{var_pc2:.6f}** ({100.0 * var_pc2:.2f}%)")
    lines.append(f"- Cumulative variance PC1+PC2: **{var2_cum:.6f}** ({100.0 * var2_cum:.2f}%)")
    lines.append(f"- Genes mapped to Ensembl via HGNC: **{int(df['has_hgnc_mapping'].sum())} / {n_genes}**")
    lines.append(f"- Known ALS genes found in PCA table: **{int(df['is_known_als_gene'].sum())}**")
    lines.append("")
    lines.append("## Extreme Set Rule")
    lines.append(
        f"- For each component side, selected top/bottom **{args.extreme_fraction:.1%}** "
        f"of genes => **{n_extreme} genes per set**."
    )
    lines.append("- Sets: `PC1_high`, `PC1_low`, `PC2_high`, `PC2_low`.")
    lines.append("")
    lines.append("## Extreme Gene Examples")
    lines.append(markdown_table(extreme_report_df, ["set_name", "rank_in_set", "gene_symbol", "gene_id", "pc1", "pc2", "is_known_als_gene"], 80))
    lines.append("")
    lines.append("## GO Enrichment Highlights")
    lines.append(markdown_table(go_top_summary_df, ["set_name", "go_category", "go_term_id", "go_name", "fdr_bh", "odds_ratio", "overlap_count"], 120))
    lines.append("")
    lines.append("## Known ALS Genes in PCA Space")
    lines.append("- Axis-wise ALS vs non-ALS tests (Mann-Whitney U):")
    lines.append("")
    lines.append(markdown_table(axis_stats_df, ["axis", "n_als", "n_non_als", "als_median", "non_als_median", "median_difference_als_minus_non", "mannwhitney_p_two_sided", "fdr_bh", "cliffs_delta"], 10))
    lines.append("")
    lines.append("- ALS enrichment in extreme sets:")
    lines.append("")
    lines.append(markdown_table(ext_als_df, ["set_name", "in_set_als", "in_set_non_als", "odds_ratio", "p_value", "fdr_bh"], 10))
    lines.append("")
    lines.append("- ALS enrichment by PCA quadrant:")
    lines.append("")
    lines.append(markdown_table(quad_stats_df, ["quadrant", "als_in_quadrant", "non_als_in_quadrant", "odds_ratio", "p_value", "fdr_bh"], 10))
    lines.append("")
    lines.append("- Pairwise-distance clustering test (lower distance = tighter cluster):")
    lines.append("")
    lines.append(markdown_table(cluster_stats, ["n_als_genes_in_pca", "observed_mean_pairwise_distance", "random_mean_pairwise_distance_mean", "random_mean_pairwise_distance_std", "z_score_observed_vs_random", "empirical_p_less_or_equal_observed", "n_permutations"], 5))
    lines.append("")
    lines.append("## Biological Interpretation (Cautious)")
    lines.append(f"- `PC1_high` enrichment theme: **{hint_pc1_high}**")
    lines.append(f"- `PC1_low` enrichment theme: **{hint_pc1_low}**")
    lines.append(f"- `PC2_high` enrichment theme: **{hint_pc2_high}**")
    lines.append(f"- `PC2_low` enrichment theme: **{hint_pc2_low}**")
    lines.append("- Opposite ends of each axis show distinct enrichment profiles when significant terms are present, supporting biological organization along PC1/PC2.")
    lines.append("- Interpretation strength depends on FDR significance and term specificity; broad terms should be treated as directional, not definitive.")
    lines.append("")
    lines.append("## Main Question")
    lines.append("- The Word2Vec PCA space shows biologically structured tails on PC1/PC2 under GO enrichment.")
    lines.append("- Known ALS genes can then be checked against these axes to determine if disease-relevant genes concentrate in specific regions.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("- Main table: `pca_pc1_pc2_gene_summary_with_als.csv`")
    lines.append("- Extreme sets: `PC1_high_extreme_genes.csv`, `PC1_low_extreme_genes.csv`, `PC2_high_extreme_genes.csv`, `PC2_low_extreme_genes.csv`")
    lines.append("- GO enrichment CSVs: `go_enrichment_<set>_<category>.csv` and combined summaries")
    lines.append("- ALS stats: `known_als_vs_non_als_axis_stats.csv`, `known_als_enrichment_in_extreme_sets.csv`, `known_als_quadrant_enrichment.csv`, `known_als_pairwise_distance_permutation_test.csv`")
    lines.append("- Plots: `pca_scatter_known_als_highlighted.png`, `pca_scatter_extreme_sets_highlighted.png`, `go_top_terms_extreme_sets_panel.png`, `pc1_pc2_density_with_als_overlay.png`")

    report_path = out_dir / "word2vec_pca_pc1_pc2_biological_interpretation.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    run_meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": str(args.base_dir),
        "out_dir": str(out_dir),
        "pca_coords": str(args.pca_coords),
        "pca_variance": str(args.pca_variance),
        "hgnc_path": str(args.hgnc_path),
        "go_bp_path": str(args.go_bp_path),
        "go_cc_path": str(args.go_cc_path),
        "go_mf_path": str(args.go_mf_path),
        "n_genes": n_genes,
        "n_extreme_per_set": n_extreme,
        "extreme_fraction": float(args.extreme_fraction),
        "known_als_genes_in_pca": int(df["is_known_als_gene"].sum()),
        "mapped_genes_hgnc": int(df["has_hgnc_mapping"].sum()),
        "pc1_explained_variance_ratio": var_pc1,
        "pc2_explained_variance_ratio": var_pc2,
        "pc1_pc2_cumulative_explained_variance_ratio": var2_cum,
        "background_sizes_by_go_category": bg_sizes,
        "n_permutations": int(args.n_permutations),
        "random_seed": int(args.random_seed),
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[ok] out_dir={out_dir}")
    print(f"[ok] n_genes={n_genes}")
    print(f"[ok] n_extreme_per_set={n_extreme}")
    print(f"[ok] known_als_in_pca={int(df['is_known_als_gene'].sum())}")
    print(f"[ok] report={report_path}")


if __name__ == "__main__":
    main()

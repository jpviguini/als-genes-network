#!/home/viguinijpv/python310/bin/python3.10
"""Module 7 PCA-vs-GWAS network visualization and summary analysis.

This script builds a publication-style analysis focused on one convergent module:
module 7 from INTACT consensus clusters.

Outputs:
- Network plot (PNG, SVG) with node color = PCA-GWAS bias and size = mean score.
- Full gene table CSV for module 7.
- Scatter plot (GWAS vs PCA normalized scores).
- Optional module-7 GO top-term bar plot from existing Fisher outputs.
- Markdown report with summary stats and cautious interpretation.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

# Keep SVG text as editable text objects (not converted to paths).
plt.rcParams["svg.fonttype"] = "none"


PROJECT_ROOT = Path("/home/viguinijpv/200.18.99.75:8000/IC/src")
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ALS_TABLE_DIR = PROJECT_ROOT / "data" / "als_cs_gene_tables"

CONSENSUS_MODULE_PATH = REFERENCE_DIR / "intact_netw_consensus_clusters.csv"
NETWORK_PATH = REFERENCE_DIR / "intact_netw_filtered_networkx.obj"
HGNC_PATH = REFERENCE_DIR / "hgnc_complete_set.txt"
OT_JSON_PATH = PROJECT_ROOT / "external" / "OT-MONDO_0004976-associated-targets-2_12_2026-v25_12.json"

MODULE_ID = "7"


@dataclass
class InputBundle:
    module_enrichment_dir: Path
    pca_rank_path: Path
    gwas_rank_path: Path
    module_comparison_path: Optional[Path]
    go_dir: Optional[Path]
    go_bp_path: Optional[Path]
    go_cc_path: Optional[Path]
    go_mf_path: Optional[Path]
    model_run_dir: Optional[Path]
    pca_seed_path: Optional[Path]
    known_als_label_path: Optional[Path]


def normalize_gene_id(raw: object) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    if "." in s:
        s = s.split(".", 1)[0]
    return s if s else None


def normalize_symbol(raw: object) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    return s if s else None


def latest_path(paths: Sequence[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return sorted(existing, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def latest_dir_with_files(base_dir: Path, pattern: str, required_files: Sequence[str]) -> Path:
    candidates = [p for p in base_dir.glob(pattern) if p.is_dir()]
    filtered: List[Path] = []
    for d in candidates:
        if all((d / f).exists() for f in required_files):
            filtered.append(d)
    if not filtered:
        raise FileNotFoundError(
            f"No directory under {base_dir} matched {pattern} containing required files: {list(required_files)}"
        )
    return sorted(filtered, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def ensure_unique_out_dir(base_dir: Path, stem: str) -> Path:
    proposed = base_dir / stem
    if not proposed.exists():
        proposed.mkdir(parents=True, exist_ok=False)
        return proposed
    for i in range(1, 100):
        p = base_dir / f"{stem}_{i:02d}"
        if not p.exists():
            p.mkdir(parents=True, exist_ok=False)
            return p
    raise RuntimeError(f"Could not find a unique output folder for stem {stem}")


def find_inputs() -> InputBundle:
    module_dir = latest_dir_with_files(
        base_dir=REFERENCE_DIR,
        pattern="intact_ppr_module_enrichment*",
        required_files=["_rank_pca.csv", "_rank_gwas.csv"],
    )
    pca_rank_path = module_dir / "_rank_pca.csv"
    gwas_rank_path = module_dir / "_rank_gwas.csv"
    module_comparison_path = module_dir / "module_enrichment_comparison_summary.csv"
    if not module_comparison_path.exists():
        module_comparison_path = None

    go_dir: Optional[Path] = None
    go_bp_path: Optional[Path] = None
    go_cc_path: Optional[Path] = None
    go_mf_path: Optional[Path] = None
    go_candidates = [p for p in REFERENCE_DIR.glob("module_go_fisher_*") if p.is_dir()]
    go_candidates = sorted(go_candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    for d in go_candidates:
        bp = d / "module_7_go_biological_process_fisher.csv"
        cc = d / "module_7_go_cellular_component_fisher.csv"
        mf = d / "module_7_go_molecular_function_fisher.csv"
        if bp.exists() and cc.exists() and mf.exists():
            go_dir = d
            go_bp_path, go_cc_path, go_mf_path = bp, cc, mf
            break

    model_dirs = [p for p in ALS_TABLE_DIR.glob("global_functional_model_hpa_pca_*") if p.is_dir()]
    model_dirs = sorted(model_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    model_run_dir = model_dirs[0] if model_dirs else None

    pca_seed_path = None
    known_als_label_path = None
    if model_run_dir is not None:
        pca_seed_candidate = model_run_dir / "high_confidence_genes.csv"
        if pca_seed_candidate.exists():
            pca_seed_path = pca_seed_candidate
        known_als_candidate = model_run_dir / "all_gene_predictions.csv"
        if known_als_candidate.exists():
            known_als_label_path = known_als_candidate

    return InputBundle(
        module_enrichment_dir=module_dir,
        pca_rank_path=pca_rank_path,
        gwas_rank_path=gwas_rank_path,
        module_comparison_path=module_comparison_path,
        go_dir=go_dir,
        go_bp_path=go_bp_path,
        go_cc_path=go_cc_path,
        go_mf_path=go_mf_path,
        model_run_dir=model_run_dir,
        pca_seed_path=pca_seed_path,
        known_als_label_path=known_als_label_path,
    )


def load_module_genes(consensus_path: Path, module_id: str) -> List[str]:
    df = pd.read_csv(consensus_path, usecols=["Gene", "Cluster"])
    df["gene_id"] = df["Gene"].map(normalize_gene_id)
    df["cluster_id"] = df["Cluster"].astype(str).str.strip()
    genes = (
        df.loc[(df["cluster_id"] == str(module_id)) & (df["gene_id"].notna()), "gene_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not genes:
        raise ValueError(f"Module {module_id} not found in {consensus_path}")
    return sorted(genes)


def load_network(network_path: Path) -> nx.Graph:
    with open(network_path, "rb") as f:
        g = pickle.load(f)
    # Ensure string node identifiers.
    g = nx.relabel_nodes(g, {n: str(n) for n in g.nodes()})
    return g


def load_rank_score(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "gene_id" not in df.columns or "score" not in df.columns:
        raise ValueError(f"Ranking file missing gene_id/score columns: {path}")
    cur = df.copy()
    cur["gene_id"] = cur["gene_id"].map(normalize_gene_id)
    cur["score"] = pd.to_numeric(cur["score"], errors="coerce")
    cur = cur.dropna(subset=["gene_id", "score"]).drop_duplicates(subset=["gene_id"], keep="first")
    return pd.Series(cur["score"].values, index=cur["gene_id"].astype(str).values)


def load_hgnc_maps(hgnc_path: Path) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    if "symbol" not in hgnc.columns or "ensembl_gene_id" not in hgnc.columns:
        raise ValueError(f"HGNC file missing required columns: {hgnc_path}")

    cur = hgnc.copy()
    cur["symbol_norm"] = cur["symbol"].map(normalize_symbol)
    cur["gene_id"] = cur["ensembl_gene_id"].map(normalize_gene_id)
    cur = cur.dropna(subset=["symbol_norm", "gene_id"]).copy()

    # Prefer approved symbols when available.
    if "status" in cur.columns:
        cur["is_approved"] = cur["status"].astype(str).str.lower().eq("approved")
        cur = cur.sort_values(["is_approved"], ascending=[False], kind="stable")

    gene_to_symbol = (
        cur.drop_duplicates(subset=["gene_id"], keep="first")
        .set_index("gene_id")["symbol"]
        .astype(str)
        .to_dict()
    )

    grouped = cur.groupby("symbol_norm", as_index=False)["gene_id"].agg(lambda s: sorted(set(s.tolist())))
    symbol_to_genes = dict(zip(grouped["symbol_norm"], grouped["gene_id"]))
    return gene_to_symbol, symbol_to_genes


def load_pca_seed_genes(pca_seed_path: Optional[Path], network_nodes: Set[str]) -> Tuple[Set[str], str]:
    if pca_seed_path is None or not pca_seed_path.exists():
        return set(), "missing"
    df = pd.read_csv(pca_seed_path)
    if "gene_id" not in df.columns:
        return set(), "invalid_file_no_gene_id"
    seeds = set(df["gene_id"].map(normalize_gene_id).dropna().astype(str))
    mapped = seeds.intersection(network_nodes)
    return mapped, "high_confidence_genes_csv"


def load_gwas_seed_genes_from_ot(
    ot_json_path: Path,
    symbol_to_genes: Dict[str, List[str]],
    network_nodes: Set[str],
    source_column: str = "gwasCredibleSets",
    threshold: float = 0.5,
) -> Tuple[Set[str], str, int]:
    if not ot_json_path.exists():
        return set(), "missing_ot_json", 0
    with open(ot_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ot_df = pd.DataFrame(payload)
    if "symbol" not in ot_df.columns or source_column not in ot_df.columns:
        return set(), "invalid_ot_json_columns", 0

    ot_df["symbol_norm"] = ot_df["symbol"].map(normalize_symbol)
    ot_df["source_score"] = pd.to_numeric(ot_df[source_column], errors="coerce").fillna(0.0)
    sel = ot_df.loc[ot_df["source_score"] >= float(threshold)].copy()
    sel = sel.dropna(subset=["symbol_norm"])
    sel_count = int(len(sel))

    out: Set[str] = set()
    for sym in sel["symbol_norm"].astype(str).tolist():
        ids = symbol_to_genes.get(sym, [])
        if not ids:
            continue
        # Keep behavior deterministic with first sorted gene_id for ambiguous symbols.
        out.add(ids[0])

    mapped = out.intersection(network_nodes)
    return mapped, f"ot_json_{source_column}_ge_{threshold:.2f}", sel_count


def load_known_als_genes(label_path: Optional[Path]) -> Tuple[Set[str], str]:
    if label_path is None or not label_path.exists():
        return set(), "missing"
    df = pd.read_csv(label_path)
    if "gene_id" not in df.columns or "label_positive" not in df.columns:
        return set(), "invalid_file_missing_columns"
    cur = df.copy()
    cur["gene_id"] = cur["gene_id"].map(normalize_gene_id)
    cur["label_positive"] = pd.to_numeric(cur["label_positive"], errors="coerce").fillna(0.0)
    known = set(cur.loc[cur["label_positive"] > 0, "gene_id"].dropna().astype(str))
    return known, "all_gene_predictions_label_positive"


def safe_symbol(gene_id: str, gene_to_symbol: Dict[str, str]) -> str:
    sym = gene_to_symbol.get(gene_id, "")
    if sym is None:
        return ""
    return str(sym)


def build_module7_table(
    module_genes: Sequence[str],
    module_subgraph: nx.Graph,
    pca_scores: pd.Series,
    gwas_scores: pd.Series,
    gene_to_symbol: Dict[str, str],
    pca_seeds: Set[str],
    gwas_seeds: Set[str],
    known_als: Set[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for g in module_genes:
        pca = float(pca_scores.get(g, np.nan))
        gwas = float(gwas_scores.get(g, np.nan))
        mean_score = float(np.nanmean([pca, gwas])) if np.isfinite(pca) or np.isfinite(gwas) else np.nan
        bias = float(pca - gwas) if np.isfinite(pca) and np.isfinite(gwas) else np.nan
        is_pca_seed = g in pca_seeds
        is_gwas_seed = g in gwas_seeds
        rows.append(
            {
                "gene_id": g,
                "gene_symbol": safe_symbol(g, gene_to_symbol),
                "normalized_pca_score": pca,
                "normalized_gwas_score": gwas,
                "mean_score": mean_score,
                "bias_score": bias,
                "is_pca_seed": bool(is_pca_seed),
                "is_gwas_seed": bool(is_gwas_seed),
                "is_shared_seed": bool(is_pca_seed and is_gwas_seed),
                "is_known_als_gene": bool(g in known_als),
                "degree_in_module_subgraph": int(module_subgraph.degree(g)) if g in module_subgraph else 0,
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["mean_score", "bias_score"], ascending=[False, False], kind="stable").reset_index(drop=True)
    return df


def pick_label_genes(df: pd.DataFrame, max_labels: int = 34) -> List[str]:
    label_order: List[str] = []
    for subset in [
        df.nlargest(12, "mean_score"),
        df.nlargest(9, "bias_score"),
        df.nsmallest(9, "bias_score"),
        df.loc[df["is_shared_seed"]],
        df.loc[df["is_pca_seed"] & ~df["is_shared_seed"]],
        df.loc[df["is_gwas_seed"] & ~df["is_shared_seed"]],
        df.loc[df["is_known_als_gene"]],
    ]:
        for g in subset["gene_id"].astype(str).tolist():
            if g not in label_order:
                label_order.append(g)
    return label_order[:max_labels]


def pick_visualization_nodes(df: pd.DataFrame, top_n_mean: int = 55) -> Tuple[List[str], str]:
    top_mean = df.nlargest(top_n_mean, "mean_score")["gene_id"].astype(str).tolist()
    important = (
        pd.concat(
            [
                df.nlargest(10, "bias_score"),
                df.nsmallest(10, "bias_score"),
                df.loc[df["is_pca_seed"] | df["is_gwas_seed"] | df["is_known_als_gene"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=["gene_id"], keep="first")["gene_id"]
        .astype(str)
        .tolist()
    )
    chosen: List[str] = []
    for g in top_mean + important:
        if g not in chosen:
            chosen.append(g)
    reason = f"Top-{top_n_mean} by mean score + seeds/known ALS + strongest bias genes"
    return chosen, reason


def ensure_seed_genes_in_vis_nodes(df: pd.DataFrame, vis_nodes: Sequence[str]) -> List[str]:
    out = list(vis_nodes)
    seed_nodes = (
        df.loc[df["is_pca_seed"] | df["is_gwas_seed"], "gene_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    for g in seed_nodes:
        if g not in out:
            out.append(g)
    return out


def make_network_plot(
    df_full: pd.DataFrame,
    module_subgraph: nx.Graph,
    vis_nodes: Sequence[str],
    label_genes: Sequence[str],
    out_png: Path,
    out_svg: Path,
    out_pdf: Optional[Path] = None,
    full_module_gene_count: Optional[int] = None,
) -> Dict[str, object]:
    vis_nodes = ensure_seed_genes_in_vis_nodes(df_full, vis_nodes)
    vis_set = set(vis_nodes)
    vis_nodes_final = [n for n in module_subgraph.nodes() if n in vis_set]
    vis_g = module_subgraph.subgraph(vis_nodes_final).copy()
    if vis_g.number_of_nodes() == 0:
        raise RuntimeError("Visualization subgraph is empty.")

    lookup = df_full.set_index("gene_id")
    bias_vals = np.array([float(lookup.loc[n, "bias_score"]) for n in vis_g.nodes()], dtype=float)
    mean_vals = np.array([float(lookup.loc[n, "mean_score"]) for n in vis_g.nodes()], dtype=float)

    finite_mean = np.isfinite(mean_vals)
    if finite_mean.any() and float(np.nanmax(mean_vals) - np.nanmin(mean_vals)) > 0:
        min_s, max_s = float(np.nanmin(mean_vals)), float(np.nanmax(mean_vals))
        node_sizes = 170.0 + ((mean_vals - min_s) / (max_s - min_s)) * 880.0
    else:
        node_sizes = np.full(shape=len(mean_vals), fill_value=520.0, dtype=float)

    vmax = float(np.nanmax(np.abs(bias_vals))) if np.isfinite(bias_vals).any() else 1.0
    vmax = max(vmax, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    node_edge_colors: List[str] = []
    node_line_widths: List[float] = []
    for n in vis_g.nodes():
        row = lookup.loc[n]
        shared = bool(row["is_shared_seed"])
        pca_seed = bool(row["is_pca_seed"])
        gwas_seed = bool(row["is_gwas_seed"])
        if shared:
            node_edge_colors.append("#8e2c8a")
            node_line_widths.append(3.1)
        elif pca_seed:
            node_edge_colors.append("#0f5fbf")
            node_line_widths.append(2.6)
        elif gwas_seed:
            node_edge_colors.append("#c43d1a")
            node_line_widths.append(2.6)
        else:
            node_edge_colors.append("#484848")
            node_line_widths.append(0.50)

    k = 1.65 / math.sqrt(max(vis_g.number_of_nodes(), 1))
    pos = nx.spring_layout(vis_g, seed=42, k=k, iterations=300)
    xs = [pos[n][0] for n in vis_g.nodes()]
    ys = [pos[n][1] for n in vis_g.nodes()]

    fig, ax = plt.subplots(figsize=(15, 11))
    nx.draw_networkx_edges(
        vis_g,
        pos=pos,
        ax=ax,
        width=0.25,
        alpha=0.12,
        edge_color="#b4b4b4",
    )
    ax.scatter(
        xs,
        ys,
        c=bias_vals,
        s=node_sizes,
        cmap=cmap,
        norm=norm,
        edgecolors=node_edge_colors,
        linewidths=node_line_widths,
        alpha=0.94,
        zorder=3,
    )

    # Label a selected subset only, with simple overlap-reduction offsets.
    if xs and ys:
        x_span = float(max(xs) - min(xs))
        y_span = float(max(ys) - min(ys))
    else:
        x_span = 1.0
        y_span = 1.0
    dx = max(0.012, 0.025 * x_span)
    dy = max(0.012, 0.025 * y_span)
    candidate_offsets = [
        (0.0, 0.0),
        (dx, dy),
        (-dx, dy),
        (dx, -dy),
        (-dx, -dy),
        (1.6 * dx, 0.0),
        (-1.6 * dx, 0.0),
        (0.0, 1.6 * dy),
        (0.0, -1.6 * dy),
    ]
    placed: List[Tuple[float, float]] = []
    min_label_dist = 0.04 * max(x_span, y_span, 1.0)

    for g in label_genes:
        if g not in pos:
            continue
        row = lookup.loc[g]
        label = str(row["gene_symbol"]).strip() if str(row["gene_symbol"]).strip() else g
        x, y = pos[g]
        chosen = candidate_offsets[0]
        for ox, oy in candidate_offsets:
            tx, ty = x + ox, y + oy
            if not placed:
                chosen = (ox, oy)
                break
            dmin = min(math.hypot(tx - px, ty - py) for px, py in placed)
            if dmin >= min_label_dist:
                chosen = (ox, oy)
                break
        tx, ty = x + chosen[0], y + chosen[1]
        if chosen != (0.0, 0.0):
            ax.plot([x, tx], [y, ty], color="#9a9a9a", linewidth=0.45, alpha=0.65, zorder=4)
        ax.text(
            tx,
            ty,
            label,
            fontsize=7.2,
            color="#111111",
            ha="center",
            va="center",
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, boxstyle="round,pad=0.12"),
        )
        placed.append((tx, ty))

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Bias = normalized PCA/text-model score - normalized GWAS score", fontsize=10)
    fig.text(
        0.83,
        0.23,
        "red/orange = more PCA/text-model-biased\nblue = more GWAS-biased\nneutral = similar scores",
        fontsize=8.5,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="#d9d9d9", alpha=0.80, boxstyle="round,pad=0.25"),
    )

    q_vals = np.nanpercentile(mean_vals[np.isfinite(mean_vals)], [25, 50, 75]) if np.isfinite(mean_vals).any() else [0, 0, 0]
    q_sizes = np.nanpercentile(node_sizes, [25, 50, 75]) if np.isfinite(node_sizes).any() else [260, 540, 860]
    size_handles = [
        ax.scatter(
            [],
            [],
            s=float(q_sizes[i]),
            c="#f2f2f2",
            edgecolors="#444444",
            linewidths=0.8,
            label=f"mean ~ {q_vals[i]:.2f}",
        )
        for i in range(3)
    ]

    seed_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f2f2f2", markeredgecolor="#0f5fbf", markeredgewidth=2.6, markersize=9, label="PCA seed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f2f2f2", markeredgecolor="#c43d1a", markeredgewidth=2.6, markersize=9, label="GWAS seed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f2f2f2", markeredgecolor="#8e2c8a", markeredgewidth=3.1, markersize=9, label="Shared seed"),
    ]
    seed_legend = ax.legend(
        handles=seed_handles,
        loc="upper left",
        frameon=True,
        fontsize=8,
        title="Node border = seed status",
        title_fontsize=9,
    )
    ax.add_artist(seed_legend)
    ax.legend(
        handles=size_handles,
        loc="upper right",
        frameon=True,
        fontsize=8,
        title="Node size = mean normalized propagation score",
        title_fontsize=9,
    )

    full_n = int(full_module_gene_count) if full_module_gene_count is not None else int(module_subgraph.number_of_nodes())
    ax.set_title(
        "Module 7 PCA/Text-model vs GWAS Network\n"
        f"Module 7 synaptic subnetwork: visualized high-score/high-bias subset ({vis_g.number_of_nodes()}/{full_n} genes)\n"
        "Full Module 7 gene table is saved separately",
        fontsize=13,
    )
    ax.set_axis_off()
    fig.tight_layout()

    # Keep artists as vector in SVG/PDF (no global rasterization).
    for coll in ax.collections:
        coll.set_rasterized(False)
    for line in ax.lines:
        line.set_rasterized(False)
    if hasattr(cbar, "solids") and cbar.solids is not None:
        cbar.solids.set_rasterized(False)

    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "visualized_nodes": int(vis_g.number_of_nodes()),
        "visualized_edges": int(vis_g.number_of_edges()),
        "all_module_seed_nodes_included": bool(
            set(
                df_full.loc[df_full["is_pca_seed"] | df_full["is_gwas_seed"], "gene_id"]
                .astype(str)
                .tolist()
            ).issubset(set(vis_g.nodes()))
        ),
    }


def make_scatter_plot(
    df: pd.DataFrame,
    label_genes: Sequence[str],
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    categories = [
        ("shared_seed", df["is_shared_seed"], "#7b3294"),
        ("pca_seed_only", df["is_pca_seed"] & ~df["is_shared_seed"], "#2166ac"),
        ("gwas_seed_only", df["is_gwas_seed"] & ~df["is_shared_seed"], "#b2182b"),
        ("non_seed", ~(df["is_pca_seed"] | df["is_gwas_seed"]), "#757575"),
    ]
    for label, mask, color in categories:
        cur = df.loc[mask].copy()
        if cur.empty:
            continue
        ax.scatter(
            cur["normalized_gwas_score"].astype(float).values,
            cur["normalized_pca_score"].astype(float).values,
            s=40,
            c=color,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.4,
            label=label,
        )

    known = df.loc[df["is_known_als_gene"]].copy()
    if not known.empty:
        ax.scatter(
            known["normalized_gwas_score"].astype(float).values,
            known["normalized_pca_score"].astype(float).values,
            s=90,
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
            label="known ALS label",
        )

    all_x = df["normalized_gwas_score"].astype(float).values
    all_y = df["normalized_pca_score"].astype(float).values
    low = float(min(np.nanmin(all_x), np.nanmin(all_y)))
    high = float(max(np.nanmax(all_x), np.nanmax(all_y)))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.0, color="#4d4d4d", alpha=0.8)

    for g in label_genes:
        row = df.loc[df["gene_id"] == g]
        if row.empty:
            continue
        x = float(row["normalized_gwas_score"].iloc[0])
        y = float(row["normalized_pca_score"].iloc[0])
        sym = str(row["gene_symbol"].iloc[0]).strip()
        text = sym if sym else g
        ax.text(
            x,
            y,
            text,
            fontsize=7.0,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, boxstyle="round,pad=0.10"),
        )

    ax.set_xlabel("Normalized GWAS propagation score")
    ax.set_ylabel("Normalized PCA/text-model propagation score")
    ax.set_title("Module 7: GWAS vs PCA/Text-model Scores")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=True, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_go_plot(
    go_bp_path: Optional[Path],
    go_cc_path: Optional[Path],
    go_mf_path: Optional[Path],
    out_png: Path,
) -> Tuple[Optional[pd.DataFrame], str]:
    if go_bp_path is None or go_cc_path is None or go_mf_path is None:
        return None, "go_files_missing"

    frames: List[pd.DataFrame] = []
    for cat, path in [
        ("BP", go_bp_path),
        ("CC", go_cc_path),
        ("MF", go_mf_path),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "fdr_within_module_category" not in df.columns:
            continue
        cur = df.copy()
        cur["cat_short"] = cat
        frames.append(cur)
    if not frames:
        return None, "go_files_invalid"

    all_go = pd.concat(frames, ignore_index=True)
    all_go["fdr_within_module_category"] = pd.to_numeric(all_go["fdr_within_module_category"], errors="coerce")
    all_go = all_go.dropna(subset=["fdr_within_module_category", "go_term_id", "go_name", "cat_short"]).copy()
    sig = all_go.loc[all_go["fdr_within_module_category"] <= 0.05].copy()
    if sig.empty:
        plot_df = all_go.nsmallest(12, "fdr_within_module_category").copy()
    else:
        plot_df = sig.nsmallest(12, "fdr_within_module_category").copy()

    plot_df["score"] = -np.log10(np.clip(plot_df["fdr_within_module_category"].astype(float).values, 1e-300, 1.0))
    plot_df["label"] = plot_df["cat_short"].astype(str) + ": " + plot_df["go_name"].astype(str)
    plot_df = plot_df.iloc[::-1].copy()

    cat_colors = {"BP": "#4c78a8", "CC": "#f58518", "MF": "#54a24b"}
    colors = [cat_colors.get(c, "#777777") for c in plot_df["cat_short"].astype(str).tolist()]

    fig, ax = plt.subplots(figsize=(12, max(6, 0.43 * len(plot_df))))
    ax.barh(np.arange(len(plot_df)), plot_df["score"].values, color=colors)
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["label"].tolist(), fontsize=8)
    ax.set_xlabel("-log10(FDR within module/category)")
    ax.set_title("Module 7 Top GO Terms")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_df, "ok"


def fmt_top_table(df: pd.DataFrame, cols: Sequence[str], n: int = 10) -> str:
    cur = df.loc[:, list(cols)].head(int(n)).copy()
    if cur.empty:
        return "_No rows._"
    for c in cur.columns:
        if cur[c].dtype.kind in {"f"}:
            cur[c] = cur[c].map(lambda x: f"{x:.6g}")
    header = "| " + " | ".join(cur.columns.tolist()) + " |"
    sep = "| " + " | ".join(["---"] * len(cur.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in cur.astype(str).values.tolist()]
    return "\n".join([header, sep] + rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 7 PCA/text-model vs GWAS network analysis.")
    parser.add_argument(
        "--improve-existing-output-dir",
        type=Path,
        default=None,
        help=(
            "If provided, only regenerate improved network visualization files in this existing output directory "
            "(no table/report/scatter/GO overwrite)."
        ),
    )
    parser.add_argument(
        "--write-pdf",
        action="store_true",
        help="Also export `module_7_pca_vs_gwas_network_improved.pdf` when improving an existing output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = find_inputs()
    improve_mode = args.improve_existing_output_dir is not None
    if improve_mode:
        out_dir = Path(args.improve_existing_output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = ensure_unique_out_dir(
            base_dir=REFERENCE_DIR,
            stem=f"module_7_pca_vs_gwas_network_{date.today().isoformat()}",
        )

    module_genes = load_module_genes(CONSENSUS_MODULE_PATH, MODULE_ID)
    graph = load_network(NETWORK_PATH)
    network_nodes = set(graph.nodes())

    module_genes_in_network = [g for g in module_genes if g in network_nodes]
    module_subgraph = graph.subgraph(module_genes_in_network).copy()

    pca_scores = load_rank_score(inputs.pca_rank_path)
    gwas_scores = load_rank_score(inputs.gwas_rank_path)

    gene_to_symbol, symbol_to_genes = load_hgnc_maps(HGNC_PATH)

    pca_seed_set, pca_seed_source = load_pca_seed_genes(inputs.pca_seed_path, network_nodes)
    gwas_seed_set, gwas_seed_source, gwas_ot_selected_rows = load_gwas_seed_genes_from_ot(
        ot_json_path=OT_JSON_PATH,
        symbol_to_genes=symbol_to_genes,
        network_nodes=network_nodes,
        source_column="gwasCredibleSets",
        threshold=0.5,
    )
    known_als_set, known_als_source = load_known_als_genes(inputs.known_als_label_path)

    df = build_module7_table(
        module_genes=module_genes_in_network,
        module_subgraph=module_subgraph,
        pca_scores=pca_scores,
        gwas_scores=gwas_scores,
        gene_to_symbol=gene_to_symbol,
        pca_seeds=pca_seed_set,
        gwas_seeds=gwas_seed_set,
        known_als=known_als_set,
    )

    # Save full module table only in full-analysis mode.
    gene_table_path = out_dir / "module_7_pca_vs_gwas_gene_table.csv"
    if not improve_mode:
        df.to_csv(gene_table_path, index=False)

    # Build visualization subset for dense subgraph.
    vis_nodes, vis_rule = pick_visualization_nodes(df, top_n_mean=55)
    vis_nodes = [n for n in vis_nodes if n in module_subgraph]
    vis_graph = module_subgraph.subgraph(vis_nodes).copy()
    if vis_graph.number_of_edges() > 1200:
        vis_nodes, vis_rule = pick_visualization_nodes(df, top_n_mean=45)
        vis_nodes = [n for n in vis_nodes if n in module_subgraph]
        vis_graph = module_subgraph.subgraph(vis_nodes).copy()
    if vis_graph.number_of_edges() > 850:
        vis_nodes, vis_rule = pick_visualization_nodes(df, top_n_mean=35)
        vis_nodes = [n for n in vis_nodes if n in module_subgraph]
        vis_graph = module_subgraph.subgraph(vis_nodes).copy()

    label_genes = pick_label_genes(df, max_labels=34)

    if improve_mode:
        network_png = out_dir / "module_7_pca_vs_gwas_network_improved.png"
        network_svg = out_dir / "module_7_pca_vs_gwas_network_improved.svg"
        network_pdf = out_dir / "module_7_pca_vs_gwas_network_improved.pdf" if args.write_pdf else None
    else:
        network_png = out_dir / "module_7_pca_vs_gwas_network.png"
        network_svg = out_dir / "module_7_pca_vs_gwas_network.svg"
        network_pdf = None
    net_info = make_network_plot(
        df_full=df,
        module_subgraph=module_subgraph,
        vis_nodes=vis_nodes,
        label_genes=label_genes,
        out_png=network_png,
        out_svg=network_svg,
        out_pdf=network_pdf,
        full_module_gene_count=len(module_genes),
    )

    if improve_mode:
        corr_df = df.dropna(subset=["normalized_pca_score", "normalized_gwas_score"]).copy()
        pearson_r, _pearson_p = stats.pearsonr(
            corr_df["normalized_pca_score"].astype(float).values,
            corr_df["normalized_gwas_score"].astype(float).values,
        )
        spearman_rho, _spearman_p = stats.spearmanr(
            corr_df["normalized_pca_score"].astype(float).values,
            corr_df["normalized_gwas_score"].astype(float).values,
        )
        print(f"Improved network output directory: {out_dir}")
        print(f"Improved PNG: {network_png}")
        print(f"Improved SVG: {network_svg}")
        if network_pdf is not None:
            print(f"Improved PDF: {network_pdf}")
        print(f"Module 7 genes total: {len(module_genes)}")
        print(f"Module 7 genes in network: {len(module_genes_in_network)}")
        print(f"Genes with both PCA and GWAS scores: {len(corr_df)}")
        print(f"Full subgraph edges: {module_subgraph.number_of_edges()}")
        print(f"Visualized nodes: {net_info['visualized_nodes']}")
        print(f"Visualized edges: {net_info['visualized_edges']}")
        print(f"All seed genes in Module 7 included in visualization: {net_info['all_module_seed_nodes_included']}")
        print(f"Pearson r: {pearson_r:.4f}")
        print(f"Spearman rho: {spearman_rho:.4f}")
        return

    scatter_png = out_dir / "module_7_pca_vs_gwas_score_scatter.png"
    make_scatter_plot(df=df, label_genes=label_genes, out_png=scatter_png)

    go_png = out_dir / "module_7_top_go_terms.png"
    go_df, go_status = make_go_plot(
        go_bp_path=inputs.go_bp_path,
        go_cc_path=inputs.go_cc_path,
        go_mf_path=inputs.go_mf_path,
        out_png=go_png,
    )

    # Correlations inside module 7.
    corr_df = df.dropna(subset=["normalized_pca_score", "normalized_gwas_score"]).copy()
    pearson_r, pearson_p = stats.pearsonr(
        corr_df["normalized_pca_score"].astype(float).values,
        corr_df["normalized_gwas_score"].astype(float).values,
    )
    spearman_rho, spearman_p = stats.spearmanr(
        corr_df["normalized_pca_score"].astype(float).values,
        corr_df["normalized_gwas_score"].astype(float).values,
    )

    top_mean = df.nlargest(10, "mean_score")[["gene_id", "gene_symbol", "mean_score", "bias_score"]].copy()
    top_pca_bias = df.nlargest(10, "bias_score")[["gene_id", "gene_symbol", "bias_score", "mean_score"]].copy()
    top_gwas_bias = df.nsmallest(10, "bias_score")[["gene_id", "gene_symbol", "bias_score", "mean_score"]].copy()

    mod7_cmp_row = None
    if inputs.module_comparison_path is not None and inputs.module_comparison_path.exists():
        cmp_df = pd.read_csv(inputs.module_comparison_path)
        cmp_df["module_id"] = cmp_df["module_id"].astype(str)
        row = cmp_df.loc[cmp_df["module_id"] == MODULE_ID]
        if not row.empty:
            mod7_cmp_row = row.iloc[0].to_dict()

    pca_seed_in_module = int(df["is_pca_seed"].sum())
    gwas_seed_in_module = int(df["is_gwas_seed"].sum())
    shared_seed_in_module = int(df["is_shared_seed"].sum())

    # Report.
    report_path = out_dir / "module_7_pca_vs_gwas_network_report.md"
    report_lines: List[str] = []
    report_lines.append("# Module 7 PCA-vs-GWAS Network Report")
    report_lines.append("")
    report_lines.append("## 1. Why Module 7 Was Selected")
    report_lines.append(
        "- Module 7 was selected as a strongly convergent module in prior PCA/text-model and GWAS module enrichment outputs."
    )
    if mod7_cmp_row is not None:
        report_lines.append(
            f"- Module-level enrichment snapshot: NES_pca={float(mod7_cmp_row.get('NES_pca', np.nan)):.3f}, "
            f"NES_gwas={float(mod7_cmp_row.get('NES_gwas', np.nan)):.3f}, "
            f"padj_pca={float(mod7_cmp_row.get('padj_pca', np.nan)):.3g}, "
            f"padj_gwas={float(mod7_cmp_row.get('padj_gwas', np.nan)):.3g}."
        )
    report_lines.append("- Prior GO enrichments for this module include synapse-related terms (for example synapse, glutamatergic synapse, postsynaptic membrane, chemical synaptic transmission).")
    report_lines.append("")
    report_lines.append("## 2. Inputs Used")
    report_lines.append(f"- Consensus module file: `{CONSENSUS_MODULE_PATH}`")
    report_lines.append(f"- Interaction network: `{NETWORK_PATH}`")
    report_lines.append(f"- PCA normalized score source used: `{inputs.pca_rank_path}`")
    report_lines.append(f"- GWAS normalized score source used: `{inputs.gwas_rank_path}`")
    report_lines.append(f"- PCA seed source: `{pca_seed_source}` (`{inputs.pca_seed_path}`)")
    report_lines.append(
        f"- GWAS seed source: `{gwas_seed_source}` (from `{OT_JSON_PATH}`, selected OT rows before symbol dedup={gwas_ot_selected_rows})"
    )
    report_lines.append(f"- Known ALS label source: `{known_als_source}` (`{inputs.known_als_label_path}`)")
    if inputs.go_dir is not None:
        report_lines.append(f"- GO Fisher source directory: `{inputs.go_dir}`")
    report_lines.append("")
    report_lines.append("## 3. Formulas Used")
    report_lines.append("- `mean_score = (normalized_pca_score + normalized_gwas_score) / 2`")
    report_lines.append("- `bias_score = normalized_pca_score - normalized_gwas_score`")
    report_lines.append(
        "- Node-size scaling (visualization): linear min-max scaling of `mean_score` to marker size range `[170, 1050]`."
    )
    report_lines.append("")
    report_lines.append("## 4. Module 7 Counts")
    report_lines.append(f"- Number of genes in Module 7: `{len(module_genes)}`")
    report_lines.append(f"- Number of Module 7 genes in network object: `{len(module_genes_in_network)}`")
    report_lines.append(f"- Number of genes mapped to both PCA and GWAS scores: `{len(corr_df)}`")
    report_lines.append(f"- Number of edges in full Module 7 induced subgraph: `{module_subgraph.number_of_edges()}`")
    report_lines.append(f"- Number of PCA/text-model seeds in Module 7: `{pca_seed_in_module}`")
    report_lines.append(f"- Number of GWAS seeds in Module 7: `{gwas_seed_in_module}`")
    report_lines.append(f"- Number of shared seeds in Module 7: `{shared_seed_in_module}`")
    report_lines.append("")
    report_lines.append("## 5. Visualization Filtering")
    report_lines.append(
        f"- Full Module 7 graph is dense (`{module_subgraph.number_of_nodes()}` nodes, `{module_subgraph.number_of_edges()}` edges), so visualization used a reduced subgraph."
    )
    report_lines.append(f"- Rule used: `{vis_rule}`")
    report_lines.append(
        f"- Visualized subgraph: `{net_info['visualized_nodes']}` nodes, `{net_info['visualized_edges']}` edges."
    )
    report_lines.append("")
    report_lines.append("## 6. Top Genes by Average Score")
    report_lines.append(fmt_top_table(top_mean, ["gene_id", "gene_symbol", "mean_score", "bias_score"], n=10))
    report_lines.append("")
    report_lines.append("## 7. Strongest PCA-Biased Genes")
    report_lines.append(fmt_top_table(top_pca_bias, ["gene_id", "gene_symbol", "bias_score", "mean_score"], n=10))
    report_lines.append("")
    report_lines.append("## 8. Strongest GWAS-Biased Genes")
    report_lines.append(fmt_top_table(top_gwas_bias, ["gene_id", "gene_symbol", "bias_score", "mean_score"], n=10))
    report_lines.append("")
    report_lines.append("## 9. Correlation (Module 7)")
    report_lines.append(f"- Pearson r: `{pearson_r:.4f}` (p=`{pearson_p:.3g}`)")
    report_lines.append(f"- Spearman rho: `{spearman_rho:.4f}` (p=`{spearman_p:.3g}`)")
    report_lines.append("")
    report_lines.append("## 10. Cautious Interpretation")
    if spearman_rho >= 0.5:
        interp = "PCA/text-model and GWAS show notable within-module agreement, with additional source-specific ranking differences."
    elif spearman_rho >= 0.2:
        interp = "PCA/text-model and GWAS partially agree inside this synaptic module, while still emphasizing different genes/subregions."
    else:
        interp = "PCA/text-model and GWAS converge on the same synaptic module but prioritize substantially different genes/subregions."
    report_lines.append(f"- {interp}")
    report_lines.append("- Interpretation is descriptive and hypothesis-generating; it does not establish causal biology.")
    report_lines.append("")
    report_lines.append("## 11. Output Files")
    report_lines.append(f"- Full gene table CSV: `{gene_table_path.name}`")
    report_lines.append(f"- Network plot PNG: `{network_png.name}`")
    report_lines.append(f"- Network plot SVG: `{network_svg.name}`")
    report_lines.append(f"- Score scatter PNG: `{scatter_png.name}`")
    if go_status == "ok":
        report_lines.append(f"- GO terms PNG: `{go_png.name}`")
    else:
        report_lines.append(f"- GO terms PNG not generated (`{go_status}`)")
    report_lines.append("")
    report_lines.append("## 12. Notes on Missing Original Three-Way Folder")
    report_lines.append(
        "- The original referenced three-way output directory (`intact_ppr_functional_text_hpa_vs_clinvar_vs_gwas_2026-04-17`) is not present in this workspace."
    )
    report_lines.append(
        "- Therefore, this analysis uses the latest available derived ranking files (`_rank_pca.csv`, `_rank_gwas.csv`) from module enrichment outputs and available seed sources."
    )

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Save concise machine-readable summary.
    summary = {
        "module_id": MODULE_ID,
        "out_dir": str(out_dir),
        "inputs": {
            "consensus_module_path": str(CONSENSUS_MODULE_PATH),
            "network_path": str(NETWORK_PATH),
            "pca_rank_path": str(inputs.pca_rank_path),
            "gwas_rank_path": str(inputs.gwas_rank_path),
            "module_enrichment_dir": str(inputs.module_enrichment_dir),
            "pca_seed_source": pca_seed_source,
            "pca_seed_path": str(inputs.pca_seed_path) if inputs.pca_seed_path is not None else None,
            "gwas_seed_source": gwas_seed_source,
            "ot_json_path": str(OT_JSON_PATH),
            "known_als_source": known_als_source,
            "known_als_label_path": str(inputs.known_als_label_path) if inputs.known_als_label_path is not None else None,
            "go_dir": str(inputs.go_dir) if inputs.go_dir is not None else None,
        },
        "counts": {
            "module_genes_total": int(len(module_genes)),
            "module_genes_in_network": int(len(module_genes_in_network)),
            "module_edges_full_subgraph": int(module_subgraph.number_of_edges()),
            "module_genes_with_both_scores": int(len(corr_df)),
            "pca_seed_in_module": pca_seed_in_module,
            "gwas_seed_in_module": gwas_seed_in_module,
            "shared_seed_in_module": shared_seed_in_module,
            "known_als_in_module": int(df["is_known_als_gene"].sum()),
            "visualized_nodes": int(net_info["visualized_nodes"]),
            "visualized_edges": int(net_info["visualized_edges"]),
        },
        "correlation": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
        },
        "outputs": {
            "gene_table_csv": str(gene_table_path),
            "network_png": str(network_png),
            "network_svg": str(network_svg),
            "scatter_png": str(scatter_png),
            "go_png": str(go_png) if go_status == "ok" else None,
            "report_md": str(report_path),
        },
    }
    (out_dir / "module_7_pca_vs_gwas_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Output directory: {out_dir}")
    print(f"Module 7 genes total: {len(module_genes)}")
    print(f"Module 7 genes in network: {len(module_genes_in_network)}")
    print(f"Genes with both PCA and GWAS scores: {len(corr_df)}")
    print(f"Full subgraph edges: {module_subgraph.number_of_edges()}")
    print(f"Pearson r: {pearson_r:.4f}")
    print(f"Spearman rho: {spearman_rho:.4f}")
    print(f"PCA rank file: {inputs.pca_rank_path}")
    print(f"GWAS rank file: {inputs.gwas_rank_path}")
    print(f"PCA seed source: {pca_seed_source}")
    print(f"GWAS seed source: {gwas_seed_source}")
    print(f"Known ALS source: {known_als_source}")


if __name__ == "__main__":
    main()

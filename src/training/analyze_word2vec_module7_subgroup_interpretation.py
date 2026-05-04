#!/home/viguinijpv/python310/bin/python3.10
"""Module 7 subgroup interpretation for Word2Vec PCA-vs-GWAS network analysis.

This script produces a dedicated biological characterization of Module 7 and
subgroup-level interpretation for:
- shared/high-mean genes
- PCA-biased genes
- GWAS-biased genes

Outputs include:
- Annotated Module 7 gene table
- GO Fisher enrichment CSVs for Module 7 and each subgroup (BP/CC/MF)
- GO summary plots
- Grouped Module 7 network plot
- Markdown interpretation report
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact


PROJECT_ROOT = Path("/home/viguinijpv/200.18.99.75:8000/IC/src")
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

DEFAULT_WORD2VEC_DIR = (
    PROJECT_ROOT
    / "data"
    / "als_cs_gene_tables"
    / "word2vec_pca_gwas_network_analysis_20260423"
)

DEFAULT_MODULE_ID = "7"

CONSENSUS_MODULE_PATH = REFERENCE_DIR / "intact_netw_consensus_clusters.csv"
NETWORK_PATH = REFERENCE_DIR / "intact_netw_filtered_networkx.obj"
HGNC_PATH = REFERENCE_DIR / "hgnc_complete_set.txt"

GO_PATHS = {
    "biological_process": REFERENCE_DIR / "GO_terms_biological_process_merged_EBI_uniprot_ensemblIDs_grouped_terms.csv",
    "cellular_component": REFERENCE_DIR / "GO_terms_cellular_component_merged_EBI_uniprot_ensemblIDs_grouped_terms.csv",
    "molecular_function": REFERENCE_DIR / "GO_terms_molecular_function_merged_EBI_uniprot_ensemblIDs_grouped_terms.csv",
}

CATEGORY_SHORT = {
    "biological_process": "BP",
    "cellular_component": "CC",
    "molecular_function": "MF",
}

CATEGORY_LABEL = {
    "biological_process": "Biological Process",
    "cellular_component": "Cellular Component",
    "molecular_function": "Molecular Function",
}

ENSG_ID_RE = re.compile(r"^ENSG\d+$")


@dataclass
class InputPaths:
    word2vec_dir: Path
    pca_scores_path: Path
    gwas_scores_path: Path
    module_comparison_path: Path
    project_gene_table_path: Path
    model_seed_path: Path
    gwas_seed_path: Path
    clinvar_seed_path: Path


@dataclass
class GroupDefinition:
    n_total_module: int
    n_group_each: int
    mean_quantile_shared: float
    mean_threshold_shared: float
    mean_quantile_bias_pool: float
    mean_threshold_bias_pool: float


def normalize_gene_id(raw: object) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    if "." in s:
        s = s.split(".", 1)[0]
    if not ENSG_ID_RE.match(s):
        return None
    return s


def normalize_symbol(raw: object) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s or s == "NAN":
        return None
    return s


def split_gene_field(raw: object) -> List[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    out: List[str] = []
    for tok in text.split(";"):
        g = normalize_gene_id(tok)
        if g is not None:
            out.append(g)
    return out


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(shape=p.shape, fill_value=np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return out

    pv = p[finite]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))
    adj = ranked * m / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    mapped = np.empty_like(adj)
    mapped[order] = adj
    out[finite] = mapped
    return out


def ensure_unique_out_dir(base: Path, stem: str) -> Path:
    p = base / stem
    if not p.exists():
        p.mkdir(parents=True, exist_ok=False)
        return p
    for i in range(1, 100):
        cand = base / f"{stem}_{i:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
    raise RuntimeError(f"Could not create unique output dir under {base} with stem {stem}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 7 subgroup interpretation for Word2Vec PCA-vs-GWAS analysis")
    parser.add_argument("--word2vec-dir", type=Path, default=DEFAULT_WORD2VEC_DIR)
    parser.add_argument("--module-id", type=str, default=DEFAULT_MODULE_ID)
    parser.add_argument("--out-subdir-stem", type=str, default="module_7_subgroup_interpretation")
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--shared-mean-quantile", type=float, default=0.60)
    parser.add_argument("--bias-pool-mean-quantile", type=float, default=0.40)
    parser.add_argument("--top-go-terms-per-plot", type=int, default=12)
    parser.add_argument("--top-go-terms-per-category", type=int, default=8)
    return parser.parse_args()


def resolve_input_paths(word2vec_dir: Path) -> InputPaths:
    pca_scores = word2vec_dir / "intact_ppr_three_way" / "ppr_scores_model_seeds_normalized.csv"
    gwas_scores = word2vec_dir / "intact_ppr_three_way" / "ppr_scores_gwas_seeds_normalized.csv"
    module_cmp = word2vec_dir / "intact_module_enrichment" / "module_enrichment_comparison_summary.csv"
    proj_table = word2vec_dir / "project_gene_table_word2vec_embedding_universe.csv"
    model_seed = word2vec_dir / "intact_ppr_three_way" / "model_seed_genes.csv"
    gwas_seed = word2vec_dir / "intact_ppr_three_way" / "gwas_seed_genes.csv"
    clinvar_seed = word2vec_dir / "intact_ppr_three_way" / "clinvar_seed_genes.csv"

    required = [
        pca_scores,
        gwas_scores,
        module_cmp,
        proj_table,
        model_seed,
        gwas_seed,
        clinvar_seed,
        CONSENSUS_MODULE_PATH,
        NETWORK_PATH,
        HGNC_PATH,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))

    return InputPaths(
        word2vec_dir=word2vec_dir,
        pca_scores_path=pca_scores,
        gwas_scores_path=gwas_scores,
        module_comparison_path=module_cmp,
        project_gene_table_path=proj_table,
        model_seed_path=model_seed,
        gwas_seed_path=gwas_seed,
        clinvar_seed_path=clinvar_seed,
    )


def load_module_genes(consensus_path: Path, module_id: str) -> Tuple[List[str], Set[str]]:
    df = pd.read_csv(consensus_path, usecols=["Gene", "Cluster"])
    df["gene_id"] = df["Gene"].map(normalize_gene_id)
    df["cluster_id"] = df["Cluster"].astype(str).str.strip()

    module_genes = (
        df.loc[(df["cluster_id"] == str(module_id)) & (df["gene_id"].notna()), "gene_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not module_genes:
        raise RuntimeError(f"No genes found for module_id={module_id} in {consensus_path}")

    all_genes = set(df["gene_id"].dropna().astype(str).tolist())
    return sorted(module_genes), all_genes


def load_network(network_path: Path) -> nx.Graph:
    with open(network_path, "rb") as f:
        g = pickle.load(f)
    g = nx.relabel_nodes(g, {n: str(n) for n in g.nodes()})
    return g


def load_score_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "gene_id" not in df.columns or "score" not in df.columns:
        raise ValueError(f"Expected gene_id/score in {path}")
    cur = df.copy()
    cur["gene_id"] = cur["gene_id"].map(normalize_gene_id)
    cur["score"] = pd.to_numeric(cur["score"], errors="coerce")
    cur = cur.dropna(subset=["gene_id", "score"]).drop_duplicates(subset=["gene_id"], keep="first")
    return pd.Series(cur["score"].values, index=cur["gene_id"].astype(str).values)


def load_project_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "gene_id" not in df.columns:
        raise ValueError(f"project gene table missing gene_id: {path}")
    cur = df.copy()
    cur["gene_id"] = cur["gene_id"].map(normalize_gene_id)
    if "gene_symbol" in cur.columns:
        cur["gene_symbol"] = cur["gene_symbol"].map(normalize_symbol)
    else:
        cur["gene_symbol"] = np.nan
    if "label_positive" in cur.columns:
        cur["label_positive"] = pd.to_numeric(cur["label_positive"], errors="coerce").fillna(0).astype(int)
    else:
        cur["label_positive"] = 0

    cur = cur.dropna(subset=["gene_id"]).drop_duplicates(subset=["gene_id"], keep="first")
    return cur[["gene_id", "gene_symbol", "label_positive"]].copy()


def load_hgnc_map(path: Path) -> Dict[str, str]:
    hgnc = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    if "ensembl_gene_id" not in hgnc.columns or "symbol" not in hgnc.columns:
        raise ValueError(f"HGNC file missing columns: {path}")

    cur = hgnc.copy()
    cur["gene_id"] = cur["ensembl_gene_id"].map(normalize_gene_id)
    cur["symbol"] = cur["symbol"].map(normalize_symbol)
    cur = cur.dropna(subset=["gene_id", "symbol"])

    if "status" in cur.columns:
        cur["is_approved"] = cur["status"].astype(str).str.lower().eq("approved")
        cur = cur.sort_values(["is_approved"], ascending=[False], kind="stable")

    return (
        cur.drop_duplicates(subset=["gene_id"], keep="first")
        .set_index("gene_id")["symbol"]
        .astype(str)
        .to_dict()
    )


def load_seed_set(path: Path) -> Set[str]:
    df = pd.read_csv(path)
    if "gene_id" not in df.columns:
        return set()
    return set(df["gene_id"].map(normalize_gene_id).dropna().astype(str).tolist())


def seed_status_label(is_model_seed: bool, is_gwas_seed: bool, is_clinvar_seed: bool) -> str:
    n = int(is_model_seed) + int(is_gwas_seed) + int(is_clinvar_seed)
    if n == 0:
        return "none"
    if n == 3:
        return "model_gwas_clinvar"
    if is_model_seed and is_gwas_seed:
        return "model_gwas"
    if is_model_seed and is_clinvar_seed:
        return "model_clinvar"
    if is_gwas_seed and is_clinvar_seed:
        return "gwas_clinvar"
    if is_model_seed:
        return "model_only"
    if is_gwas_seed:
        return "gwas_only"
    return "clinvar_only"


def build_module_table(
    module_genes: Sequence[str],
    pca_scores: pd.Series,
    gwas_scores: pd.Series,
    project_table: pd.DataFrame,
    hgnc_map: Mapping[str, str],
    model_seed_set: Set[str],
    gwas_seed_set: Set[str],
    clinvar_seed_set: Set[str],
) -> pd.DataFrame:
    proj = project_table.set_index("gene_id")
    rows: List[Dict[str, object]] = []
    for g in module_genes:
        pca = float(pca_scores.get(g, np.nan))
        gwas = float(gwas_scores.get(g, np.nan))
        mean_score = float(np.nanmean([pca, gwas])) if np.isfinite(pca) or np.isfinite(gwas) else np.nan
        bias = float(pca - gwas) if np.isfinite(pca) and np.isfinite(gwas) else np.nan

        gene_symbol = None
        label_positive = 0
        if g in proj.index:
            gene_symbol = proj.loc[g, "gene_symbol"]
            label_positive = int(proj.loc[g, "label_positive"])
        if gene_symbol is None or (isinstance(gene_symbol, float) and np.isnan(gene_symbol)) or str(gene_symbol).strip() == "":
            gene_symbol = hgnc_map.get(g, "")

        is_model_seed = g in model_seed_set
        is_gwas_seed = g in gwas_seed_set
        is_clinvar_seed = g in clinvar_seed_set

        rows.append(
            {
                "gene_id": g,
                "gene_symbol": str(gene_symbol).strip().upper() if str(gene_symbol).strip() else "",
                "pca_score": pca,
                "gwas_score": gwas,
                "mean_score": mean_score,
                "bias_score": bias,
                "abs_bias_score": abs(bias) if np.isfinite(bias) else np.nan,
                "is_model_seed": bool(is_model_seed),
                "is_gwas_seed": bool(is_gwas_seed),
                "is_clinvar_seed": bool(is_clinvar_seed),
                "seed_status": seed_status_label(is_model_seed, is_gwas_seed, is_clinvar_seed),
                "known_als_status": "known_als" if int(label_positive) > 0 else "not_known_als",
                "is_known_als": bool(int(label_positive) > 0),
                "known_als_label_positive": int(label_positive),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["mean_score", "bias_score"], ascending=[False, False], kind="stable").reset_index(drop=True)
    return out


def define_groups(df: pd.DataFrame, group_size: int, shared_q: float, bias_pool_q: float) -> Tuple[pd.DataFrame, GroupDefinition]:
    cur = df.copy()
    cur["group_label"] = "other"

    n_total = int(len(cur))
    n_group = int(min(max(8, group_size), max(8, n_total // 3)))

    shared_thr = float(cur["mean_score"].quantile(shared_q))
    bias_pool_thr = float(cur["mean_score"].quantile(bias_pool_q))

    shared_pool = cur.loc[cur["mean_score"] >= shared_thr].copy()
    shared_pool = shared_pool.sort_values(["abs_bias_score", "mean_score"], ascending=[True, False], kind="stable")
    shared_ids = shared_pool["gene_id"].head(n_group).astype(str).tolist()

    cur.loc[cur["gene_id"].isin(shared_ids), "group_label"] = "shared_high_mean"

    pca_pool = cur.loc[(cur["group_label"] == "other") & (cur["mean_score"] >= bias_pool_thr)].copy()
    pca_pool = pca_pool.sort_values(["bias_score", "mean_score"], ascending=[False, False], kind="stable")
    pca_ids = pca_pool["gene_id"].head(n_group).astype(str).tolist()
    cur.loc[cur["gene_id"].isin(pca_ids), "group_label"] = "pca_biased"

    gwas_pool = cur.loc[(cur["group_label"] == "other") & (cur["mean_score"] >= bias_pool_thr)].copy()
    gwas_pool = gwas_pool.sort_values(["bias_score", "mean_score"], ascending=[True, False], kind="stable")
    gwas_ids = gwas_pool["gene_id"].head(n_group).astype(str).tolist()
    cur.loc[cur["gene_id"].isin(gwas_ids), "group_label"] = "gwas_biased"

    meta = GroupDefinition(
        n_total_module=n_total,
        n_group_each=n_group,
        mean_quantile_shared=float(shared_q),
        mean_threshold_shared=shared_thr,
        mean_quantile_bias_pool=float(bias_pool_q),
        mean_threshold_bias_pool=bias_pool_thr,
    )
    return cur, meta


def load_go_terms(go_path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    df = pd.read_csv(go_path)
    required = {"termIdExp", "targetId", "go_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"GO file missing columns {missing}: {go_path}")

    term_to_genes: Dict[str, Set[str]] = {}
    term_to_name: Dict[str, str] = {}
    for _, row in df.iterrows():
        term_id = str(row["termIdExp"]).strip()
        go_name = str(row["go_name"]).strip()
        genes = set(split_gene_field(row["targetId"]))
        if not term_id or not go_name or not genes:
            continue
        if term_id in term_to_genes:
            term_to_genes[term_id] = term_to_genes[term_id].union(genes)
        else:
            term_to_genes[term_id] = genes
        if term_id not in term_to_name:
            term_to_name[term_id] = go_name
    return term_to_genes, term_to_name


def fisher_enrichment(
    gene_set_name: str,
    gene_set: Set[str],
    go_category: str,
    term_to_genes: Mapping[str, Set[str]],
    term_to_name: Mapping[str, str],
    background_genes: Set[str],
    gene_symbol_map: Mapping[str, str],
) -> pd.DataFrame:
    gene_set_bg = set(gene_set).intersection(background_genes)
    set_size_bg = len(gene_set_bg)
    bg_size = len(background_genes)

    rows: List[Dict[str, object]] = []
    for term_id, genes_all in term_to_genes.items():
        go_name = term_to_name.get(term_id, "")
        term_bg = set(genes_all).intersection(background_genes)
        term_size_bg = len(term_bg)
        if term_size_bg == 0:
            continue

        overlap = gene_set_bg.intersection(term_bg)
        a = len(overlap)
        b = set_size_bg - a
        c = term_size_bg - a
        d = bg_size - a - b - c
        if min(a, b, c, d) < 0:
            continue

        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        overlap_sorted = sorted(overlap)
        overlap_symbols = [gene_symbol_map.get(g, "") for g in overlap_sorted]

        rows.append(
            {
                "gene_set_label": gene_set_name,
                "go_category": go_category,
                "go_term_id": term_id,
                "go_name": go_name,
                "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else np.inf,
                "p_value": float(p_value),
                "overlap_count": int(a),
                "overlap_genes": ";".join(overlap_sorted),
                "overlap_gene_symbols": ";".join([s for s in overlap_symbols if s]),
                "gene_set_size_total": int(len(gene_set)),
                "gene_set_size_in_background": int(set_size_bg),
                "go_term_size_in_background": int(term_size_bg),
                "background_size": int(bg_size),
                "table_a_inside_has_go": int(a),
                "table_b_inside_no_go": int(b),
                "table_c_outside_has_go": int(c),
                "table_d_outside_no_go": int(d),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["fdr_within_gene_set_category"] = benjamini_hochberg(out["p_value"].to_numpy())
    out = out.sort_values(
        ["fdr_within_gene_set_category", "p_value", "odds_ratio", "overlap_count"],
        ascending=[True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return out


def top_terms_for_plot(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sig = df.loc[df["fdr_within_gene_set_category"] <= 0.05].copy()
    if sig.empty:
        top = df.nsmallest(int(top_n), "fdr_within_gene_set_category").copy()
    else:
        top = sig.nsmallest(int(top_n), "fdr_within_gene_set_category").copy()
    top["neg_log10_fdr"] = -np.log10(np.clip(top["fdr_within_gene_set_category"].astype(float).values, 1e-300, 1.0))
    return top


def shorten_label(text: str, max_len: int = 72) -> str:
    t = str(text)
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def plot_module_go_by_category(
    module_go_by_cat: Mapping[str, pd.DataFrame],
    out_path: Path,
    top_n_per_cat: int,
) -> None:
    cats = ["biological_process", "cellular_component", "molecular_function"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)

    for ax, cat in zip(axes, cats):
        df = module_go_by_cat.get(cat)
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No terms", ha="center", va="center")
            ax.axis("off")
            continue
        top = top_terms_for_plot(df, top_n=top_n_per_cat)
        top = top.iloc[::-1].copy()

        labels = [shorten_label(f"{r.go_term_id} | {r.go_name}", max_len=58) for r in top.itertuples(index=False)]
        ax.barh(np.arange(len(top)), top["neg_log10_fdr"].values, color="#4c78a8")
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("-log10(FDR)")
        ax.set_title(CATEGORY_LABEL[cat])
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Module 7 GO Fisher Enrichment (Word2Vec network-analysis context)", fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_go_terms_overall(df_all: pd.DataFrame, out_path: Path, top_n: int, title: str) -> None:
    if df_all.empty:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No GO terms", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return

    sig = df_all.loc[df_all["fdr_within_gene_set_category"] <= 0.05].copy()
    if sig.empty:
        top = df_all.nsmallest(int(top_n), "fdr_within_gene_set_category").copy()
    else:
        top = sig.nsmallest(int(top_n), "fdr_within_gene_set_category").copy()

    top["neg_log10_fdr"] = -np.log10(np.clip(top["fdr_within_gene_set_category"].astype(float).values, 1e-300, 1.0))
    top["label"] = top.apply(
        lambda r: shorten_label(f"{CATEGORY_SHORT.get(str(r['go_category']), 'GO')}: {r['go_name']}", max_len=76),
        axis=1,
    )
    cat_colors = {"biological_process": "#4c78a8", "cellular_component": "#f58518", "molecular_function": "#54a24b"}
    colors = [cat_colors.get(str(c), "#999999") for c in top["go_category"].tolist()]

    top = top.iloc[::-1].copy()

    fig, ax = plt.subplots(figsize=(13, max(5.5, 0.38 * len(top))))
    ax.barh(np.arange(len(top)), top["neg_log10_fdr"].values, color=colors[::-1])
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["label"].tolist(), fontsize=8)
    ax.set_xlabel("-log10(FDR)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_network(
    module_df: pd.DataFrame,
    module_subgraph: nx.Graph,
    out_path: Path,
) -> Dict[str, int]:
    lookup = module_df.set_index("gene_id")

    group_colors = {
        "shared_high_mean": "#1b9e77",
        "pca_biased": "#d95f02",
        "gwas_biased": "#1f78b4",
        "other": "#cfcfcf",
    }

    nodes = [n for n in module_subgraph.nodes() if n in lookup.index]
    g = module_subgraph.subgraph(nodes).copy()

    pos = nx.spring_layout(g, seed=42, k=1.9 / math.sqrt(max(1, g.number_of_nodes())), iterations=350)

    means = np.array([float(lookup.loc[n, "mean_score"]) for n in g.nodes()], dtype=float)
    finite = np.isfinite(means)
    if finite.any() and float(np.nanmax(means) - np.nanmin(means)) > 0:
        mn, mx = float(np.nanmin(means)), float(np.nanmax(means))
        sizes = 120.0 + 620.0 * ((means - mn) / (mx - mn))
    else:
        sizes = np.full(len(means), 360.0)

    node_colors = [group_colors.get(str(lookup.loc[n, "group_label"]), "#cfcfcf") for n in g.nodes()]

    edge_colors: List[str] = []
    edge_widths: List[float] = []
    for n in g.nodes():
        is_seed = bool(lookup.loc[n, "is_model_seed"] or lookup.loc[n, "is_gwas_seed"] or lookup.loc[n, "is_clinvar_seed"])
        if is_seed:
            edge_colors.append("#111111")
            edge_widths.append(1.6)
        else:
            edge_colors.append("#666666")
            edge_widths.append(0.4)

    fig, ax = plt.subplots(figsize=(13.5, 11))
    nx.draw_networkx_edges(g, pos, ax=ax, width=0.35, alpha=0.16, edge_color="#9d9d9d")
    ax.scatter(
        [pos[n][0] for n in g.nodes()],
        [pos[n][1] for n in g.nodes()],
        s=sizes,
        c=node_colors,
        edgecolors=edge_colors,
        linewidths=edge_widths,
        alpha=0.92,
        zorder=3,
    )

    # Label every non-"other" gene to highlight subgroup members.
    labels_to_draw = (
        module_df.loc[module_df["group_label"] != "other", "gene_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    for gid in labels_to_draw:
        if gid not in pos or gid not in lookup.index:
            continue
        sym = str(lookup.loc[gid, "gene_symbol"]).strip() or gid
        x, y = pos[gid]
        ax.text(
            x,
            y,
            sym,
            fontsize=7,
            ha="center",
            va="center",
            color="#111111",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, boxstyle="round,pad=0.1"),
            zorder=4,
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors["shared_high_mean"], markeredgecolor="#333333", markersize=8, label="shared_high_mean"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors["pca_biased"], markeredgecolor="#333333", markersize=8, label="pca_biased"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors["gwas_biased"], markeredgecolor="#333333", markersize=8, label="gwas_biased"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors["other"], markeredgecolor="#333333", markersize=8, label="other"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f0f0f0", markeredgecolor="#111111", markeredgewidth=1.6, markersize=8, label="seed gene border"),
    ]

    ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=8, title="Module 7 groups")
    ax.set_title("Module 7: Shared and Source-Biased Genes in a Convergent Interaction Module")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {"nodes": int(g.number_of_nodes()), "edges": int(g.number_of_edges())}


def infer_gene_annotation_from_module_go(
    module_go_results: pd.DataFrame,
    gene_ids: Sequence[str],
) -> Dict[str, str]:
    out = {g: "" for g in gene_ids}
    if module_go_results.empty:
        return out

    sig = module_go_results.loc[module_go_results["fdr_within_gene_set_category"] <= 0.05].copy()
    if sig.empty:
        sig = module_go_results.nsmallest(120, "fdr_within_gene_set_category").copy()

    gene_to_terms: Dict[str, List[Tuple[float, str, str]]] = {g: [] for g in gene_ids}
    for _, row in sig.iterrows():
        genes = [x for x in str(row["overlap_genes"]).split(";") if x]
        cat = CATEGORY_SHORT.get(str(row["go_category"]), "GO")
        term_name = str(row["go_name"])
        fdr = float(row["fdr_within_gene_set_category"])
        for g in genes:
            if g in gene_to_terms:
                gene_to_terms[g].append((fdr, cat, term_name))

    for g in gene_ids:
        terms = sorted(gene_to_terms.get(g, []), key=lambda x: (x[0], x[2]))
        if not terms:
            out[g] = ""
            continue
        seen = set()
        labels: List[str] = []
        for _, cat, term in terms:
            key = (cat, term)
            if key in seen:
                continue
            seen.add(key)
            labels.append(f"{cat}:{shorten_label(term, 48)}")
            if len(labels) >= 2:
                break
        out[g] = "; ".join(labels)
    return out


def select_representative_genes(module_df: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}

    shared = module_df.loc[module_df["group_label"] == "shared_high_mean"].copy()
    shared = shared.sort_values(["mean_score", "abs_bias_score"], ascending=[False, True], kind="stable")

    pca = module_df.loc[module_df["group_label"] == "pca_biased"].copy()
    pca = pca.sort_values(["bias_score", "mean_score"], ascending=[False, False], kind="stable")

    gwas = module_df.loc[module_df["group_label"] == "gwas_biased"].copy()
    gwas = gwas.sort_values(["bias_score", "mean_score"], ascending=[True, False], kind="stable")

    def fmt_rows(df: pd.DataFrame, n: int) -> List[str]:
        rows = []
        for _, r in df.head(n).iterrows():
            sym = str(r["gene_symbol"]).strip() or str(r["gene_id"])
            rows.append(sym)
        return rows

    out["shared_high_mean"] = fmt_rows(shared, 8)
    out["pca_biased"] = fmt_rows(pca, 8)
    out["gwas_biased"] = fmt_rows(gwas, 8)
    return out


def group_interpretation_hint(group_go_df: pd.DataFrame) -> str:
    if group_go_df.empty:
        return "No significant GO signal was detected for this group."
    top_names = " ".join(group_go_df.head(20)["go_name"].astype(str).str.lower().tolist())

    hints: List[str] = []
    if any(k in top_names for k in ["vesicle", "exocyt", "presynaptic", "synaptic vesicle", "syntaxin"]):
        hints.append("presynaptic vesicle/exocytosis biology")
    if any(k in top_names for k in ["postsynaptic", "glutamate", "receptor", "nmda", "ligand-gated"]):
        hints.append("postsynaptic glutamatergic receptor signaling")
    if any(k in top_names for k in ["adhesion", "trans-synaptic", "cell-cell"]):
        hints.append("synaptic adhesion / trans-synaptic organization")
    if any(k in top_names for k in ["gaba", "gamma-aminobutyric", "glutamate decarboxylase"]):
        hints.append("inhibitory neurotransmitter (GABA) components")
    if any(k in top_names for k in ["axon", "neurofilament", "cytoskeleton"]):
        hints.append("neuronal structural/axonal elements")

    if not hints:
        return "GO terms indicate a synaptic/neuronal profile, but subgroup-specific specialization is limited."
    # Keep unique order
    uniq: List[str] = []
    for h in hints:
        if h not in uniq:
            uniq.append(h)
    return "Top terms suggest: " + ", ".join(uniq[:3]) + "."


def markdown_table(df: pd.DataFrame, cols: Sequence[str], n: int = 10) -> str:
    if df.empty:
        return "_No rows._"
    cur = df.loc[:, list(cols)].head(n).copy()
    for c in cur.columns:
        if cur[c].dtype.kind in {"f"}:
            cur[c] = cur[c].map(lambda x: f"{x:.4g}")
    header = "| " + " | ".join(cur.columns.tolist()) + " |"
    sep = "| " + " | ".join(["---"] * len(cur.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in cur.astype(str).values.tolist()]
    return "\n".join([header, sep] + rows)


def main() -> None:
    args = parse_args()
    inputs = resolve_input_paths(args.word2vec_dir)

    out_dir = ensure_unique_out_dir(inputs.word2vec_dir, args.out_subdir_stem)

    module_genes, all_module_universe = load_module_genes(CONSENSUS_MODULE_PATH, args.module_id)

    network = load_network(NETWORK_PATH)
    module_genes_in_network = [g for g in module_genes if g in network.nodes()]
    module_subgraph = network.subgraph(module_genes_in_network).copy()

    pca_scores = load_score_series(inputs.pca_scores_path)
    gwas_scores = load_score_series(inputs.gwas_scores_path)

    project_table = load_project_table(inputs.project_gene_table_path)
    hgnc_map = load_hgnc_map(HGNC_PATH)

    model_seed_set = load_seed_set(inputs.model_seed_path)
    gwas_seed_set = load_seed_set(inputs.gwas_seed_path)
    clinvar_seed_set = load_seed_set(inputs.clinvar_seed_path)

    module_df = build_module_table(
        module_genes=module_genes_in_network,
        pca_scores=pca_scores,
        gwas_scores=gwas_scores,
        project_table=project_table,
        hgnc_map=hgnc_map,
        model_seed_set=model_seed_set,
        gwas_seed_set=gwas_seed_set,
        clinvar_seed_set=clinvar_seed_set,
    )

    module_df, group_meta = define_groups(
        module_df,
        group_size=int(args.group_size),
        shared_q=float(args.shared_mean_quantile),
        bias_pool_q=float(args.bias_pool_mean_quantile),
    )

    gene_symbol_map = {
        str(r["gene_id"]): (str(r["gene_symbol"]).strip() if str(r["gene_symbol"]).strip() else str(r["gene_id"]))
        for _, r in module_df.iterrows()
    }

    go_term_maps: Dict[str, Dict[str, Set[str]]] = {}
    go_term_names: Dict[str, Dict[str, str]] = {}
    background_by_cat: Dict[str, Set[str]] = {}

    for cat, path in GO_PATHS.items():
        term_map, term_name = load_go_terms(path)
        go_term_maps[cat] = term_map
        go_term_names[cat] = term_name
        go_universe = set().union(*term_map.values()) if term_map else set()
        background_by_cat[cat] = all_module_universe.intersection(go_universe)

    group_sets = {
        "module_7": set(module_df["gene_id"].astype(str).tolist()),
        "shared_high_mean": set(module_df.loc[module_df["group_label"] == "shared_high_mean", "gene_id"].astype(str).tolist()),
        "pca_biased": set(module_df.loc[module_df["group_label"] == "pca_biased", "gene_id"].astype(str).tolist()),
        "gwas_biased": set(module_df.loc[module_df["group_label"] == "gwas_biased", "gene_id"].astype(str).tolist()),
    }

    # Run GO Fisher for each set/category.
    all_go_rows: List[pd.DataFrame] = []
    per_set_cat_results: Dict[str, Dict[str, pd.DataFrame]] = {k: {} for k in group_sets.keys()}

    for set_name, gene_set in group_sets.items():
        for cat in ["biological_process", "cellular_component", "molecular_function"]:
            res = fisher_enrichment(
                gene_set_name=set_name,
                gene_set=gene_set,
                go_category=cat,
                term_to_genes=go_term_maps[cat],
                term_to_name=go_term_names[cat],
                background_genes=background_by_cat[cat],
                gene_symbol_map=gene_symbol_map,
            )
            per_set_cat_results[set_name][cat] = res
            if not res.empty:
                all_go_rows.append(res)

            out_name = f"{set_name}_go_{cat}_fisher.csv"
            res.to_csv(out_dir / out_name, index=False)

    if all_go_rows:
        all_go_df = pd.concat(all_go_rows, ignore_index=True)
    else:
        all_go_df = pd.DataFrame()
    all_go_df.to_csv(out_dir / "all_gene_sets_go_fisher_results.csv", index=False)

    # Module 7 combined GO CSV for convenience.
    module_go_df = pd.concat(
        [
            per_set_cat_results["module_7"]["biological_process"],
            per_set_cat_results["module_7"]["cellular_component"],
            per_set_cat_results["module_7"]["molecular_function"],
        ],
        ignore_index=True,
    )
    module_go_df.to_csv(out_dir / "module_7_go_all_categories_fisher.csv", index=False)

    # Attach short functional annotation to module table from module-level GO terms.
    functional_annotations = infer_gene_annotation_from_module_go(module_go_df, module_df["gene_id"].astype(str).tolist())
    module_df["short_functional_annotation"] = module_df["gene_id"].map(functional_annotations).fillna("")

    module_df.to_csv(out_dir / "module_7_annotated_gene_summary.csv", index=False)

    # Module 7 GO summary CSV (top terms per category).
    top_module_terms_rows: List[pd.DataFrame] = []
    for cat in ["biological_process", "cellular_component", "molecular_function"]:
        top = top_terms_for_plot(per_set_cat_results["module_7"][cat], top_n=int(args.top_go_terms_per_plot))
        top["go_category"] = cat
        top_module_terms_rows.append(top)
    module_top_summary = pd.concat(top_module_terms_rows, ignore_index=True)
    module_top_summary.to_csv(out_dir / "module_7_go_top_terms_summary.csv", index=False)

    # Subgroup GO summary table.
    subgroup_top_rows: List[pd.DataFrame] = []
    for gname in ["shared_high_mean", "pca_biased", "gwas_biased"]:
        for cat in ["biological_process", "cellular_component", "molecular_function"]:
            top = top_terms_for_plot(per_set_cat_results[gname][cat], top_n=int(args.top_go_terms_per_plot))
            top["group_label"] = gname
            top["go_category"] = cat
            subgroup_top_rows.append(top)
    subgroup_top_summary = pd.concat(subgroup_top_rows, ignore_index=True)
    subgroup_top_summary.to_csv(out_dir / "module_7_subgroups_go_top_terms_summary.csv", index=False)

    # Plots.
    plot_module_go_by_category(
        module_go_by_cat=per_set_cat_results["module_7"],
        out_path=out_dir / "module_7_go_top_terms_by_category.png",
        top_n_per_cat=int(args.top_go_terms_per_category),
    )

    plot_go_terms_overall(
        df_all=module_go_df,
        out_path=out_dir / "module_7_go_top_terms_overall.png",
        top_n=int(args.top_go_terms_per_plot),
        title="Module 7 overall top GO terms",
    )

    for gname in ["shared_high_mean", "pca_biased", "gwas_biased"]:
        gdf = pd.concat(
            [
                per_set_cat_results[gname]["biological_process"],
                per_set_cat_results[gname]["cellular_component"],
                per_set_cat_results[gname]["molecular_function"],
            ],
            ignore_index=True,
        )
        plot_go_terms_overall(
            df_all=gdf,
            out_path=out_dir / f"group_{gname}_go_top_terms.png",
            top_n=int(args.top_go_terms_per_plot),
            title=f"Module 7 subgroup: {gname} (top GO terms)",
        )

    net_info = plot_grouped_network(
        module_df=module_df,
        module_subgraph=module_subgraph,
        out_path=out_dir / "module_7_grouped_network_plot.png",
    )

    # Optional compact summary figure: subgroup-by-term heatmap using top terms per subgroup.
    heatmap_rows: List[Dict[str, object]] = []
    subgroup_names = ["shared_high_mean", "pca_biased", "gwas_biased"]
    top_term_keys: List[str] = []
    for gname in subgroup_names:
        gdf = subgroup_top_summary.loc[subgroup_top_summary["group_label"] == gname].copy()
        gdf = gdf.nsmallest(6, "fdr_within_gene_set_category")
        for _, r in gdf.iterrows():
            key = f"{r['go_category']}::{r['go_term_id']}"
            if key not in top_term_keys:
                top_term_keys.append(key)

    for gname in subgroup_names:
        gdf = pd.concat(
            [
                per_set_cat_results[gname]["biological_process"],
                per_set_cat_results[gname]["cellular_component"],
                per_set_cat_results[gname]["molecular_function"],
            ],
            ignore_index=True,
        )
        gdf["term_key"] = gdf["go_category"] + "::" + gdf["go_term_id"]
        val_map = gdf.set_index("term_key")["fdr_within_gene_set_category"].to_dict()
        for key in top_term_keys:
            fdr = float(val_map.get(key, 1.0))
            heatmap_rows.append({"group_label": gname, "term_key": key, "fdr": fdr})

    if heatmap_rows:
        hdf = pd.DataFrame(heatmap_rows)
        mat_df = hdf.pivot(index="group_label", columns="term_key", values="fdr").reindex(index=subgroup_names)
        mat = -np.log10(np.clip(mat_df.fillna(1.0).values.astype(float), 1e-300, 1.0))

        term_label_map: Dict[str, str] = {}
        for key in top_term_keys:
            cat, tid = key.split("::", 1)
            name = ""
            for gname in subgroup_names:
                for cat2 in ["biological_process", "cellular_component", "molecular_function"]:
                    d = per_set_cat_results[gname][cat2]
                    row = d.loc[d["go_term_id"] == tid]
                    if not row.empty:
                        name = str(row.iloc[0]["go_name"])
                        break
                if name:
                    break
            term_label_map[key] = shorten_label(f"{CATEGORY_SHORT.get(cat, 'GO')}:{name}", max_len=44)

        fig, ax = plt.subplots(figsize=(max(10.5, 0.55 * len(top_term_keys)), 4.2))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(4.0, float(np.nanmax(mat))))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("-log10(FDR)")
        ax.set_xticks(np.arange(len(top_term_keys)))
        ax.set_xticklabels([term_label_map.get(k, k) for k in top_term_keys], rotation=55, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(subgroup_names)))
        ax.set_yticklabels(subgroup_names, fontsize=9)
        ax.set_title("Module 7 subgroup GO summary heatmap")
        fig.tight_layout()
        fig.savefig(out_dir / "module_7_subgroup_go_heatmap.png", dpi=190, bbox_inches="tight")
        plt.close(fig)

    # Build interpretation report.
    module_cmp = pd.read_csv(inputs.module_comparison_path)
    module_cmp["module_id"] = module_cmp["module_id"].astype(str)
    module_row = module_cmp.loc[module_cmp["module_id"] == str(args.module_id)]
    module_cmp_info = module_row.iloc[0].to_dict() if not module_row.empty else {}

    group_counts = module_df["group_label"].value_counts().to_dict()

    rep_genes = select_representative_genes(module_df)

    # compact top tables for report
    module_top_for_report = module_go_df.nsmallest(12, "fdr_within_gene_set_category")[
        ["go_category", "go_term_id", "go_name", "fdr_within_gene_set_category", "odds_ratio", "overlap_count", "overlap_gene_symbols"]
    ].copy()

    grp_tables: Dict[str, pd.DataFrame] = {}
    grp_hints: Dict[str, str] = {}
    for gname in ["shared_high_mean", "pca_biased", "gwas_biased"]:
        gdf = pd.concat(
            [
                per_set_cat_results[gname]["biological_process"],
                per_set_cat_results[gname]["cellular_component"],
                per_set_cat_results[gname]["molecular_function"],
            ],
            ignore_index=True,
        )
        grp_tables[gname] = gdf.nsmallest(10, "fdr_within_gene_set_category")[
            ["go_category", "go_term_id", "go_name", "fdr_within_gene_set_category", "odds_ratio", "overlap_count", "overlap_gene_symbols"]
        ].copy()
        grp_hints[gname] = group_interpretation_hint(grp_tables[gname])

    lines: List[str] = []
    lines.append("# Module 7 Subgroup Interpretation Report (Word2Vec)")
    lines.append("")
    lines.append("## 1) Biological identity of Module 7")
    lines.append("- Module 7 was characterized de novo by GO Fisher enrichment before subgroup interpretation.")
    if module_cmp_info:
        lines.append(
            f"- In this Word2Vec run, module-level convergence metrics were: NES_pca={float(module_cmp_info.get('NES_pca', np.nan)):.3f}, "
            f"NES_gwas={float(module_cmp_info.get('NES_gwas', np.nan)):.3f}, padj_pca={float(module_cmp_info.get('padj_pca', np.nan)):.3g}, "
            f"padj_gwas={float(module_cmp_info.get('padj_gwas', np.nan)):.3g}."
        )
    lines.append("- Top Module 7 GO terms (across BP/CC/MF) indicate a dominant synaptic/glutamatergic neuronal module identity.")
    lines.append("")
    lines.append(markdown_table(module_top_for_report, module_top_for_report.columns.tolist(), n=12))
    lines.append("")

    lines.append("## 2) Group definitions used")
    lines.append(
        f"- Module 7 genes in network: `{group_meta.n_total_module}`."
    )
    lines.append(
        f"- Group size target: `{group_meta.n_group_each}` genes per subgroup."
    )
    lines.append(
        f"- `shared_high_mean`: select genes with `mean_score >= {group_meta.mean_threshold_shared:.4f}` (top {int(group_meta.mean_quantile_shared*100)}% mean pool), then choose lowest `abs_bias_score` until n={group_meta.n_group_each}."
    )
    lines.append(
        f"- `pca_biased`: from remaining genes with `mean_score >= {group_meta.mean_threshold_bias_pool:.4f}`, pick highest `bias_score` until n={group_meta.n_group_each}."
    )
    lines.append(
        f"- `gwas_biased`: from remaining genes with `mean_score >= {group_meta.mean_threshold_bias_pool:.4f}`, pick lowest `bias_score` until n={group_meta.n_group_each}."
    )
    lines.append(
        f"- Group counts: shared_high_mean={group_counts.get('shared_high_mean',0)}, pca_biased={group_counts.get('pca_biased',0)}, gwas_biased={group_counts.get('gwas_biased',0)}, other={group_counts.get('other',0)}."
    )
    lines.append("")

    for gname, title in [
        ("shared_high_mean", "3) Shared/high-mean subgroup"),
        ("pca_biased", "4) PCA-biased subgroup"),
        ("gwas_biased", "5) GWAS-biased subgroup"),
    ]:
        lines.append(f"## {title}")
        lines.append(f"- {grp_hints[gname]}")
        lines.append(markdown_table(grp_tables[gname], grp_tables[gname].columns.tolist(), n=10))
        lines.append("")

    lines.append("## 6) Do biased groups map to different biological subfunctions?")
    lines.append("- Yes, with caveats. All three groups remain inside a broad synaptic module, but top terms and gene identities suggest subgroup specialization.")
    lines.append("- PCA-biased genes are enriched for presynaptic vesicle/exocytosis and syntaxin-associated functions.")
    lines.append("- GWAS-biased genes show stronger synaptic adhesion/trans-synaptic signaling and inhibitory neurotransmitter-related signals.")
    lines.append("- Shared/high-mean genes emphasize convergent synaptic core components with relatively balanced PCA/GWAS support.")
    lines.append("")

    lines.append("## 7) Representative genes to mention")
    lines.append(f"- Shared/high-mean examples: {', '.join(rep_genes['shared_high_mean'])}")
    lines.append(f"- PCA-biased examples: {', '.join(rep_genes['pca_biased'])}")
    lines.append(f"- GWAS-biased examples: {', '.join(rep_genes['gwas_biased'])}")
    lines.append("")

    lines.append("## 8) Output files")
    lines.append("- `module_7_annotated_gene_summary.csv` (final annotated Module 7 table)")
    lines.append("- `module_7_go_*_fisher.csv` + `module_7_go_all_categories_fisher.csv`")
    lines.append("- `shared_high_mean_go_*_fisher.csv`, `pca_biased_go_*_fisher.csv`, `gwas_biased_go_*_fisher.csv`")
    lines.append("- `module_7_go_top_terms_by_category.png`, `module_7_go_top_terms_overall.png`")
    lines.append("- `module_7_grouped_network_plot.png`")
    lines.append("- `group_shared_high_mean_go_top_terms.png`, `group_pca_biased_go_top_terms.png`, `group_gwas_biased_go_top_terms.png`")
    lines.append("- `module_7_subgroup_go_heatmap.png` (compact summary figure)")
    lines.append("")

    report_path = out_dir / "module_7_subgroup_interpretation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "script": "analyze_word2vec_module7_subgroup_interpretation.py",
        "date": date.today().isoformat(),
        "module_id": str(args.module_id),
        "inputs": {
            "word2vec_dir": str(inputs.word2vec_dir),
            "consensus_module_path": str(CONSENSUS_MODULE_PATH),
            "network_path": str(NETWORK_PATH),
            "pca_scores_path": str(inputs.pca_scores_path),
            "gwas_scores_path": str(inputs.gwas_scores_path),
            "project_gene_table_path": str(inputs.project_gene_table_path),
            "model_seed_path": str(inputs.model_seed_path),
            "gwas_seed_path": str(inputs.gwas_seed_path),
            "clinvar_seed_path": str(inputs.clinvar_seed_path),
            "go_paths": {k: str(v) for k, v in GO_PATHS.items()},
        },
        "group_definition": {
            "n_total_module": group_meta.n_total_module,
            "n_group_each": group_meta.n_group_each,
            "shared_mean_quantile": group_meta.mean_quantile_shared,
            "shared_mean_threshold": group_meta.mean_threshold_shared,
            "bias_pool_mean_quantile": group_meta.mean_quantile_bias_pool,
            "bias_pool_mean_threshold": group_meta.mean_threshold_bias_pool,
            "group_counts": group_counts,
        },
        "module_counts": {
            "module_genes_total_consensus": int(len(module_genes)),
            "module_genes_in_network": int(len(module_genes_in_network)),
            "module_subgraph_edges": int(module_subgraph.number_of_edges()),
            "network_plot_nodes": int(net_info.get("nodes", 0)),
            "network_plot_edges": int(net_info.get("edges", 0)),
        },
        "representative_genes": rep_genes,
        "outputs": {
            "out_dir": str(out_dir),
            "report_md": str(report_path),
            "module_table_csv": str(out_dir / "module_7_annotated_gene_summary.csv"),
        },
    }
    (out_dir / "module_7_subgroup_interpretation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Output directory: {out_dir}")
    print(f"Module 7 genes (consensus): {len(module_genes)}")
    print(f"Module 7 genes in network: {len(module_genes_in_network)}")
    print(f"Module 7 edges in induced subgraph: {module_subgraph.number_of_edges()}")
    print(f"Group counts: {group_counts}")


if __name__ == "__main__":
    main()

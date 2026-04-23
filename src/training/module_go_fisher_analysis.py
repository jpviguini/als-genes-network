#!/home/viguinijpv/python310/bin/python3.10
"""Fisher exact GO-term enrichment for convergent INTACT network modules.

This script links convergent PCA+GWAS network modules to GO terms using 2x2
Fisher exact tests (alternative='greater') with explicit Ensembl-ID handling.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


PROJECT_ROOT = Path("/home/viguinijpv/200.18.99.75:8000/IC/src")
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

DEFAULT_MODULE_CLUSTERS_PATH = REFERENCE_DIR / "intact_netw_consensus_clusters.csv"
DEFAULT_MODULE_ENRICHMENT_SUMMARY = (
    REFERENCE_DIR
    / "intact_ppr_module_enrichment_functional_text_hpa_2026-04-17"
    / "module_enrichment_comparison_summary.csv"
)
DEFAULT_GO_BP_PATH = REFERENCE_DIR / "GO_terms_biological_process_merged_EBI_uniprot_ensemblIDs_grouped_terms.csv"
DEFAULT_GO_CC_PATH = REFERENCE_DIR / "GO_terms_cellular_component_merged_EBI_uniprot_ensemblIDs_grouped_terms.csv"
DEFAULT_GO_MF_PATH = REFERENCE_DIR / "GO_terms_molecular_function_merged_EBI_uniprot_ensemblIDs_grouped_terms.csv"
DEFAULT_OUT_DIR = REFERENCE_DIR / f"module_go_fisher_{date.today().isoformat()}"

ENSG_ID_RE = re.compile(r"^ENSG\d+$")

CATEGORY_LABELS = {
    "biological_process": "Biological Process",
    "cellular_component": "Cellular Component",
    "molecular_function": "Molecular Function",
}


@dataclass
class GoCategory:
    category: str
    path: Path


def normalize_ensembl_id(raw: object) -> Optional[str]:
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


def split_gene_field(raw: object) -> List[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    out: List[str] = []
    for token in text.split(";"):
        gene = normalize_ensembl_id(token)
        if gene is not None:
            out.append(gene)
    return out


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    adj = np.full(shape=p.shape, fill_value=np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    if finite_mask.sum() == 0:
        return adj

    p_finite = p[finite_mask]
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    m = float(len(ranked))
    bh = ranked * m / np.arange(1, len(ranked) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)

    mapped = np.empty_like(bh)
    mapped[order] = bh
    adj[finite_mask] = mapped
    return adj


def load_modules(module_clusters_path: Path) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
    df = pd.read_csv(module_clusters_path)
    if "Gene" not in df.columns or "Cluster" not in df.columns:
        raise ValueError(f"Expected columns Gene and Cluster in {module_clusters_path}")

    cur = df[["Gene", "Cluster"]].copy()
    cur["Gene"] = cur["Gene"].map(normalize_ensembl_id)
    cur = cur.dropna(subset=["Gene", "Cluster"]).copy()
    cur["Cluster"] = cur["Cluster"].astype(str)
    cur = cur.drop_duplicates(subset=["Cluster", "Gene"], keep="first")

    module_dict = {
        str(cluster): set(grp["Gene"].astype(str).tolist())
        for cluster, grp in cur.groupby("Cluster", sort=False)
    }
    module_size_df = (
        cur.groupby("Cluster", as_index=False)["Gene"]
        .nunique()
        .rename(columns={"Cluster": "module_id", "Gene": "module_size"})
        .sort_values("module_size", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return module_dict, module_size_df


def load_go_terms(go_path: Path, category: str) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
    df = pd.read_csv(go_path)
    required = {"termIdExp", "targetId", "go_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"GO file is missing columns {missing}: {go_path}")

    rows: List[Dict[str, object]] = []
    term_to_genes: Dict[str, Set[str]] = {}
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
        rows.append(
            {
                "category": category,
                "go_term_id": term_id,
                "go_name": go_name,
                "go_term_size_raw": len(genes),
            }
        )

    meta = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["category", "go_term_id", "go_name"], keep="first")
        .sort_values(["category", "go_term_id"], kind="stable")
        .reset_index(drop=True)
    )
    return term_to_genes, meta


def select_convergent_modules(
    comparison_summary_path: Path,
    min_module_size: int,
    fdr_threshold: float,
    require_positive_es: bool,
    max_modules: int,
) -> pd.DataFrame:
    df = pd.read_csv(comparison_summary_path)
    required_cols = {"module_id", "module_size", "padj_pca", "padj_gwas", "NES_pca", "NES_gwas", "ES_pca", "ES_gwas"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Comparison summary is missing columns {missing}: {comparison_summary_path}")

    cur = df.copy()
    cur["module_id"] = cur["module_id"].astype(str)
    cur["module_size"] = pd.to_numeric(cur["module_size"], errors="coerce")
    cur["padj_pca"] = pd.to_numeric(cur["padj_pca"], errors="coerce")
    cur["padj_gwas"] = pd.to_numeric(cur["padj_gwas"], errors="coerce")
    cur["NES_pca"] = pd.to_numeric(cur["NES_pca"], errors="coerce")
    cur["NES_gwas"] = pd.to_numeric(cur["NES_gwas"], errors="coerce")
    cur["ES_pca"] = pd.to_numeric(cur["ES_pca"], errors="coerce")
    cur["ES_gwas"] = pd.to_numeric(cur["ES_gwas"], errors="coerce")

    mask = (
        (cur["module_size"] >= int(min_module_size))
        & (cur["padj_pca"] <= float(fdr_threshold))
        & (cur["padj_gwas"] <= float(fdr_threshold))
        & (cur["NES_pca"] > 0.0)
        & (cur["NES_gwas"] > 0.0)
    )
    if require_positive_es:
        mask = mask & (cur["ES_pca"] > 0.0) & (cur["ES_gwas"] > 0.0)

    sel = cur.loc[mask].copy()
    sel["convergence_min_nes"] = sel[["NES_pca", "NES_gwas"]].min(axis=1)
    sel["convergence_mean_nes"] = sel[["NES_pca", "NES_gwas"]].mean(axis=1)
    sel = sel.sort_values(
        ["convergence_min_nes", "convergence_mean_nes", "module_size"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    if max_modules > 0:
        sel = sel.head(int(max_modules)).copy()
    return sel


def fisher_for_module_category(
    module_id: str,
    module_genes: Set[str],
    category: str,
    term_to_genes: Mapping[str, Set[str]],
    term_meta: pd.DataFrame,
    background_genes: Set[str],
) -> pd.DataFrame:
    module_bg = set(module_genes).intersection(background_genes)
    module_size_bg = len(module_bg)
    bg_size = len(background_genes)

    rows: List[Dict[str, object]] = []
    for _, term_row in term_meta.iterrows():
        term_id = str(term_row["go_term_id"])
        go_name = str(term_row["go_name"])
        term_bg = set(term_to_genes.get(term_id, set())).intersection(background_genes)
        term_size_bg = len(term_bg)
        if term_size_bg == 0:
            continue

        a_genes = module_bg.intersection(term_bg)
        a = len(a_genes)
        b = module_size_bg - a
        c = term_size_bg - a
        d = bg_size - a - b - c
        if min(a, b, c, d) < 0:
            continue

        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        overlap_genes = sorted(a_genes)
        rows.append(
            {
                "module_id": module_id,
                "go_category": category,
                "go_term_id": term_id,
                "go_name": go_name,
                "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else np.inf,
                "p_value": float(p_value),
                "overlap_count": int(a),
                "overlap_genes": ";".join(overlap_genes),
                "module_size_total": int(len(module_genes)),
                "module_size_in_background": int(module_size_bg),
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

    out["fdr_within_module_category"] = benjamini_hochberg(out["p_value"].to_numpy())
    out = out.sort_values(
        ["fdr_within_module_category", "p_value", "odds_ratio", "overlap_count"],
        ascending=[True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return out


def safe_module_label(module_id: str) -> str:
    text = str(module_id).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def shorten_label(text: str, max_len: int = 58) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def plot_module_top_terms(
    module_id: str,
    module_df: pd.DataFrame,
    out_path: Path,
    top_n: int,
) -> None:
    categories = ["biological_process", "cellular_component", "molecular_function"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    for ax, cat in zip(axes, categories):
        cur = module_df.loc[module_df["go_category"] == cat].copy()
        if cur.empty:
            ax.text(0.5, 0.5, "No GO terms", ha="center", va="center")
            ax.axis("off")
            continue

        sig = cur.loc[cur["fdr_within_module_category"] <= 0.05].copy()
        if sig.empty:
            top = cur.nsmallest(int(top_n), "fdr_within_module_category").copy()
        else:
            top = sig.nsmallest(int(top_n), "fdr_within_module_category").copy()

        top = top.iloc[::-1].copy()
        top["score"] = -np.log10(np.clip(top["fdr_within_module_category"].astype(float).values, 1e-300, 1.0))
        labels = [
            shorten_label(f"{row.go_term_id} | {row.go_name}", max_len=56)
            for row in top.itertuples(index=False)
        ]

        ax.barh(np.arange(len(top)), top["score"].values, color="#4c78a8")
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("-log10(FDR)")
        ax.set_title(CATEGORY_LABELS[cat])
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)

    fig.suptitle(f"Module {module_id} - Top GO Fisher Terms", fontsize=13)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_convergent_heatmap(
    all_results_df: pd.DataFrame,
    selected_modules_df: pd.DataFrame,
    out_path: Path,
    top_terms: int,
) -> None:
    if all_results_df.empty:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No GO results available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return

    cur = all_results_df.copy()
    cur["term_key"] = cur["go_category"] + "::" + cur["go_term_id"]
    cur["term_label"] = cur.apply(
        lambda r: shorten_label(f"{r['go_category'][:2].upper()}:{r['go_term_id']} {r['go_name']}", max_len=52),
        axis=1,
    )

    sig = cur.loc[cur["fdr_within_module_category"] <= 0.05].copy()
    if sig.empty:
        ranking_df = cur.groupby("term_key", as_index=False)["fdr_within_module_category"].min()
    else:
        ranking_df = sig.groupby("term_key", as_index=False)["fdr_within_module_category"].min()
    ranking_df = ranking_df.sort_values("fdr_within_module_category", ascending=True, kind="stable")
    term_keys = ranking_df["term_key"].head(int(top_terms)).tolist()
    if not term_keys:
        term_keys = (
            cur.groupby("term_key", as_index=False)["fdr_within_module_category"]
            .min()
            .sort_values("fdr_within_module_category", ascending=True, kind="stable")
            .head(int(top_terms))["term_key"]
            .tolist()
        )

    sub = cur.loc[cur["term_key"].isin(term_keys)].copy()
    if sub.empty:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No GO terms selected for heatmap", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return

    term_label_map = (
        sub.sort_values("fdr_within_module_category", ascending=True, kind="stable")
        .drop_duplicates(subset=["term_key"], keep="first")
        .set_index("term_key")["term_label"]
        .to_dict()
    )
    module_order = selected_modules_df["module_id"].astype(str).tolist()

    mat_df = sub.pivot_table(
        index="module_id",
        columns="term_key",
        values="fdr_within_module_category",
        aggfunc="min",
    )
    mat_df = mat_df.reindex(index=module_order, columns=term_keys)
    mat = -np.log10(np.clip(mat_df.fillna(1.0).values.astype(float), 1e-300, 1.0))

    plt.figure(figsize=(max(10, 0.55 * len(term_keys)), max(6, 0.35 * len(module_order))))
    im = plt.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(5.0, float(np.nanmax(mat))))
    plt.colorbar(im, label="-log10(FDR within module/category)")

    x_labels = [term_label_map.get(k, k) for k in term_keys]
    plt.xticks(np.arange(len(term_keys)), x_labels, rotation=65, ha="right", fontsize=8)
    plt.yticks(np.arange(len(module_order)), module_order, fontsize=9)
    plt.xlabel("Top GO terms across convergent modules")
    plt.ylabel("Convergent module ID")
    plt.title("Convergent Module GO Enrichment Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int) -> str:
    if df.empty:
        return "_No rows to display._"
    show = df.loc[:, list(columns)].head(int(max_rows)).copy()
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows = []
    for _, row in show.iterrows():
        vals = []
        for col in show.columns:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def keyword_summary(top_terms: List[str]) -> str:
    if not top_terms:
        return "No enriched terms to summarize."
    joined = " ".join(t.lower() for t in top_terms)
    hints = []
    if any(k in joined for k in ["mitochond", "oxidative", "atp"]):
        hints.append("mitochondrial/energy biology")
    if any(k in joined for k in ["synap", "axon", "neuron", "neuro"]):
        hints.append("neuronal or synaptic architecture")
    if any(k in joined for k in ["immune", "inflamm", "interferon", "cytokine"]):
        hints.append("immune/inflammatory signaling")
    if any(k in joined for k in ["rna", "splice", "translation", "ribosome"]):
        hints.append("RNA processing / translation")
    if any(k in joined for k in ["cytoskeleton", "microtubule", "vesicle", "transport"]):
        hints.append("cellular transport/cytoskeletal organization")
    if not hints:
        return "Top terms do not converge on one dominant theme; interpretation remains descriptive."
    return "Top terms suggest: " + ", ".join(hints) + "."


def build_report(
    out_dir: Path,
    selected_modules_df: pd.DataFrame,
    all_results_df: pd.DataFrame,
    significant_df: pd.DataFrame,
    module_gene_map_counts: pd.DataFrame,
    selection_rule_text: str,
    background_sizes: Mapping[str, int],
) -> None:
    top_by_module: List[Dict[str, object]] = []
    for module_id, grp in significant_df.groupby("module_id"):
        cur = grp.sort_values("fdr_within_module_category", ascending=True, kind="stable").head(6)
        for _, row in cur.iterrows():
            top_by_module.append(
                {
                    "module_id": module_id,
                    "category": row["go_category"],
                    "go_term_id": row["go_term_id"],
                    "go_name": row["go_name"],
                    "fdr": row["fdr_within_module_category"],
                    "odds_ratio": row["odds_ratio"],
                    "overlap_count": row["overlap_count"],
                }
            )
    top_table_df = pd.DataFrame(top_by_module).sort_values(
        ["module_id", "fdr", "overlap_count"], ascending=[True, True, False], kind="stable"
    )

    top_terms_for_theme = (
        significant_df.sort_values("fdr_within_module_category", ascending=True, kind="stable")["go_name"].head(30).tolist()
    )
    interpretation_hint = keyword_summary(top_terms_for_theme)

    tested_counts = (
        all_results_df.groupby("go_category", as_index=False)
        .size()
        .rename(columns={"size": "tests"})
        .sort_values("tests", ascending=False, kind="stable")
    )
    total_tests = int(len(all_results_df))

    lines = [
        "# Convergent Module GO Fisher Report",
        "",
        "## Goal",
        "- Assess whether network modules convergent between PCA-model propagation and GWAS propagation map to GO biology.",
        "- Test framework: 2x2 Fisher exact test with `alternative='greater'`.",
        "",
        "## Convergent Module Selection",
        f"- Selection rule: {selection_rule_text}",
        f"- Modules selected: `{len(selected_modules_df)}`",
        "",
        markdown_table(
            selected_modules_df,
            columns=[
                "module_id",
                "module_size",
                "NES_pca",
                "NES_gwas",
                "padj_pca",
                "padj_gwas",
                "convergence_min_nes",
            ],
            max_rows=50,
        ),
        "",
        "## Module Gene Mapping to GO",
        markdown_table(
            module_gene_map_counts,
            columns=[
                "module_id",
                "module_size_total",
                "mapped_bp",
                "mapped_cc",
                "mapped_mf",
            ],
            max_rows=50,
        ),
        "",
        "## Background Universe Sizes",
        f"- BP background size: `{background_sizes.get('biological_process', 0)}` genes",
        f"- CC background size: `{background_sizes.get('cellular_component', 0)}` genes",
        f"- MF background size: `{background_sizes.get('molecular_function', 0)}` genes",
        "",
        "## Number of GO Tests",
        markdown_table(tested_counts, columns=["go_category", "tests"], max_rows=10),
        f"- Total tests across modules and GO categories: `{total_tests}`",
        "",
        "## Top Significant GO Terms by Module",
        markdown_table(
            top_table_df,
            columns=["module_id", "category", "go_term_id", "go_name", "fdr", "odds_ratio", "overlap_count"],
            max_rows=120,
        ),
        "",
        "## Visual Outputs",
        "- Per-module top GO bars: `module_<id>_top_go_terms.png`",
        "- Cross-module summary heatmap: `convergent_modules_go_heatmap.png`",
        "",
        "## Cautious Interpretation",
        f"- {interpretation_hint}",
        "- Results are statistical enrichments over the selected background and should be treated as hypothesis-generating.",
        "- These findings do not establish causality; they indicate which pathways/complexes are overrepresented in convergent modules.",
        "",
        "## Main Question",
        "- The convergent PCA+GWAS modules do map to recognizable GO structures in this run, with signal concentrated in a subset of modules/terms.",
        "- Biological interpretation appears feasible but should be validated with orthogonal evidence and disease-context filtering.",
    ]
    (out_dir / "module_go_fisher_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run module-level GO Fisher exact enrichment for convergent PCA+GWAS INTACT modules."
    )
    parser.add_argument("--module-clusters-path", type=Path, default=DEFAULT_MODULE_CLUSTERS_PATH)
    parser.add_argument("--module-enrichment-summary", type=Path, default=DEFAULT_MODULE_ENRICHMENT_SUMMARY)
    parser.add_argument("--go-bp-path", type=Path, default=DEFAULT_GO_BP_PATH)
    parser.add_argument("--go-cc-path", type=Path, default=DEFAULT_GO_CC_PATH)
    parser.add_argument("--go-mf-path", type=Path, default=DEFAULT_GO_MF_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    parser.add_argument("--module-fdr-threshold", type=float, default=0.05)
    parser.add_argument("--min-module-size", type=int, default=10)
    parser.add_argument(
        "--no-require-positive-es",
        action="store_true",
        help="Disable ES>0 requirement for PCA/GWAS when selecting convergent modules.",
    )
    parser.add_argument("--max-convergent-modules", type=int, default=0)

    parser.add_argument("--plot-top-terms-per-category", type=int, default=8)
    parser.add_argument("--heatmap-top-terms", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    module_dict, module_size_df = load_modules(args.module_clusters_path)
    all_module_universe = set().union(*module_dict.values()) if module_dict else set()

    selected_modules_df = select_convergent_modules(
        comparison_summary_path=args.module_enrichment_summary,
        min_module_size=int(args.min_module_size),
        fdr_threshold=float(args.module_fdr_threshold),
        require_positive_es=not bool(args.no_require_positive_es),
        max_modules=int(args.max_convergent_modules),
    )
    if selected_modules_df.empty:
        raise RuntimeError("No convergent modules selected with current rule. Relax thresholds and rerun.")

    selected_ids = selected_modules_df["module_id"].astype(str).tolist()
    missing_ids = [m for m in selected_ids if m not in module_dict]
    if missing_ids:
        raise RuntimeError(f"Selected modules are missing from cluster file: {missing_ids[:20]}")

    selected_modules_df.to_csv(out_dir / "selected_convergent_modules.csv", index=False)

    go_categories = [
        GoCategory("biological_process", args.go_bp_path),
        GoCategory("cellular_component", args.go_cc_path),
        GoCategory("molecular_function", args.go_mf_path),
    ]

    go_term_maps: Dict[str, Dict[str, Set[str]]] = {}
    go_term_meta: Dict[str, pd.DataFrame] = {}
    background_by_category: Dict[str, Set[str]] = {}
    go_universe_sizes: Dict[str, int] = {}
    for cat in go_categories:
        term_map, term_meta = load_go_terms(cat.path, cat.category)
        go_term_maps[cat.category] = term_map
        go_term_meta[cat.category] = term_meta
        cat_go_universe = set().union(*term_map.values()) if term_map else set()
        go_universe_sizes[cat.category] = len(cat_go_universe)
        background_by_category[cat.category] = all_module_universe.intersection(cat_go_universe)

    per_module_rows: List[pd.DataFrame] = []
    mapping_rows: List[Dict[str, object]] = []

    for module_id in selected_ids:
        module_genes = module_dict[module_id]
        module_label = safe_module_label(module_id)
        module_result_parts: List[pd.DataFrame] = []

        mapped_bp = len(module_genes.intersection(background_by_category["biological_process"]))
        mapped_cc = len(module_genes.intersection(background_by_category["cellular_component"]))
        mapped_mf = len(module_genes.intersection(background_by_category["molecular_function"]))
        mapping_rows.append(
            {
                "module_id": module_id,
                "module_size_total": int(len(module_genes)),
                "mapped_bp": int(mapped_bp),
                "mapped_cc": int(mapped_cc),
                "mapped_mf": int(mapped_mf),
            }
        )

        for cat in go_categories:
            category_df = fisher_for_module_category(
                module_id=module_id,
                module_genes=module_genes,
                category=cat.category,
                term_to_genes=go_term_maps[cat.category],
                term_meta=go_term_meta[cat.category],
                background_genes=background_by_category[cat.category],
            )
            if category_df.empty:
                continue
            category_out = out_dir / f"module_{module_label}_go_{cat.category}_fisher.csv"
            category_df.to_csv(category_out, index=False)
            module_result_parts.append(category_df)

        if module_result_parts:
            module_full = pd.concat(module_result_parts, ignore_index=True)
            plot_module_top_terms(
                module_id=module_id,
                module_df=module_full,
                out_path=out_dir / f"module_{module_label}_top_go_terms.png",
                top_n=int(args.plot_top_terms_per_category),
            )
            per_module_rows.append(module_full)

    if not per_module_rows:
        raise RuntimeError("No Fisher results were produced for selected modules.")

    all_results_df = pd.concat(per_module_rows, ignore_index=True)
    all_results_df["fdr_global_all_tests"] = benjamini_hochberg(all_results_df["p_value"].to_numpy())
    all_results_df["fdr_within_go_category_global"] = np.nan
    for category in all_results_df["go_category"].dropna().unique().tolist():
        idx = all_results_df["go_category"] == category
        all_results_df.loc[idx, "fdr_within_go_category_global"] = benjamini_hochberg(
            all_results_df.loc[idx, "p_value"].to_numpy()
        )
    all_results_df = all_results_df.sort_values(
        ["fdr_within_module_category", "fdr_global_all_tests", "p_value", "odds_ratio"],
        ascending=[True, True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    all_results_df.to_csv(out_dir / "all_convergent_modules_go_fisher_results.csv", index=False)

    significant_df = all_results_df.loc[all_results_df["fdr_within_module_category"] <= 0.05].copy()
    significant_df = significant_df.sort_values(
        ["module_id", "go_category", "fdr_within_module_category", "odds_ratio"],
        ascending=[True, True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    significant_df.to_csv(out_dir / "significant_go_terms_by_module.csv", index=False)

    module_gene_map_counts = pd.DataFrame(mapping_rows).sort_values("module_id", kind="stable").reset_index(drop=True)
    selected_modules_report = selected_modules_df.merge(
        module_gene_map_counts,
        on="module_id",
        how="left",
    )
    selected_modules_report = selected_modules_report.sort_values(
        ["convergence_min_nes", "convergence_mean_nes"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)
    selected_modules_report.to_csv(out_dir / "selected_convergent_modules.csv", index=False)

    plot_convergent_heatmap(
        all_results_df=all_results_df,
        selected_modules_df=selected_modules_report,
        out_path=out_dir / "convergent_modules_go_heatmap.png",
        top_terms=int(args.heatmap_top_terms),
    )

    selection_rule_text = (
        f"module_size >= {int(args.min_module_size)}, "
        f"padj_pca <= {float(args.module_fdr_threshold)}, "
        f"padj_gwas <= {float(args.module_fdr_threshold)}, "
        "NES_pca > 0, NES_gwas > 0"
        + (", ES_pca > 0, ES_gwas > 0" if not bool(args.no_require_positive_es) else "")
    )
    build_report(
        out_dir=out_dir,
        selected_modules_df=selected_modules_report,
        all_results_df=all_results_df,
        significant_df=significant_df,
        module_gene_map_counts=module_gene_map_counts,
        selection_rule_text=selection_rule_text,
        background_sizes={k: len(v) for k, v in background_by_category.items()},
    )

    summary = {
        "script": "module_go_fisher_analysis.py",
        "status": "completed",
        "inputs": {
            "module_clusters_path": str(args.module_clusters_path),
            "module_enrichment_summary": str(args.module_enrichment_summary),
            "go_bp_path": str(args.go_bp_path),
            "go_cc_path": str(args.go_cc_path),
            "go_mf_path": str(args.go_mf_path),
        },
        "selection_rule": selection_rule_text,
        "selected_module_count": int(len(selected_modules_report)),
        "selected_modules": selected_modules_report["module_id"].astype(str).tolist(),
        "module_universe_size": int(len(all_module_universe)),
        "go_universe_sizes": {k: int(v) for k, v in go_universe_sizes.items()},
        "background_sizes": {k: int(len(v)) for k, v in background_by_category.items()},
        "total_tests": int(len(all_results_df)),
        "significant_tests_fdr_module_category_0_05": int((all_results_df["fdr_within_module_category"] <= 0.05).sum()),
        "outputs": {
            "out_dir": str(out_dir),
            "selected_convergent_modules": str(out_dir / "selected_convergent_modules.csv"),
            "all_results": str(out_dir / "all_convergent_modules_go_fisher_results.csv"),
            "significant_results": str(out_dir / "significant_go_terms_by_module.csv"),
            "heatmap_png": str(out_dir / "convergent_modules_go_heatmap.png"),
            "report_md": str(out_dir / "module_go_fisher_report.md"),
        },
    }
    (out_dir / "module_go_fisher_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Module GO Fisher analysis completed.")
    print(f"Output directory: {out_dir}")
    print(f"Selected convergent modules: {len(selected_modules_report)}")
    print(f"Total GO Fisher tests: {len(all_results_df)}")
    print(f"Significant terms (FDR within module/category <= 0.05): {len(significant_df)}")


if __name__ == "__main__":
    main()

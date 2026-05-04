#!/home/viguinijpv/python310/bin/python3.10
"""Run Word2Vec-based global PCA(L1) vs GWAS network propagation analysis.

This wrapper keeps comparability with the reference INTACT settings used around:
- module_go_fisher_2026-04-21
- intact_ppr_module_enrichment_functional_text_hpa_2026-04-17

Only the text embedding source changes to Word2Vec.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/viguinijpv/200.18.99.75:8000/IC/src")
TRAINING_DIR = PROJECT_ROOT / "training"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

DEFAULT_EMBEDDING_PICKLE = (
    PROJECT_ROOT
    / "data"
    / "als_cs_gene_tables"
    / "word2vec_hpa_brain_muscle_embeddings_only_20260423"
    / "embeddings"
    / "word2vec_gene_embeddings.pkl"
)

DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "als_cs_gene_tables"
    / f"word2vec_pca_gwas_network_analysis_{date.today().strftime('%Y%m%d')}"
)

DEFAULT_NETWORK_PATH = REFERENCE_DIR / "intact_netw_filtered_networkx.obj"
DEFAULT_BACKGROUND_PPR_PATH = REFERENCE_DIR / "ppr_scores_ec_maxit2000_curtol1e18.csv"
DEFAULT_HPA_PATH = REFERENCE_DIR / "rna_tissue_consensus.tsv"
DEFAULT_HGNC_PATH = REFERENCE_DIR / "hgnc_complete_set.txt"
DEFAULT_OT_JSON_PATH = PROJECT_ROOT / "external" / "OT-MONDO_0004976-associated-targets-2_12_2026-v25_12.json"


@dataclass
class StagePaths:
    root: Path
    global_model: Path
    ppr_three_way: Path
    module_enrichment: Path
    reports: Path
    logs: Path


def normalize_gene_id(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    if "." in s:
        s = s.split(".", 1)[0]
    return s if s else None


def normalize_gene_symbol(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().upper()
    if not s or s == "NAN":
        return None
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Word2Vec global PCA(L1) + INTACT propagation analysis focused on PCA-model vs GWAS, "
            "using gold-standard ALS labels."
        )
    )
    parser.add_argument("--embedding-pickle", type=Path, default=DEFAULT_EMBEDDING_PICKLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--hpa-path", type=Path, default=DEFAULT_HPA_PATH)
    parser.add_argument("--hgnc-path", type=Path, default=DEFAULT_HGNC_PATH)
    parser.add_argument("--network-path", type=Path, default=DEFAULT_NETWORK_PATH)
    parser.add_argument("--background-ppr-path", type=Path, default=DEFAULT_BACKGROUND_PPR_PATH)
    parser.add_argument("--ot-json-path", type=Path, default=DEFAULT_OT_JSON_PATH)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--regularization-strength", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-module-enrichment",
        action="store_true",
        help="Skip fgsea module-enrichment stage (network propagation stage will still run).",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="If set, remove existing output directory before running.",
    )
    return parser.parse_args()


def init_stage_paths(root: Path) -> StagePaths:
    return StagePaths(
        root=root,
        global_model=root / "global_model_l1",
        ppr_three_way=root / "intact_ppr_three_way",
        module_enrichment=root / "intact_module_enrichment",
        reports=root / "reports",
        logs=root / "logs",
    )


def ensure_dirs(paths: StagePaths) -> None:
    for p in [paths.root, paths.global_model, paths.ppr_three_way, paths.module_enrichment, paths.reports, paths.logs]:
        p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(x) for x in cmd) + "\n\n")
        proc = subprocess.run([str(x) for x in cmd], stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
        except Exception:
            tail = "(could not read log tail)"
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}.\n"
            f"Command: {' '.join(str(x) for x in cmd)}\n"
            f"Log: {log_path}\n\nLast log lines:\n{tail}"
        )


def train_global_word2vec_model(args: argparse.Namespace, paths: StagePaths) -> None:
    cmd = [
        str(args.python_bin),
        str(TRAINING_DIR / "train_global_functional_model_hpa_pca.py"),
        "--embedding-path",
        str(args.embedding_pickle),
        "--hpa-path",
        str(args.hpa_path),
        "--hgnc-path",
        str(args.hgnc_path),
        "--out-dir",
        str(paths.global_model),
        "--penalty",
        "l1",
        "--regularization-strength",
        str(float(args.regularization_strength)),
        "--pca-dim",
        str(int(args.pca_dim)),
        "--max-iter",
        str(int(args.max_iter)),
        "--random-state",
        str(int(args.random_state)),
    ]
    run_cmd(cmd, paths.logs / "01_train_global_model_l1.log")


def build_project_embedding_universe_table(global_model_dir: Path, out_csv: Path) -> Dict[str, int]:
    src = global_model_dir / "candidate_universe_embeddings_hpa.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing expected universe table: {src}")

    df = pd.read_csv(src)
    if "gene_symbol" not in df.columns:
        raise ValueError("candidate_universe_embeddings_hpa.csv missing gene_symbol")

    cur = df.copy()
    if "gene_id" not in cur.columns:
        cur["gene_id"] = np.nan
    if "label_positive" not in cur.columns:
        cur["label_positive"] = 0

    cur["gene_id"] = cur["gene_id"].map(normalize_gene_id)
    cur["gene_symbol"] = cur["gene_symbol"].map(normalize_gene_symbol)
    cur["label_positive"] = pd.to_numeric(cur["label_positive"], errors="coerce").fillna(0).astype(int)
    cur = cur.dropna(subset=["gene_symbol"]).copy()

    with_gid = cur.loc[cur["gene_id"].notna()].drop_duplicates(subset=["gene_id"], keep="first")
    without_gid = cur.loc[cur["gene_id"].isna()].drop_duplicates(subset=["gene_symbol"], keep="first")
    out = pd.concat([with_gid, without_gid], axis=0, ignore_index=True)
    out = out.sort_values(["label_positive", "gene_symbol"], ascending=[False, True], kind="stable").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out[["gene_id", "gene_symbol", "label_positive"]].to_csv(out_csv, index=False)

    return {
        "rows_total": int(len(cur)),
        "rows_exported": int(len(out)),
        "unique_gene_symbols_exported": int(out["gene_symbol"].nunique()),
        "unique_gene_ids_exported": int(out["gene_id"].dropna().nunique()),
        "positive_rows_exported": int(out["label_positive"].sum()),
        "positive_unique_symbols_exported": int(out.loc[out["label_positive"] == 1, "gene_symbol"].nunique()),
    }


def run_intact_three_way(args: argparse.Namespace, paths: StagePaths, project_gene_table: Path) -> None:
    pred_path = paths.global_model / "all_gene_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing model predictions for seed extraction: {pred_path}")

    cmd = [
        str(args.python_bin),
        str(TRAINING_DIR / "intact_ppr_pipeline.py"),
        "--run-three-way-comparison",
        "--model-predictions-path",
        str(pred_path),
        "--project-gene-table",
        str(project_gene_table),
        "--network-path",
        str(args.network_path),
        "--background-ppr-path",
        str(args.background_ppr_path),
        "--ot-json-path",
        str(args.ot_json_path),
        "--hgnc-path",
        str(args.hgnc_path),
        "--model-source-label",
        "Word2Vec_PCA_L1",
        "--model-prior-description",
        "Word2Vec embeddings + HPA brain/muscle global model with PCA and L1 regularization.",
        "--gwas-source-column",
        "gwasCredibleSets",
        "--gwas-threshold",
        "0.5",
        "--clinvar-source-column",
        "eva",
        "--clinvar-threshold",
        "0.5",
        "--ppr-alpha",
        "0.85",
        "--ppr-max-iter",
        "500",
        "--ppr-tol",
        "1e-9",
        "--out-dir",
        str(paths.ppr_three_way),
    ]
    run_cmd(cmd, paths.logs / "02_intact_three_way.log")


def run_module_enrichment(args: argparse.Namespace, paths: StagePaths) -> None:
    cmd = [
        str(args.python_bin),
        str(TRAINING_DIR / "intact_ppr_pipeline.py"),
        "--run-module-enrichment",
        "--module-enrichment-input-dir",
        str(paths.ppr_three_way),
        "--module-min-size",
        "3",
        "--module-fdr-threshold",
        "0.01",
        "--module-top-plot-n",
        "15",
        "--module-heatmap-top-rows",
        "60",
        "--out-dir",
        str(paths.module_enrichment),
    ]
    run_cmd(cmd, paths.logs / "03_module_enrichment.log")


def load_network_node_set(network_path: Path) -> set[str]:
    with network_path.open("rb") as f:
        graph = pickle.load(f)
    return {str(n) for n in graph.nodes()}


def compute_candidate_network_stats(project_table_csv: Path, network_path: Path) -> Dict[str, object]:
    df = pd.read_csv(project_table_csv)
    df["gene_id_norm"] = df["gene_id"].map(normalize_gene_id)
    df["gene_symbol_norm"] = df["gene_symbol"].map(normalize_gene_symbol)
    df["label_positive"] = pd.to_numeric(df.get("label_positive", 0), errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["gene_symbol_norm"]).copy()

    node_set = load_network_node_set(network_path)
    df["mapped_to_network"] = df["gene_id_norm"].isin(node_set)

    mapped = df.loc[df["mapped_to_network"]].copy()
    mapped_pos = mapped.loc[mapped["label_positive"] == 1].copy()

    all_pos = df.loc[df["label_positive"] == 1].copy()

    return {
        "candidate_genes_total_embeddings_universe_unique_symbols": int(df["gene_symbol_norm"].nunique()),
        "candidate_genes_total_embeddings_universe_unique_gene_ids": int(df["gene_id_norm"].dropna().nunique()),
        "candidate_genes_in_network_unique_gene_ids": int(mapped["gene_id_norm"].dropna().nunique()),
        "candidate_genes_in_network_unique_symbols": int(mapped["gene_symbol_norm"].nunique()),
        "positive_genes_gold_standard_in_universe_unique_symbols": int(all_pos["gene_symbol_norm"].nunique()),
        "positive_genes_gold_standard_in_universe_unique_gene_ids": int(all_pos["gene_id_norm"].dropna().nunique()),
        "positive_genes_gold_standard_in_network_unique_symbols": int(mapped_pos["gene_symbol_norm"].nunique()),
        "positive_genes_gold_standard_in_network_unique_gene_ids": int(mapped_pos["gene_id_norm"].dropna().nunique()),
        "network_node_count": int(len(node_set)),
    }


def load_pairwise_model_vs_gwas_metrics(three_way_dir: Path) -> Dict[str, float]:
    pairwise_path = three_way_dir / "pairwise_comparison_metrics.csv"
    if not pairwise_path.exists():
        raise FileNotFoundError(f"Missing pairwise metrics: {pairwise_path}")
    df = pd.read_csv(pairwise_path)
    row = df.loc[df["pair"].astype(str) == "model_vs_gwas"]
    if row.empty:
        raise ValueError("pairwise_comparison_metrics.csv does not contain pair=model_vs_gwas")
    r = row.iloc[0]
    return {
        "raw_pearson": float(r["raw_pearson"]),
        "raw_spearman": float(r["raw_spearman"]),
        "normalized_pearson": float(r["norm_pearson"]),
        "normalized_spearman": float(r["norm_spearman"]),
        "raw_top50_overlap": float(r["raw_top50_overlap"]),
        "raw_top100_overlap": float(r["raw_top100_overlap"]),
        "normalized_top50_overlap": float(r["norm_top50_overlap"]),
        "normalized_top100_overlap": float(r["norm_top100_overlap"]),
    }


def load_score_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "gene_id" not in df.columns or "score" not in df.columns:
        raise ValueError(f"Score file missing gene_id/score columns: {path}")
    cur = df.copy()
    cur["gene_id"] = cur["gene_id"].map(normalize_gene_id)
    cur["score"] = pd.to_numeric(cur["score"], errors="coerce")
    cur = cur.dropna(subset=["gene_id", "score"]).drop_duplicates(subset=["gene_id"], keep="first")
    return pd.Series(cur["score"].values, index=cur["gene_id"].astype(str).values)


def compute_direct_correlations(three_way_dir: Path) -> Dict[str, float]:
    model_raw = load_score_series(three_way_dir / "ppr_scores_model_seeds.csv")
    gwas_raw = load_score_series(three_way_dir / "ppr_scores_gwas_seeds.csv")
    model_norm = load_score_series(three_way_dir / "ppr_scores_model_seeds_normalized.csv")
    gwas_norm = load_score_series(three_way_dir / "ppr_scores_gwas_seeds_normalized.csv")

    shared_raw = model_raw.index.intersection(gwas_raw.index)
    shared_norm = model_norm.index.intersection(gwas_norm.index)

    raw_a = model_raw.loc[shared_raw].astype(float)
    raw_b = gwas_raw.loc[shared_raw].astype(float)
    norm_a = model_norm.loc[shared_norm].astype(float)
    norm_b = gwas_norm.loc[shared_norm].astype(float)

    return {
        "shared_raw_genes": float(len(shared_raw)),
        "shared_normalized_genes": float(len(shared_norm)),
        "raw_pearson_direct": float(raw_a.corr(raw_b, method="pearson")),
        "raw_spearman_direct": float(raw_a.corr(raw_b, method="spearman")),
        "normalized_pearson_direct": float(norm_a.corr(norm_b, method="pearson")),
        "normalized_spearman_direct": float(norm_a.corr(norm_b, method="spearman")),
    }


def plot_model_vs_gwas_scatter(
    three_way_dir: Path,
    project_table_csv: Path,
    out_png: Path,
) -> Dict[str, int]:
    model_norm = load_score_series(three_way_dir / "ppr_scores_model_seeds_normalized.csv")
    gwas_norm = load_score_series(three_way_dir / "ppr_scores_gwas_seeds_normalized.csv")

    shared = model_norm.index.intersection(gwas_norm.index)
    df = pd.DataFrame(
        {
            "gene_id": shared.astype(str),
            "model_score": model_norm.loc[shared].astype(float).values,
            "gwas_score": gwas_norm.loc[shared].astype(float).values,
        }
    )

    labels = pd.read_csv(project_table_csv)
    labels["gene_id"] = labels["gene_id"].map(normalize_gene_id)
    labels["label_positive"] = pd.to_numeric(labels.get("label_positive", 0), errors="coerce").fillna(0).astype(int)
    labels = labels.dropna(subset=["gene_id"]).drop_duplicates(subset=["gene_id"], keep="first")
    df = df.merge(labels[["gene_id", "label_positive"]], on="gene_id", how="left")
    df["label_positive"] = df["label_positive"].fillna(0).astype(int)

    pos = df.loc[df["label_positive"] == 1]
    neg = df.loc[df["label_positive"] == 0]

    plt.figure(figsize=(8.5, 7.0))
    plt.scatter(neg["gwas_score"], neg["model_score"], s=12, alpha=0.45, c="#4C78A8", label="Non-gold")
    if not pos.empty:
        plt.scatter(
            pos["gwas_score"],
            pos["model_score"],
            s=36,
            alpha=0.9,
            c="#D62728",
            edgecolors="white",
            linewidths=0.4,
            label="Gold-standard",
        )
    low = float(min(df["gwas_score"].min(), df["model_score"].min()))
    high = float(max(df["gwas_score"].max(), df["model_score"].max()))
    plt.plot([low, high], [low, high], linestyle="--", linewidth=1.0, color="#666666")
    plt.xlabel("GWAS normalized propagation score")
    plt.ylabel("Word2Vec PCA(L1) normalized propagation score")
    plt.title("INTACT Propagation: Word2Vec PCA(L1) vs GWAS")
    plt.grid(alpha=0.25, linewidth=0.5)
    plt.legend(frameon=True, fontsize=9, loc="best")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()

    return {
        "shared_points": int(len(df)),
        "shared_gold_standard_points": int((df["label_positive"] == 1).sum()),
    }


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(
    args: argparse.Namespace,
    paths: StagePaths,
    project_table_path: Path,
    universe_table_stats: Dict[str, int],
    candidate_network_stats: Dict[str, object],
    pair_metrics: Dict[str, float],
    direct_corr: Dict[str, float],
    scatter_info: Dict[str, int],
) -> Path:
    run_summary = load_json(paths.global_model / "run_summary.json")
    model_summary = load_json(paths.global_model / "model_summary.json")
    three_way_summary = load_json(paths.ppr_three_way / "three_way_summary.json")

    report_path = paths.reports / "word2vec_pca_l1_vs_gwas_network_propagation_report.md"
    lines: List[str] = []
    lines.append("# Word2Vec PCA(L1) vs GWAS Network Propagation Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Non-locus global network propagation analysis (INTACT graph).")
    lines.append("- PCA model uses Word2Vec embeddings with L1 regularization.")
    lines.append("- Positive labels use gold-standard ALS genes from `config.VALIDATION_GENES` (reference article list).")
    lines.append("")
    lines.append("## Settings Aligned to Reference")
    lines.append("- PPR alpha: `0.85`")
    lines.append("- PPR max_iter: `500`")
    lines.append("- PPR tol: `1e-9`")
    lines.append("- GWAS seed source: `gwasCredibleSets >= 0.5`")
    lines.append("- Module enrichment settings: `min_module_size=3`, `FDR=0.01`")
    lines.append("- Reference setting target: `module_go_fisher_2026-04-21` lineage")
    lines.append("")
    lines.append("## Candidate Universe (Embeddings)")
    lines.append(f"- Embedding pickle: `{args.embedding_pickle}`")
    lines.append(f"- Exported project table: `{project_table_path}`")
    lines.append(f"- Exported rows: `{universe_table_stats['rows_exported']}`")
    lines.append(
        f"- Candidate genes in network (max with embeddings, unique gene IDs): `{candidate_network_stats['candidate_genes_in_network_unique_gene_ids']}`"
    )
    lines.append(
        f"- Positive genes in embedding universe (gold-standard, unique symbols): `{candidate_network_stats['positive_genes_gold_standard_in_universe_unique_symbols']}`"
    )
    lines.append(
        f"- Positive genes in network (gold-standard, unique gene IDs): `{candidate_network_stats['positive_genes_gold_standard_in_network_unique_gene_ids']}`"
    )
    lines.append("")
    lines.append("## Global Model (Word2Vec PCA L1)")
    lines.append(f"- Penalty: `{model_summary.get('penalty')}`")
    lines.append(f"- C: `{model_summary.get('regularization_strength_C')}`")
    lines.append(
        f"- PCA components used: `{model_summary.get('embedding_pca_components_used')}` "
        f"(explained variance sum `{model_summary.get('embedding_pca_explained_variance_ratio_sum')}`)"
    )
    lines.append(f"- Model output dir: `{paths.global_model}`")
    lines.append("")
    lines.append("## PCA vs GWAS Correlations (Network Scores)")
    lines.append("- Pairwise metrics from `pairwise_comparison_metrics.csv` (`pair=model_vs_gwas`):")
    lines.append(f"  - Raw Pearson: `{pair_metrics['raw_pearson']:.6f}`")
    lines.append(f"  - Raw Spearman: `{pair_metrics['raw_spearman']:.6f}`")
    lines.append(f"  - Normalized Pearson: `{pair_metrics['normalized_pearson']:.6f}`")
    lines.append(f"  - Normalized Spearman: `{pair_metrics['normalized_spearman']:.6f}`")
    lines.append("- Direct recomputation check:")
    lines.append(f"  - Raw Pearson: `{direct_corr['raw_pearson_direct']:.6f}`")
    lines.append(f"  - Raw Spearman: `{direct_corr['raw_spearman_direct']:.6f}`")
    lines.append(f"  - Normalized Pearson: `{direct_corr['normalized_pearson_direct']:.6f}`")
    lines.append(f"  - Normalized Spearman: `{direct_corr['normalized_spearman_direct']:.6f}`")
    lines.append("")
    lines.append("## Main Outputs")
    lines.append(f"- Three-way propagation dir: `{paths.ppr_three_way}`")
    lines.append(f"- Pairwise metrics CSV: `{paths.ppr_three_way / 'pairwise_comparison_metrics.csv'}`")
    lines.append(f"- Three-way summary JSON: `{paths.ppr_three_way / 'three_way_summary.json'}`")
    lines.append(f"- Scatter plot (PCA vs GWAS normalized): `{paths.reports / 'model_vs_gwas_normalized_scatter.png'}`")
    if not args.skip_module_enrichment:
        lines.append(f"- Module enrichment dir: `{paths.module_enrichment}`")
        lines.append(f"- Module enrichment summary: `{paths.module_enrichment / 'module_enrichment_summary.json'}`")
    lines.append("")
    lines.append("## Seed Mapping Snapshot")
    model_mapped = (
        three_way_summary.get("model_seed_mapping", {}).get("mapped_seed_count")
        if isinstance(three_way_summary.get("model_seed_mapping"), dict)
        else None
    )
    gwas_mapped = (
        three_way_summary.get("gwas_seed_mapping", {}).get("mapped_seed_count")
        if isinstance(three_way_summary.get("gwas_seed_mapping"), dict)
        else None
    )
    lines.append(f"- Model seeds mapped: `{model_mapped}`")
    lines.append(f"- GWAS seeds mapped: `{gwas_mapped}`")
    lines.append(f"- Shared genes used in scatter: `{scatter_info['shared_points']}`")
    lines.append(f"- Shared gold-standard genes in scatter: `{scatter_info['shared_gold_standard_points']}`")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    if not args.embedding_pickle.exists():
        raise FileNotFoundError(f"Word2Vec embedding pickle not found: {args.embedding_pickle}")

    if args.out_dir.exists() and args.overwrite_output:
        import shutil

        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = init_stage_paths(args.out_dir)
    ensure_dirs(paths)

    train_global_word2vec_model(args, paths)

    project_table_path = paths.root / "project_gene_table_word2vec_embedding_universe.csv"
    universe_table_stats = build_project_embedding_universe_table(paths.global_model, project_table_path)

    run_intact_three_way(args, paths, project_table_path)

    if not args.skip_module_enrichment:
        run_module_enrichment(args, paths)

    candidate_network_stats = compute_candidate_network_stats(project_table_path, args.network_path)
    pair_metrics = load_pairwise_model_vs_gwas_metrics(paths.ppr_three_way)
    direct_corr = compute_direct_correlations(paths.ppr_three_way)

    scatter_path = paths.reports / "model_vs_gwas_normalized_scatter.png"
    scatter_info = plot_model_vs_gwas_scatter(paths.ppr_three_way, project_table_path, scatter_path)

    report_path = write_report(
        args=args,
        paths=paths,
        project_table_path=project_table_path,
        universe_table_stats=universe_table_stats,
        candidate_network_stats=candidate_network_stats,
        pair_metrics=pair_metrics,
        direct_corr=direct_corr,
        scatter_info=scatter_info,
    )

    summary = {
        "script": str(Path(__file__).resolve()),
        "output_root": str(paths.root),
        "global_model_dir": str(paths.global_model),
        "three_way_dir": str(paths.ppr_three_way),
        "module_enrichment_dir": str(paths.module_enrichment) if not args.skip_module_enrichment else None,
        "project_embedding_universe_table": str(project_table_path),
        "candidate_network_stats": candidate_network_stats,
        "model_vs_gwas_pairwise_metrics": pair_metrics,
        "model_vs_gwas_direct_correlations": direct_corr,
        "report_md": str(report_path),
        "scatter_png": str(scatter_path),
    }
    (paths.reports / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Completed Word2Vec PCA(L1) vs GWAS network analysis.\nOutput root: {paths.root}")
    print(f"Report: {report_path}")
    print(f"Model-vs-GWAS normalized Spearman: {pair_metrics['normalized_spearman']:.6f}")
    print(f"Candidate genes in network: {candidate_network_stats['candidate_genes_in_network_unique_gene_ids']}")
    print(f"Gold-standard positives in network: {candidate_network_stats['positive_genes_gold_standard_in_network_unique_gene_ids']}")


if __name__ == "__main__":
    main()


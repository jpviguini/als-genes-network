#!/home/viguinijpv/python310/bin/python3.10
"""INTACT PPR pipeline adapted from supp_fig4 eqstep10a/10b notebooks.

Notebook-faithful core logic preserved:
- PPR via networkx.pagerank with personalization dict
- log normalization: log(PPR) - log(background_ec_score)

Current implementation executes:
1) setup/inspection (network, background, project genes)
2) model seed selection from selected comparison run
3) ClinVar/EVA seed selection from existing Open Targets disease-association logic
4) seed-to-network mapping
5) PPR for model and ClinVar seed sets
6) normalization against background EC scores
7) basic comparison metrics
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import pickle
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

PROJECT_ROOT = Path("/home/viguinijpv/200.18.99.75:8000/IC/src")
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
DEFAULT_NETWORK_PATH = DEFAULT_REFERENCE_DIR / "intact_netw_filtered_networkx.obj"
DEFAULT_BACKGROUND_PPR_PATH = DEFAULT_REFERENCE_DIR / "ppr_scores_ec_maxit2000_curtol1e18.csv"
DEFAULT_HGNC_PATH = DEFAULT_REFERENCE_DIR / "hgnc_complete_set.txt"
DEFAULT_GENE_TABLE_DIR = PROJECT_ROOT / "data" / "als_cs_gene_tables"
DEFAULT_COMPARISON_ROOT = (
    PROJECT_ROOT
    / "data"
    / "als_cs_gene_tables"
    / "full_comparison"
    / "reduced_baseline_full_comparison_20260406"
)
DEFAULT_OT_ASSOC_JSON = PROJECT_ROOT / "external" / "OT-MONDO_0004976-associated-targets-2_12_2026-v25_12.json"
DEFAULT_OUT_DIR = DEFAULT_REFERENCE_DIR / f"intact_ppr_model_vs_clinvar_{date.today().isoformat()}"
DEFAULT_SWEEP_OUT_DIR = DEFAULT_REFERENCE_DIR / f"intact_ppr_threshold_sweep_{date.today().isoformat()}"
DEFAULT_THREE_WAY_OUT_DIR = DEFAULT_REFERENCE_DIR / f"intact_ppr_model_vs_clinvar_vs_gwas_{date.today().isoformat()}"
DEFAULT_MODULE_ENRICH_OUT_DIR = DEFAULT_REFERENCE_DIR / f"intact_ppr_module_enrichment_{date.today().isoformat()}"

PREFERRED_PROJECT_GENE_TABLES = [
    DEFAULT_GENE_TABLE_DIR / "GCST90027164_cs_gene_candidate_feature_table_neurodegenerative_disease.csv",
    DEFAULT_GENE_TABLE_DIR / "GCST90027164_cs_gene_candidate_feature_table.csv",
    DEFAULT_GENE_TABLE_DIR / "GCST90027164_cs_gene_candidate_feature_table_motor_neuron_disease.csv",
]

ENSG_NO_VERSION_RE = re.compile(r"^ENSG\d+$")
ENSG_WITH_VERSION_RE = re.compile(r"^ENSG\d+\.\d+$")


@dataclass
class NetworkLoadResult:
    graph: nx.Graph
    node_set: set[str]
    report: Dict[str, object]


@dataclass
class BackgroundLoadResult:
    background_df: pd.DataFrame
    report: Dict[str, object]


@dataclass
class ProjectGenesResult:
    source_path: Path
    unique_genes_df: pd.DataFrame
    report: Dict[str, object]


@dataclass
class MappingResult:
    mapped_df: pd.DataFrame
    unmapped_df: pd.DataFrame
    report: Dict[str, object]


def normalize_gene_id(gene_id: object) -> Optional[str]:
    if gene_id is None:
        return None
    s = str(gene_id).strip()
    if not s or s.lower() == "nan":
        return None
    if "." in s:
        s = s.split(".", 1)[0]
    return s if s else None


def normalize_gene_symbol(gene_symbol: object) -> Optional[str]:
    if gene_symbol is None:
        return None
    s = str(gene_symbol).strip().upper()
    return s if s else None


def infer_node_identifier_format(node_ids: Sequence[str]) -> Dict[str, object]:
    total = len(node_ids)
    ensg_no_version = sum(1 for n in node_ids if ENSG_NO_VERSION_RE.match(n))
    ensg_with_version = sum(1 for n in node_ids if ENSG_WITH_VERSION_RE.match(n))
    string_nodes = sum(1 for n in node_ids if isinstance(n, str))

    if total == 0:
        identifier_guess = "unknown (empty graph)"
    elif ensg_no_version == total:
        identifier_guess = "Ensembl gene IDs (ENSG...) without version suffix"
    elif ensg_no_version + ensg_with_version == total:
        identifier_guess = "Ensembl gene IDs (mixed with/without version suffix)"
    else:
        identifier_guess = "mixed / non-Ensembl identifiers"

    return {
        "total_nodes": total,
        "string_nodes": string_nodes,
        "ensg_no_version_count": ensg_no_version,
        "ensg_with_version_count": ensg_with_version,
        "identifier_guess": identifier_guess,
    }


def load_network(network_path: Path) -> NetworkLoadResult:
    with open(network_path, "rb") as handle:
        graph = pickle.load(handle)

    node_ids = [str(n) for n in graph.nodes()]
    node_set = set(node_ids)
    format_info = infer_node_identifier_format(node_ids)

    report = {
        "network_path": str(network_path),
        "graph_type": type(graph).__name__,
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "example_node_ids": node_ids[:15],
        "identifier_format": format_info,
    }
    return NetworkLoadResult(graph=graph, node_set=node_set, report=report)


def load_background_ppr(background_path: Path, network_node_set: set[str]) -> BackgroundLoadResult:
    raw_df = pd.read_csv(background_path)

    # Notebook behavior: ec_df = pd.read_csv(..., index_col=0)
    bg_df = pd.read_csv(background_path, index_col=0)
    bg_df.index = bg_df.index.astype(str)

    if "score" not in bg_df.columns and len(bg_df.columns) == 1:
        bg_df = bg_df.rename(columns={bg_df.columns[0]: "score"})

    bg_node_set = set(bg_df.index.tolist())
    overlap = len(network_node_set.intersection(bg_node_set))

    report = {
        "background_path": str(background_path),
        "num_rows_raw": int(len(raw_df)),
        "raw_columns": [str(c) for c in raw_df.columns.tolist()],
        "num_rows_indexed": int(len(bg_df)),
        "indexed_columns": [str(c) for c in bg_df.columns.tolist()],
        "background_identifier_source": "CSV index column (loaded with index_col=0)",
        "example_background_ids": bg_df.index[:15].tolist(),
        "network_background_overlap": int(overlap),
        "network_only_ids": int(len(network_node_set - bg_node_set)),
        "background_only_ids": int(len(bg_node_set - network_node_set)),
        "background_exactly_matches_network_nodes": bool(bg_node_set == network_node_set),
    }
    return BackgroundLoadResult(background_df=bg_df, report=report)


def _table_has_required_columns(path: Path, required_cols: Sequence[str]) -> bool:
    try:
        cols = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return False
    return all(col in cols for col in required_cols)


def choose_project_gene_table(
    explicit_path: Optional[Path],
    gene_table_dir: Path,
    required_cols: Sequence[str] = ("gene_id", "gene_symbol"),
) -> Path:
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Provided project gene table does not exist: {explicit_path}")
        if not _table_has_required_columns(explicit_path, required_cols):
            raise ValueError(
                f"Provided project gene table is missing required columns {list(required_cols)}: {explicit_path}"
            )
        return explicit_path

    ordered_candidates: List[Path] = []
    for p in PREFERRED_PROJECT_GENE_TABLES:
        if p not in ordered_candidates and p.is_file():
            ordered_candidates.append(p)

    discovered = sorted(
        gene_table_dir.glob("*cs_gene_candidate_feature_table*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in discovered:
        if p not in ordered_candidates:
            ordered_candidates.append(p)

    for p in ordered_candidates:
        if _table_has_required_columns(p, required_cols):
            return p

    raise FileNotFoundError(
        f"Could not find any project gene table in {gene_table_dir} with columns {list(required_cols)}"
    )


def load_project_genes(project_gene_table_path: Path) -> ProjectGenesResult:
    df = pd.read_csv(project_gene_table_path)
    if "gene_id" not in df.columns or "gene_symbol" not in df.columns:
        raise ValueError("Project gene table must include both 'gene_id' and 'gene_symbol' columns.")

    genes_df = df[["gene_id", "gene_symbol"]].copy()
    genes_df["gene_id_raw"] = genes_df["gene_id"].astype(str).str.strip()
    genes_df["gene_symbol"] = genes_df["gene_symbol"].astype(str).str.strip()
    genes_df["gene_id_normalized"] = genes_df["gene_id_raw"].map(normalize_gene_id)
    genes_df["had_version_suffix"] = genes_df["gene_id_raw"].str.contains(
        ENSG_WITH_VERSION_RE.pattern,
        regex=True,
        na=False,
    )

    valid = genes_df.loc[genes_df["gene_id_normalized"].notna()].copy()
    unique_genes_df = valid.drop_duplicates(subset=["gene_id_normalized"], keep="first").copy()

    report = {
        "project_gene_table_path": str(project_gene_table_path),
        "project_rows": int(len(df)),
        "project_unique_gene_id_raw": int(genes_df["gene_id_raw"].nunique(dropna=True)),
        "project_unique_gene_symbol": int(genes_df["gene_symbol"].nunique(dropna=True)),
        "project_unique_gene_id_normalized": int(unique_genes_df["gene_id_normalized"].nunique(dropna=True)),
        "rows_with_ensembl_version_suffix": int(genes_df["had_version_suffix"].sum()),
        "rows_missing_gene_id_after_normalization": int(genes_df["gene_id_normalized"].isna().sum()),
    }

    return ProjectGenesResult(source_path=project_gene_table_path, unique_genes_df=unique_genes_df, report=report)


def map_project_genes_to_network(project_unique_genes_df: pd.DataFrame, network_node_set: set[str]) -> MappingResult:
    cur = project_unique_genes_df.copy()
    cur["mapped_to_network"] = cur["gene_id_normalized"].isin(network_node_set)

    mapped_df = cur.loc[cur["mapped_to_network"]].copy()
    unmapped_df = cur.loc[~cur["mapped_to_network"]].copy()

    mapped_examples = (
        mapped_df[["gene_id_normalized", "gene_symbol"]]
        .head(10)
        .rename(columns={"gene_id_normalized": "gene_id"})
        .to_dict(orient="records")
    )
    unmapped_examples = (
        unmapped_df[["gene_id_normalized", "gene_symbol"]]
        .head(10)
        .rename(columns={"gene_id_normalized": "gene_id"})
        .to_dict(orient="records")
    )

    report = {
        "project_unique_genes_used_for_mapping": int(len(cur)),
        "mapped_genes_count": int(len(mapped_df)),
        "unmapped_genes_count": int(len(unmapped_df)),
        "mapped_examples": mapped_examples,
        "unmapped_examples": unmapped_examples,
    }
    return MappingResult(mapped_df=mapped_df, unmapped_df=unmapped_df, report=report)


def select_model_run_from_comparison_folder(
    comparison_root: Path,
    preferred_penalty: str = "l1",
    force_mode: Optional[str] = None,
) -> Dict[str, object]:
    summary_csv = comparison_root / "runs" / preferred_penalty / "validation_comparison_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing validation summary: {summary_csv}")

    summary = pd.read_csv(summary_csv)
    if summary.empty:
        raise ValueError(f"Validation summary is empty: {summary_csv}")

    if force_mode is not None:
        rows = summary.loc[summary["mode"].astype(str) == str(force_mode)].copy()
        if rows.empty:
            raise ValueError(f"Requested mode '{force_mode}' not found in {summary_csv}")
        row = rows.iloc[0]
    else:
        # Use strongest PR-AUC within the selected penalty (matches training comparison style).
        summary = summary.sort_values(
            ["mean_fold_pr_auc", "mean_fold_roc_auc"],
            ascending=[False, False],
            kind="stable",
        )
        row = summary.iloc[0]

    mode = str(row["mode"])
    mode_dir = comparison_root / "runs" / preferred_penalty / "cv_lolo_gene_exclusion" / f"mode_{mode}"
    pred_path = mode_dir / "all_ranked_predictions.csv"
    model_params_path = mode_dir / "model_parameters.json"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing all_ranked_predictions.csv: {pred_path}")

    model_params: Dict[str, object] = {}
    if model_params_path.exists():
        with open(model_params_path, "r", encoding="utf-8") as f:
            model_params = json.load(f)

    return {
        "comparison_root": str(comparison_root),
        "summary_csv": str(summary_csv),
        "penalty": preferred_penalty,
        "mode": mode,
        "selected_summary_row": row.to_dict(),
        "mode_dir": str(mode_dir),
        "predictions_path": str(pred_path),
        "model_parameters_path": str(model_params_path),
        "model_parameters": model_params,
    }


def resolve_model_run_from_args(args: argparse.Namespace) -> Dict[str, object]:
    """Resolve model predictions either from comparison-root metadata or explicit CSV path."""
    explicit_path = getattr(args, "model_predictions_path", None)
    if explicit_path is not None:
        pred_path = Path(explicit_path)
        if not pred_path.exists():
            raise FileNotFoundError(f"Provided model predictions file does not exist: {pred_path}")
        return {
            "comparison_root": str(getattr(args, "comparison_root", "")),
            "summary_csv": None,
            "penalty": str(getattr(args, "model_penalty", "custom")),
            "mode": str(getattr(args, "model_mode", "custom_predictions")),
            "selected_summary_row": {},
            "mode_dir": str(pred_path.parent),
            "predictions_path": str(pred_path),
            "model_parameters_path": None,
            "model_parameters": {
                "source_type": "custom_predictions_path",
                "model_source_label": str(getattr(args, "model_source_label", "model")),
                "model_prior_description": str(
                    getattr(
                        args,
                        "model_prior_description",
                        "Model scores provided from explicit predictions file.",
                    )
                ),
            },
            "source_type": "custom_predictions_path",
        }

    return select_model_run_from_comparison_folder(
        comparison_root=args.comparison_root,
        preferred_penalty=args.model_penalty,
        force_mode=args.model_mode,
    )


def _gene_level_max_scores_from_predictions(pred_df: pd.DataFrame, score_col: str = "predicted_score") -> pd.DataFrame:
    cur = pred_df.copy()
    cur[score_col] = pd.to_numeric(cur[score_col], errors="coerce").fillna(0.0)
    cur["gene_id"] = cur["gene_id"].map(normalize_gene_id)
    cur["gene_symbol"] = cur["gene_symbol"].map(normalize_gene_symbol)

    grouped = (
        cur.dropna(subset=["gene_id", "gene_symbol"])
        .groupby(["gene_id", "gene_symbol"], as_index=False)[score_col]
        .max()
        .sort_values(score_col, ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return grouped


def choose_model_seed_threshold_from_distribution(
    gene_scores_desc: Sequence[float],
    top_window: int = 50,
    min_threshold: float = 0.60,
    max_threshold: float = 0.95,
) -> Dict[str, object]:
    scores = [float(s) for s in gene_scores_desc if pd.notna(s)]
    if len(scores) < 2:
        threshold = 0.90
        return {
            "method": "fallback_single_score",
            "threshold": float(threshold),
            "gap_index": None,
            "gap_high_score": None,
            "gap_low_score": None,
            "largest_gap": None,
        }

    n = min(top_window, len(scores) - 1)
    diffs = [scores[i] - scores[i + 1] for i in range(n)]
    gap_idx = int(np.argmax(diffs))
    gap_high = float(scores[gap_idx])
    gap_low = float(scores[gap_idx + 1])
    midpoint = (gap_high + gap_low) / 2.0
    threshold = float(np.clip(midpoint, min_threshold, max_threshold))
    threshold = round(threshold, 2)

    return {
        "method": "largest_gap_midpoint_top_window",
        "threshold": threshold,
        "gap_index": int(gap_idx),
        "gap_high_score": gap_high,
        "gap_low_score": gap_low,
        "largest_gap": float(diffs[gap_idx]),
    }


def select_model_seed_genes(
    model_predictions_path: Path,
    threshold_override: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    pred_df = pd.read_csv(model_predictions_path)
    if "predicted_score" not in pred_df.columns:
        raise ValueError(f"predicted_score column missing from {model_predictions_path}")

    gene_scores = _gene_level_max_scores_from_predictions(pred_df, score_col="predicted_score")
    dist_info = choose_model_seed_threshold_from_distribution(gene_scores["predicted_score"].tolist())
    threshold = float(threshold_override) if threshold_override is not None else float(dist_info["threshold"])

    selected = gene_scores.loc[gene_scores["predicted_score"] >= threshold].copy()
    selected = selected.sort_values("predicted_score", ascending=False, kind="stable").reset_index(drop=True)
    selected["seed_source"] = "model"
    selected["seed_score"] = selected["predicted_score"].astype(float)
    selected["seed_rule"] = f"gene_max_predicted_score >= {threshold:.2f}"

    report = {
        "model_predictions_path": str(model_predictions_path),
        "total_unique_model_genes_scored": int(len(gene_scores)),
        "seed_threshold": float(threshold),
        "threshold_override_used": bool(threshold_override is not None),
        "distribution_method": dist_info,
        "selected_seed_count": int(len(selected)),
        "selected_seed_genes": selected[["gene_id", "gene_symbol", "seed_score"]].to_dict(orient="records"),
        "score_quantiles": {
            str(k): float(v)
            for k, v in gene_scores["predicted_score"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict().items()
        },
    }
    return selected, report


def load_hgnc_symbol_to_ensembl_map(hgnc_path: Path) -> Dict[str, List[str]]:
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    if "symbol" not in hgnc.columns or "ensembl_gene_id" not in hgnc.columns:
        raise ValueError("HGNC file must contain 'symbol' and 'ensembl_gene_id' columns.")

    hgnc["symbol_norm"] = hgnc["symbol"].map(normalize_gene_symbol)
    hgnc["gene_id_norm"] = hgnc["ensembl_gene_id"].map(normalize_gene_id)
    hgnc = hgnc.dropna(subset=["symbol_norm", "gene_id_norm"]).copy()

    hgnc = hgnc.loc[hgnc["gene_id_norm"].str.match(ENSG_NO_VERSION_RE)].copy()

    grouped = hgnc.groupby("symbol_norm", as_index=False)["gene_id_norm"].agg(lambda s: sorted(set(s.tolist())))
    return dict(zip(grouped["symbol_norm"], grouped["gene_id_norm"]))


def select_l2g_seed_genes_from_existing_logic(
    ot_json_path: Path,
    hgnc_path: Path,
    source_column: str = "gwasCredibleSets",
    score_threshold: float = 0.5,
    seed_source_label: str = "clinvar",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    with open(ot_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    ot_df = pd.DataFrame(payload)
    if "symbol" not in ot_df.columns:
        raise ValueError(f"Open Targets JSON missing symbol column: {ot_json_path}")
    if source_column not in ot_df.columns:
        raise ValueError(f"Open Targets JSON missing requested source column '{source_column}': {ot_json_path}")

    ot_df["gene_symbol"] = ot_df["symbol"].map(normalize_gene_symbol)
    ot_df["source_score"] = pd.to_numeric(ot_df[source_column], errors="coerce")
    ot_df = ot_df.dropna(subset=["gene_symbol"]).copy()

    selected = ot_df.loc[ot_df["source_score"].fillna(0.0) >= float(score_threshold), ["gene_symbol", "source_score"]].copy()
    selected = selected.sort_values("source_score", ascending=False, kind="stable").reset_index(drop=True)

    symbol_to_ensg = load_hgnc_symbol_to_ensembl_map(hgnc_path)

    gene_ids: List[Optional[str]] = []
    hgnc_status: List[str] = []
    for sym in selected["gene_symbol"].tolist():
        ids = symbol_to_ensg.get(sym, [])
        if len(ids) == 0:
            gene_ids.append(None)
            hgnc_status.append("no_hgnc_ensembl_mapping")
        elif len(ids) == 1:
            gene_ids.append(ids[0])
            hgnc_status.append("unique_hgnc_symbol_mapping")
        else:
            gene_ids.append(ids[0])
            hgnc_status.append("ambiguous_hgnc_symbol_mapping_first_used")

    selected["gene_id"] = gene_ids
    selected["hgnc_mapping_status"] = hgnc_status
    selected["seed_source"] = str(seed_source_label)
    selected["seed_score"] = selected["source_score"].astype(float)
    selected["seed_rule"] = f"{source_column} >= {float(score_threshold):.2f}"

    with_id = selected.loc[selected["gene_id"].notna()].copy()
    with_id = with_id.sort_values("seed_score", ascending=False, kind="stable")
    with_id = with_id.drop_duplicates(subset=["gene_id"], keep="first")

    no_id = selected.loc[selected["gene_id"].isna()].copy()
    no_id = no_id.drop_duplicates(subset=["gene_symbol"], keep="first")

    final_selected = pd.concat([with_id, no_id], axis=0, ignore_index=True)
    final_selected = final_selected.sort_values("seed_score", ascending=False, kind="stable").reset_index(drop=True)

    report = {
        "ot_json_path": str(ot_json_path),
        "hgnc_path": str(hgnc_path),
        "source_column": source_column,
        "score_threshold": float(score_threshold),
        "ot_rows_total": int(len(ot_df)),
        "ot_rows_selected_before_dedup": int(len(selected)),
        "selected_seed_count_after_dedup": int(len(final_selected)),
        "selected_with_gene_id": int(final_selected["gene_id"].notna().sum()),
        "selected_without_gene_id": int(final_selected["gene_id"].isna().sum()),
        "mapping_status_counts": final_selected["hgnc_mapping_status"].value_counts(dropna=False).to_dict(),
        "selected_seed_genes": final_selected[["gene_id", "gene_symbol", "seed_score"]].to_dict(orient="records"),
    }
    return final_selected, report


def map_seed_genes_to_network(seed_df: pd.DataFrame, network_node_set: set[str], source_name: str) -> MappingResult:
    cur = seed_df.copy()
    cur["gene_id_normalized"] = cur["gene_id"].map(normalize_gene_id)

    cur["mapping_status"] = "unmapped"
    cur.loc[cur["gene_id_normalized"].isna(), "mapping_status"] = "missing_gene_id"
    cur.loc[cur["gene_id_normalized"].isin(network_node_set), "mapping_status"] = "mapped"

    cur["mapped_to_network"] = cur["mapping_status"] == "mapped"

    mapped_df = cur.loc[cur["mapped_to_network"]].copy()
    mapped_df = mapped_df.drop_duplicates(subset=["gene_id_normalized"], keep="first")

    unmapped_df = cur.loc[~cur["mapped_to_network"]].copy()

    mapped_examples = (
        mapped_df[["gene_id_normalized", "gene_symbol", "seed_score"]]
        .head(10)
        .rename(columns={"gene_id_normalized": "gene_id"})
        .to_dict(orient="records")
    )
    unmapped_examples = (
        unmapped_df[["gene_id", "gene_symbol", "mapping_status", "seed_score"]]
        .head(10)
        .to_dict(orient="records")
    )

    report = {
        "source_name": source_name,
        "requested_seed_rows": int(len(seed_df)),
        "requested_unique_seed_gene_ids": int(seed_df["gene_id"].dropna().map(normalize_gene_id).nunique()),
        "mapped_seed_count": int(len(mapped_df)),
        "unmapped_seed_count": int(len(unmapped_df)),
        "mapping_status_counts": cur["mapping_status"].value_counts(dropna=False).to_dict(),
        "mapped_examples": mapped_examples,
        "unmapped_examples": unmapped_examples,
    }
    return MappingResult(mapped_df=mapped_df, unmapped_df=unmapped_df, report=report)


# ---- Notebook-faithful PPR and normalization ----

def ppr_score_simple(
    g: nx.Graph,
    disease_genes: Sequence[str],
    genes_in_network: Iterable[str],
    use_alpha: float = 0.85,
    max_it: int = 500,
    cur_tol: float = 1e-9,
) -> pd.DataFrame:
    genes_set = set(genes_in_network)
    valid_seed_genes = [gene for gene in disease_genes if gene in genes_set]
    cur_dict = dict(zip(valid_seed_genes, [1 for _ in range(len(valid_seed_genes))]))
    if not cur_dict:
        raise ValueError("No valid seed genes in network for PPR.")

    weighted_personalized_pagerank = nx.pagerank(
        g,
        alpha=use_alpha,
        personalization=cur_dict,
        nstart=cur_dict,
        max_iter=max_it,
        tol=cur_tol,
    )

    ppr_scores = pd.DataFrame.from_dict(weighted_personalized_pagerank, orient="index")
    return ppr_scores


def run_ppr(
    graph: nx.Graph,
    seed_genes: Sequence[str],
    genes_in_network: Iterable[str],
    alpha: float = 0.85,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> pd.Series:
    ppr_df = ppr_score_simple(
        g=graph,
        disease_genes=seed_genes,
        genes_in_network=genes_in_network,
        use_alpha=alpha,
        max_it=max_iter,
        cur_tol=tol,
    )
    series = ppr_df.iloc[:, 0].copy()
    series.index = series.index.astype(str)
    series.name = "score"
    return series


def normalize_propagated_scores(
    propagated_scores: pd.Series,
    background_ppr_df: pd.DataFrame,
) -> pd.Series:
    if "score" not in background_ppr_df.columns:
        raise ValueError("Background PPR dataframe must contain a 'score' column.")

    ppr = propagated_scores.copy()
    ppr.index = ppr.index.astype(str)
    ppr = pd.to_numeric(ppr, errors="coerce")

    bg = pd.to_numeric(background_ppr_df["score"], errors="coerce")
    bg.index = bg.index.astype(str)

    ppr = ppr.reindex(bg.index)
    if ppr.isna().any():
        missing = int(ppr.isna().sum())
        raise ValueError(f"PPR scores missing for {missing} background genes during normalization.")

    if (ppr <= 0).any() or (bg <= 0).any():
        raise ValueError("Non-positive values found; log normalization requires strictly positive scores.")

    norm = np.log(ppr) - np.log(bg)
    norm.name = "score"
    return norm


def _top_overlap_metrics(series_a: pd.Series, series_b: pd.Series, top_n: int) -> Dict[str, object]:
    top_a = set(series_a.sort_values(ascending=False).head(top_n).index.tolist())
    top_b = set(series_b.sort_values(ascending=False).head(top_n).index.tolist())
    inter = top_a.intersection(top_b)
    union = top_a.union(top_b)
    return {
        "top_n": int(top_n),
        "overlap_count": int(len(inter)),
        "jaccard": float(len(inter) / len(union)) if union else 0.0,
        "overlap_genes_sample": sorted(list(inter))[:20],
    }


def compute_comparison_metrics(
    ppr_model: pd.Series,
    ppr_clinvar: pd.Series,
    norm_model: Optional[pd.Series] = None,
    norm_clinvar: Optional[pd.Series] = None,
) -> Dict[str, object]:
    shared = ppr_model.index.intersection(ppr_clinvar.index)
    raw_a = ppr_model.loc[shared].astype(float)
    raw_b = ppr_clinvar.loc[shared].astype(float)

    metrics: Dict[str, object] = {
        "shared_gene_count_raw": int(len(shared)),
        "raw_score_pearson": float(raw_a.corr(raw_b, method="pearson")),
        "raw_score_spearman": float(raw_a.corr(raw_b, method="spearman")),
        "raw_top_overlap_top50": _top_overlap_metrics(raw_a, raw_b, top_n=50),
        "raw_top_overlap_top100": _top_overlap_metrics(raw_a, raw_b, top_n=100),
    }

    if norm_model is not None and norm_clinvar is not None:
        shared_norm = norm_model.index.intersection(norm_clinvar.index)
        norm_a = norm_model.loc[shared_norm].astype(float)
        norm_b = norm_clinvar.loc[shared_norm].astype(float)
        metrics.update(
            {
                "shared_gene_count_normalized": int(len(shared_norm)),
                "normalized_score_pearson": float(norm_a.corr(norm_b, method="pearson")),
                "normalized_score_spearman": float(norm_a.corr(norm_b, method="spearman")),
                "normalized_top_overlap_top50": _top_overlap_metrics(norm_a, norm_b, top_n=50),
                "normalized_top_overlap_top100": _top_overlap_metrics(norm_a, norm_b, top_n=100),
            }
        )

    return metrics


def flatten_comparison_metrics(comparison_metrics: Dict[str, object]) -> Dict[str, float]:
    return {
        "raw_pearson": float(comparison_metrics.get("raw_score_pearson", np.nan)),
        "raw_spearman": float(comparison_metrics.get("raw_score_spearman", np.nan)),
        "raw_top50_overlap": float(comparison_metrics.get("raw_top_overlap_top50", {}).get("overlap_count", np.nan)),
        "raw_top100_overlap": float(comparison_metrics.get("raw_top_overlap_top100", {}).get("overlap_count", np.nan)),
        "norm_pearson": float(comparison_metrics.get("normalized_score_pearson", np.nan)),
        "norm_spearman": float(comparison_metrics.get("normalized_score_spearman", np.nan)),
        "norm_top50_overlap": float(
            comparison_metrics.get("normalized_top_overlap_top50", {}).get("overlap_count", np.nan)
        ),
        "norm_top100_overlap": float(
            comparison_metrics.get("normalized_top_overlap_top100", {}).get("overlap_count", np.nan)
        ),
    }


def _threshold_label(threshold: float) -> str:
    return f"{float(threshold):.2f}"


def _save_scored_series(series: pd.Series, out_path: Path) -> None:
    out = series.sort_values(ascending=False).reset_index()
    out.columns = ["gene_id", "score"]
    out.to_csv(out_path, index=False)


def _plot_threshold_curves(
    metrics_df: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, str]:
    # Keep threshold order high -> low to align with lowering-threshold interpretation.
    cur = metrics_df.sort_values("threshold", ascending=False).copy()
    x = cur["threshold"].astype(float).values

    corr_path = out_dir / "correlation_vs_threshold.png"
    plt.figure(figsize=(6, 4))
    plt.plot(x, cur["norm_spearman"].astype(float).values, marker="o", label="Normalized Spearman")
    plt.plot(x, cur["norm_pearson"].astype(float).values, marker="o", label="Normalized Pearson")
    plt.xlabel("Model seed threshold")
    plt.ylabel("Correlation")
    plt.title("Correlation vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(corr_path, dpi=150)
    plt.close()

    overlap_path = out_dir / "overlap_vs_threshold.png"
    plt.figure(figsize=(6, 4))
    plt.plot(x, cur["norm_top50_overlap"].astype(float).values, marker="o", label="Normalized top-50 overlap")
    plt.plot(x, cur["norm_top100_overlap"].astype(float).values, marker="o", label="Normalized top-100 overlap")
    plt.xlabel("Model seed threshold")
    plt.ylabel("Overlap count")
    plt.title("Overlap vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(overlap_path, dpi=150)
    plt.close()

    seeds_path = out_dir / "num_seeds_vs_threshold.png"
    plt.figure(figsize=(6, 4))
    plt.plot(x, cur["num_seeds"].astype(float).values, marker="o")
    plt.xlabel("Model seed threshold")
    plt.ylabel("Number of mapped model seeds")
    plt.title("Seed Count vs Threshold")
    plt.tight_layout()
    plt.savefig(seeds_path, dpi=150)
    plt.close()

    return {
        "correlation_vs_threshold": corr_path.name,
        "overlap_vs_threshold": overlap_path.name,
        "num_seeds_vs_threshold": seeds_path.name,
    }


def _metrics_df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "threshold",
        "num_seeds",
        "raw_pearson",
        "raw_spearman",
        "raw_top50_overlap",
        "raw_top100_overlap",
        "norm_pearson",
        "norm_spearman",
        "norm_top50_overlap",
        "norm_top100_overlap",
    ]
    table_df = df.loc[:, cols].copy()

    fmt_cols = [c for c in cols if c != "num_seeds"]
    for c in fmt_cols:
        table_df[c] = pd.to_numeric(table_df[c], errors="coerce").map(
            lambda v: "nan" if pd.isna(v) else f"{float(v):.4f}"
        )
    table_df["num_seeds"] = pd.to_numeric(table_df["num_seeds"], errors="coerce").fillna(0).astype(int).astype(str)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in table_df[cols].astype(str).values.tolist()]
    return "\n".join([header, sep] + rows)


def _write_threshold_sweep_report(
    out_dir: Path,
    metrics_df: pd.DataFrame,
    plot_files: Dict[str, str],
    model_run_report: Dict[str, object],
    clinvar_seed_report: Dict[str, object],
    clinvar_seed_map_report: Dict[str, object],
    thresholds: Sequence[float],
) -> None:
    cur = metrics_df.sort_values("threshold", ascending=False).reset_index(drop=True)

    spearman_values = cur["norm_spearman"].astype(float).tolist()
    threshold_values = cur["threshold"].astype(float).tolist()
    monotonic_increase_when_lowering = all(
        spearman_values[i + 1] >= spearman_values[i] for i in range(len(spearman_values) - 1)
    )

    seed_values = cur["num_seeds"].astype(float).tolist()
    seed_growth = float(seed_values[-1] - seed_values[0]) if seed_values else float("nan")
    seed_growth_fold = (
        float(seed_values[-1] / seed_values[0])
        if seed_values and seed_values[0] > 0
        else float("nan")
    )

    idx_best = int(cur["norm_spearman"].astype(float).idxmax())
    best_thr = float(cur.loc[idx_best, "threshold"])
    best_spear = float(cur.loc[idx_best, "norm_spearman"])

    low_thr = float(min(threshold_values)) if threshold_values else float("nan")
    high_thr = float(max(threshold_values)) if threshold_values else float("nan")
    low_val = float(cur.loc[cur["threshold"] == low_thr, "norm_spearman"].iloc[0]) if threshold_values else float("nan")
    high_val = (
        float(cur.loc[cur["threshold"] == high_thr, "norm_spearman"].iloc[0]) if threshold_values else float("nan")
    )

    lines = [
        "# INTACT PPR Threshold Sweep Report",
        "",
        "## Setup",
        f"- Model run: `{model_run_report['penalty']}/mode_{model_run_report['mode']}`",
        f"- Model predictions file: `{model_run_report['predictions_path']}`",
        f"- ClinVar logic: `source_column={clinvar_seed_report['source_column']}`, `threshold={clinvar_seed_report['score_threshold']}`",
        f"- Fixed ClinVar mapped seeds: `{clinvar_seed_map_report['mapped_seed_count']}`",
        f"- Thresholds tested: `{', '.join(_threshold_label(t) for t in thresholds)}`",
        "",
        "## Metrics Table",
        _metrics_df_to_markdown_table(cur),
        "",
        "## Plots",
        f"![Correlation vs threshold]({plot_files['correlation_vs_threshold']})",
        "",
        f"![Overlap vs threshold]({plot_files['overlap_vs_threshold']})",
        "",
        f"![Number of seeds vs threshold]({plot_files['num_seeds_vs_threshold']})",
        "",
        "## Interpretation",
        f"- Lowering threshold from `{high_thr:.2f}` to `{low_thr:.2f}` changes normalized Spearman from `{high_val:.4f}` to `{low_val:.4f}`.",
        (
            "- Normalized Spearman increases monotonically as threshold is lowered."
            if monotonic_increase_when_lowering
            else "- Normalized Spearman is not monotonic as threshold is lowered."
        ),
        (
            f"- Mapped model seeds grow from `{int(seed_values[0])}` to `{int(seed_values[-1])}` "
            f"(delta `{seed_growth:.0f}`, fold-change `{seed_growth_fold:.2f}`)." if seed_values else "- Seed growth could not be computed."
        ),
        (
            f"- Best normalized Spearman is `{best_spear:.4f}` at threshold `{best_thr:.2f}`."
            " This suggests an agreement tradeoff between strict and relaxed seed selection."
        ),
        "- Patterns are descriptive only; this sweep does not by itself establish causality.",
    ]
    _write_text(out_dir / "threshold_sweep_report.md", "\n".join(lines) + "\n")


def run_threshold_sweep(args: argparse.Namespace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    network_result = load_network(args.network_path)
    background_result = load_background_ppr(args.background_ppr_path, network_result.node_set)
    project_gene_table_path = choose_project_gene_table(args.project_gene_table, args.gene_table_dir)
    project_genes_result = load_project_genes(project_gene_table_path)
    project_mapping_result = map_project_genes_to_network(project_genes_result.unique_genes_df, network_result.node_set)

    model_run_report = resolve_model_run_from_args(args)

    clinvar_seed_df, clinvar_seed_report = select_l2g_seed_genes_from_existing_logic(
        ot_json_path=args.ot_json_path,
        hgnc_path=args.hgnc_path,
        source_column=args.clinvar_source_column,
        score_threshold=float(args.clinvar_threshold),
        seed_source_label="clinvar",
    )
    clinvar_seed_df.to_csv(out_dir / "clinvar_seed_genes.csv", index=False)
    clinvar_seed_map = map_seed_genes_to_network(clinvar_seed_df, network_result.node_set, source_name="clinvar")
    clinvar_seed_map.mapped_df.to_csv(out_dir / "clinvar_seed_genes_mapped.csv", index=False)
    clinvar_seed_map.unmapped_df.to_csv(out_dir / "clinvar_seed_genes_unmapped.csv", index=False)

    clinvar_seed_ids = clinvar_seed_map.mapped_df["gene_id_normalized"].dropna().astype(str).tolist()
    if len(clinvar_seed_ids) == 0:
        raise ValueError("No mapped ClinVar seeds available for threshold sweep.")

    ppr_clinvar = run_ppr(
        graph=network_result.graph,
        seed_genes=clinvar_seed_ids,
        genes_in_network=network_result.node_set,
        alpha=float(args.ppr_alpha),
        max_iter=int(args.ppr_max_iter),
        tol=float(args.ppr_tol),
    )
    ppr_clinvar_norm = normalize_propagated_scores(ppr_clinvar, background_result.background_df)
    _save_scored_series(ppr_clinvar, out_dir / "ppr_scores_clinvar_seeds.csv")
    _save_scored_series(ppr_clinvar_norm, out_dir / "ppr_scores_clinvar_seeds_normalized.csv")

    project_mapped_out = project_mapping_result.mapped_df[
        ["gene_id_raw", "gene_id_normalized", "gene_symbol", "mapped_to_network"]
    ].rename(columns={"gene_id_normalized": "network_gene_id"})
    project_unmapped_out = project_mapping_result.unmapped_df[
        ["gene_id_raw", "gene_id_normalized", "gene_symbol", "mapped_to_network"]
    ].rename(columns={"gene_id_normalized": "network_gene_id"})
    project_mapped_out.to_csv(out_dir / "project_genes_mapped.csv", index=False)
    project_unmapped_out.to_csv(out_dir / "project_genes_unmapped.csv", index=False)

    rows: List[Dict[str, object]] = []
    per_threshold_summary: List[Dict[str, object]] = []
    thresholds = [float(t) for t in args.sweep_thresholds]

    for threshold in thresholds:
        thr_label = _threshold_label(threshold)
        model_seed_df, model_seed_report = select_model_seed_genes(
            model_predictions_path=Path(model_run_report["predictions_path"]),
            threshold_override=threshold,
        )
        model_seed_df.to_csv(out_dir / f"model_seeds_thr_{thr_label}.csv", index=False)

        model_seed_map = map_seed_genes_to_network(model_seed_df, network_result.node_set, source_name="model")
        model_seed_map.mapped_df.to_csv(out_dir / f"model_seeds_thr_{thr_label}_mapped.csv", index=False)
        model_seed_map.unmapped_df.to_csv(out_dir / f"model_seeds_thr_{thr_label}_unmapped.csv", index=False)

        model_seed_ids = model_seed_map.mapped_df["gene_id_normalized"].dropna().astype(str).tolist()
        if len(model_seed_ids) == 0:
            comparison_metrics = {}
            model_norm = None
            model_raw = None
        else:
            model_raw = run_ppr(
                graph=network_result.graph,
                seed_genes=model_seed_ids,
                genes_in_network=network_result.node_set,
                alpha=float(args.ppr_alpha),
                max_iter=int(args.ppr_max_iter),
                tol=float(args.ppr_tol),
            )
            model_norm = normalize_propagated_scores(model_raw, background_result.background_df)

            _save_scored_series(model_raw, out_dir / f"ppr_scores_model_seeds_thr_{thr_label}.csv")
            _save_scored_series(model_norm, out_dir / f"ppr_scores_model_seeds_thr_{thr_label}_normalized.csv")

            comparison_metrics = compute_comparison_metrics(
                ppr_model=model_raw,
                ppr_clinvar=ppr_clinvar,
                norm_model=model_norm,
                norm_clinvar=ppr_clinvar_norm,
            )

        flat = flatten_comparison_metrics(comparison_metrics)
        rows.append(
            {
                "threshold": float(threshold),
                "num_seeds": int(len(model_seed_ids)),
                **flat,
            }
        )
        per_threshold_summary.append(
            {
                "threshold": float(threshold),
                "model_seed_selection": model_seed_report,
                "model_seed_mapping": model_seed_map.report,
                "comparison_metrics": comparison_metrics,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "threshold_sweep_metrics.csv", index=False)

    plot_files = _plot_threshold_curves(metrics_df=metrics_df, out_dir=out_dir)
    _write_threshold_sweep_report(
        out_dir=out_dir,
        metrics_df=metrics_df,
        plot_files=plot_files,
        model_run_report=model_run_report,
        clinvar_seed_report=clinvar_seed_report,
        clinvar_seed_map_report=clinvar_seed_map.report,
        thresholds=thresholds,
    )

    best_row = metrics_df.loc[metrics_df["norm_spearman"].idxmax()]
    summary = {
        "script": "intact_ppr_pipeline.py",
        "status": "completed_threshold_sweep",
        "inputs": {
            "network_path": str(args.network_path),
            "background_ppr_path": str(args.background_ppr_path),
            "project_gene_table_path": str(project_gene_table_path),
            "comparison_root": str(args.comparison_root),
            "model_predictions_path": str(args.model_predictions_path) if args.model_predictions_path is not None else None,
            "ot_json_path": str(args.ot_json_path),
            "hgnc_path": str(args.hgnc_path),
        },
        "model_run_selection": model_run_report,
        "clinvar_seed_selection": clinvar_seed_report,
        "clinvar_seed_mapping": clinvar_seed_map.report,
        "ppr_parameters": {
            "alpha": float(args.ppr_alpha),
            "max_iter": int(args.ppr_max_iter),
            "tol": float(args.ppr_tol),
        },
        "thresholds_tested": thresholds,
        "best_norm_spearman_threshold": float(best_row["threshold"]),
        "best_norm_spearman_value": float(best_row["norm_spearman"]),
        "threshold_sweep_metrics": rows,
        "per_threshold_details": per_threshold_summary,
        "outputs": {
            "out_dir": str(out_dir),
            "metrics_csv": str(out_dir / "threshold_sweep_metrics.csv"),
            "report_md": str(out_dir / "threshold_sweep_report.md"),
            "plot_correlation": str(out_dir / plot_files["correlation_vs_threshold"]),
            "plot_overlap": str(out_dir / plot_files["overlap_vs_threshold"]),
            "plot_num_seeds": str(out_dir / plot_files["num_seeds_vs_threshold"]),
        },
        "network_report": network_result.report,
        "background_report": background_result.report,
        "project_gene_source_report": project_genes_result.report,
        "project_gene_mapping_report": project_mapping_result.report,
    }

    (out_dir / "threshold_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("INTACT PPR threshold-sweep pipeline completed.")
    print(f"Output directory: {out_dir}")
    print(f"Thresholds tested: {', '.join(_threshold_label(t) for t in thresholds)}")
    print(
        "Best normalized Spearman: "
        f"{float(best_row['norm_spearman']):.6f} at threshold {float(best_row['threshold']):.2f}"
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_markdown_reports(
    out_dir: Path,
    network_report: Dict[str, object],
    background_report: Dict[str, object],
    project_report: Dict[str, object],
    mapping_report: Dict[str, object],
    model_run_report: Dict[str, object],
    model_seed_report: Dict[str, object],
    clinvar_seed_report: Dict[str, object],
    model_seed_map_report: Dict[str, object],
    clinvar_seed_map_report: Dict[str, object],
    comparison_metrics: Dict[str, object],
) -> None:
    lines = [
        "# INTACT PPR Model-vs-ClinVar Report",
        "",
        "## Network",
        f"- Nodes: `{network_report['num_nodes']}`",
        f"- Edges: `{network_report['num_edges']}`",
        f"- Identifier format: `{network_report['identifier_format']['identifier_guess']}`",
        "",
        "## Background PPR",
        f"- Raw rows: `{background_report['num_rows_raw']}`",
        f"- Indexed rows: `{background_report['num_rows_indexed']}`",
        f"- Exact node-set match to network: `{background_report['background_exactly_matches_network_nodes']}`",
        "",
        "## Project Gene Mapping",
        f"- Source table: `{project_report['project_gene_table_path']}`",
        f"- Unique project genes mapped: `{mapping_report['mapped_genes_count']}` / `{mapping_report['project_unique_genes_used_for_mapping']}`",
        f"- Unmapped: `{mapping_report['unmapped_genes_count']}`",
        "",
        "## Model Seeds",
        f"- Selected run penalty/mode: `{model_run_report['penalty']}` / `{model_run_report['mode']}`",
        f"- Predictions file: `{model_run_report['predictions_path']}`",
        f"- Seed threshold rule: `{model_seed_report['distribution_method']['method']}`",
        f"- Threshold used: `{model_seed_report['seed_threshold']}`",
        f"- Seed count: `{model_seed_report['selected_seed_count']}`",
        f"- Mapped to network: `{model_seed_map_report['mapped_seed_count']}`",
        f"- Unmapped: `{model_seed_map_report['unmapped_seed_count']}`",
        "",
        "## ClinVar Seeds",
        "- Existing project logic source files inspected:",
        "  - `src/training/build_cs_gene_candidate_feature_table.py` (API disease associations; default datasource=`eva`, threshold=0.5)",
        "  - `src/eval.py` and `src/eval_word2vec.py` (Open Targets disease-association JSON; threshold=0.5)",
        f"- Selected ClinVar source column: `{clinvar_seed_report['source_column']}`",
        f"- ClinVar threshold used: `{clinvar_seed_report['score_threshold']}`",
        f"- Seed count after dedup: `{clinvar_seed_report['selected_seed_count_after_dedup']}`",
        f"- Mapped to network: `{clinvar_seed_map_report['mapped_seed_count']}`",
        f"- Unmapped: `{clinvar_seed_map_report['unmapped_seed_count']}`",
        "",
        "## Comparison Metrics",
        f"- Raw Pearson: `{comparison_metrics['raw_score_pearson']}`",
        f"- Raw Spearman: `{comparison_metrics['raw_score_spearman']}`",
        f"- Raw top-50 overlap: `{comparison_metrics['raw_top_overlap_top50']['overlap_count']}`",
        f"- Raw top-100 overlap: `{comparison_metrics['raw_top_overlap_top100']['overlap_count']}`",
    ]

    if "normalized_score_pearson" in comparison_metrics:
        lines.extend(
            [
                f"- Normalized Pearson: `{comparison_metrics['normalized_score_pearson']}`",
                f"- Normalized Spearman: `{comparison_metrics['normalized_score_spearman']}`",
                f"- Normalized top-50 overlap: `{comparison_metrics['normalized_top_overlap_top50']['overlap_count']}`",
                f"- Normalized top-100 overlap: `{comparison_metrics['normalized_top_overlap_top100']['overlap_count']}`",
            ]
        )

    _write_text(out_dir / "pipeline_report.md", "\n".join(lines) + "\n")


def _load_score_series_from_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "gene_id" not in df.columns or "score" not in df.columns:
        raise ValueError(f"Score file must contain gene_id and score columns: {path}")
    cur = df[["gene_id", "score"]].copy()
    cur["gene_id"] = cur["gene_id"].map(normalize_gene_id)
    cur["score"] = pd.to_numeric(cur["score"], errors="coerce")
    cur = cur.dropna(subset=["gene_id", "score"]).drop_duplicates(subset=["gene_id"], keep="first")
    series = pd.Series(cur["score"].values, index=cur["gene_id"].astype(str).values)
    series.name = "score"
    return series


def _load_seed_ids_from_mapped_csv(path: Path) -> Tuple[Set[str], pd.DataFrame]:
    df = pd.read_csv(path)
    if "gene_id_normalized" in df.columns:
        id_col = "gene_id_normalized"
    elif "gene_id" in df.columns:
        id_col = "gene_id"
    else:
        raise ValueError(f"Mapped seed file must include gene_id_normalized or gene_id column: {path}")

    cur = df.copy()
    cur["gene_id_norm"] = cur[id_col].map(normalize_gene_id)
    cur = cur.dropna(subset=["gene_id_norm"]).drop_duplicates(subset=["gene_id_norm"], keep="first")
    return set(cur["gene_id_norm"].astype(str).tolist()), cur


def _load_ensembl_to_symbol_map(hgnc_path: Path) -> Dict[str, str]:
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    if "symbol" not in hgnc.columns or "ensembl_gene_id" not in hgnc.columns:
        return {}
    hgnc["symbol_norm"] = hgnc["symbol"].map(normalize_gene_symbol)
    hgnc["gene_id_norm"] = hgnc["ensembl_gene_id"].map(normalize_gene_id)
    hgnc = hgnc.dropna(subset=["symbol_norm", "gene_id_norm"]).copy()
    hgnc = hgnc.drop_duplicates(subset=["gene_id_norm"], keep="first")
    return dict(zip(hgnc["gene_id_norm"].astype(str), hgnc["symbol_norm"].astype(str)))


def _nearest_distance_records(
    graph: nx.Graph,
    source_nodes: Iterable[str],
    target_nodes: Iterable[str],
    direction: str,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    source_list = sorted(set(str(s) for s in source_nodes))
    target_set = set(str(t) for t in target_nodes if str(t) in graph)

    if target_set:
        lengths_to_target = nx.multi_source_dijkstra_path_length(graph, target_set, weight=None)
    else:
        lengths_to_target = {}

    records: List[Dict[str, object]] = []
    for source in source_list:
        if source not in graph:
            dist = float("inf")
            in_graph = False
        else:
            dist = float(lengths_to_target.get(source, float("inf")))
            in_graph = True
        records.append(
            {
                "direction": direction,
                "top_n": int(top_n) if top_n is not None else np.nan,
                "source_gene_id": source,
                "source_in_graph": bool(in_graph),
                "nearest_distance": dist,
                "finite_distance": bool(np.isfinite(dist)),
            }
        )
    return pd.DataFrame.from_records(records)


def _summarize_distance_records(
    distance_df: pd.DataFrame,
    direction: str,
    top_n: Optional[int] = None,
) -> Dict[str, object]:
    vals = pd.to_numeric(distance_df["nearest_distance"], errors="coerce")
    finite_vals = vals[np.isfinite(vals)]
    inf_count = int((~np.isfinite(vals)).sum())

    dist_counter = Counter(int(v) for v in finite_vals.tolist())
    distribution = {str(k): int(v) for k, v in sorted(dist_counter.items(), key=lambda x: x[0])}
    if inf_count > 0:
        distribution["inf"] = inf_count

    return {
        "direction": direction,
        "top_n": int(top_n) if top_n is not None else np.nan,
        "n_sources": int(len(distance_df)),
        "n_finite": int(len(finite_vals)),
        "n_infinite": int(inf_count),
        "mean_distance": float(finite_vals.mean()) if len(finite_vals) else np.nan,
        "median_distance": float(finite_vals.median()) if len(finite_vals) else np.nan,
        "min_distance": float(finite_vals.min()) if len(finite_vals) else np.nan,
        "max_distance": float(finite_vals.max()) if len(finite_vals) else np.nan,
        "distance_distribution": json.dumps(distribution, sort_keys=True),
    }


def _plot_distance_distribution(
    distance_df: pd.DataFrame,
    group_col: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7, 4))
    plotted = False
    cur = distance_df.copy()
    cur["nearest_distance"] = pd.to_numeric(cur["nearest_distance"], errors="coerce")

    for key, group in cur.groupby(group_col):
        vals = group.loc[np.isfinite(group["nearest_distance"]), "nearest_distance"].astype(float).values
        if len(vals) == 0:
            continue
        max_val = int(np.max(vals))
        bins = np.arange(-0.5, max_val + 1.5, 1)
        if len(bins) < 2:
            bins = np.array([-0.5, 0.5, 1.5])
        plt.hist(vals, bins=bins, alpha=0.45, label=str(key), edgecolor="white")
        plotted = True

    if plotted:
        plt.xlabel("Nearest shortest-path distance")
        plt.ylabel("Count")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    else:
        plt.text(0.5, 0.5, "No finite distances to plot", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_top_induced_subnetwork(
    subgraph: nx.Graph,
    model_top: Set[str],
    clinvar_top: Set[str],
    model_seeds: Set[str],
    clinvar_seeds: Set[str],
    model_scores: pd.Series,
    clinvar_scores: pd.Series,
    symbol_map: Dict[str, str],
    out_path: Path,
) -> None:
    if subgraph.number_of_nodes() == 0:
        plt.figure(figsize=(7, 5))
        plt.text(0.5, 0.5, "No nodes available for subnetwork plot", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    pos = nx.spring_layout(subgraph, seed=42, k=1.0 / np.sqrt(max(subgraph.number_of_nodes(), 1)))
    model_only = sorted(list(model_top - clinvar_top))
    clinvar_only = sorted(list(clinvar_top - model_top))
    shared = sorted(list(model_top.intersection(clinvar_top)))
    seed_only = sorted(list((model_seeds.union(clinvar_seeds)) - (model_top.union(clinvar_top))))

    seed_union = model_seeds.union(clinvar_seeds)
    model_only_nonseed = [n for n in model_only if n not in seed_union]
    clinvar_only_nonseed = [n for n in clinvar_only if n not in seed_union]
    shared_nonseed = [n for n in shared if n not in seed_union]
    seed_only_nonseed = [n for n in seed_only if n not in seed_union]

    model_only_seed = [n for n in model_only if n in seed_union]
    clinvar_only_seed = [n for n in clinvar_only if n in seed_union]
    shared_seed = [n for n in shared if n in seed_union]
    seed_only_seed = [n for n in seed_only if n in seed_union]

    color_map = {
        "model_only": "#1f77b4",
        "clinvar_only": "#ff7f0e",
        "shared": "#2ca02c",
        "seed_only": "#bdbdbd",
    }

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_edges(subgraph, pos=pos, width=0.8, alpha=0.25, edge_color="#999999")

    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=model_only_nonseed, node_size=150, node_color=color_map["model_only"], linewidths=0.3
    )
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=clinvar_only_nonseed, node_size=150, node_color=color_map["clinvar_only"], linewidths=0.3
    )
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=shared_nonseed, node_size=170, node_color=color_map["shared"], linewidths=0.3
    )
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=seed_only_nonseed, node_size=120, node_color=color_map["seed_only"], linewidths=0.3
    )

    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=model_only_seed,
        node_size=260,
        node_color=color_map["model_only"],
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=clinvar_only_seed,
        node_size=260,
        node_color=color_map["clinvar_only"],
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=shared_seed,
        node_size=280,
        node_color=color_map["shared"],
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=seed_only_seed,
        node_size=220,
        node_color=color_map["seed_only"],
        edgecolors="black",
        linewidths=1.0,
    )

    ranked_nodes = sorted(
        list(model_top.union(clinvar_top)),
        key=lambda g: max(float(model_scores.get(g, -np.inf)), float(clinvar_scores.get(g, -np.inf))),
        reverse=True,
    )
    label_nodes = ranked_nodes[:20]
    labels = {g: symbol_map.get(g, g) for g in label_nodes}
    nx.draw_networkx_labels(subgraph, pos=pos, labels=labels, font_size=7)

    legend_items = [
        Patch(facecolor=color_map["model_only"], edgecolor="none", label="Model-only top genes"),
        Patch(facecolor=color_map["clinvar_only"], edgecolor="none", label="ClinVar-only top genes"),
        Patch(facecolor=color_map["shared"], edgecolor="none", label="Shared top genes"),
        Patch(facecolor=color_map["seed_only"], edgecolor="none", label="Seed-only nodes"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666", markeredgecolor="black", markersize=8, label="Seed node"),
    ]
    plt.legend(handles=legend_items, loc="best", frameon=True, fontsize=8)
    plt.title("Top-Gene subnetwork (top-100 from each model + seeds)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_community_summary(community_summary_df: pd.DataFrame, out_path: Path) -> None:
    if community_summary_df.empty:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "No communities available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    cur = community_summary_df.copy()
    cur["model_only_count"] = cur["model_top_genes_count"] - cur["shared_top_genes_count"]
    cur["clinvar_only_count"] = cur["clinvar_top_genes_count"] - cur["shared_top_genes_count"]
    cur = cur.sort_values("module_size", ascending=False).reset_index(drop=True)

    x = np.arange(len(cur))
    plt.figure(figsize=(max(8, len(cur) * 0.55), 5))
    plt.bar(x, cur["model_only_count"], label="Model-only top genes", color="#1f77b4")
    plt.bar(
        x,
        cur["shared_top_genes_count"],
        bottom=cur["model_only_count"],
        label="Shared top genes",
        color="#2ca02c",
    )
    plt.bar(
        x,
        cur["clinvar_only_count"],
        bottom=cur["model_only_count"] + cur["shared_top_genes_count"],
        label="ClinVar-only top genes",
        color="#ff7f0e",
    )

    plt.xticks(x, cur["module_id"], rotation=0)
    plt.xlabel("Detected module")
    plt.ylabel("Top-gene count")
    plt.title("Community Membership Summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _markdown_table_from_df(df: pd.DataFrame, float_digits: int = 3) -> str:
    cur = df.copy()
    for col in cur.columns:
        if pd.api.types.is_float_dtype(cur[col]) or pd.api.types.is_integer_dtype(cur[col]):
            if pd.api.types.is_float_dtype(cur[col]):
                cur[col] = cur[col].map(lambda v: "nan" if pd.isna(v) else f"{float(v):.{float_digits}f}")
            else:
                cur[col] = cur[col].astype(str)
        else:
            cur[col] = cur[col].astype(str)
    header = "| " + " | ".join(cur.columns.tolist()) + " |"
    sep = "| " + " | ".join(["---"] * len(cur.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in cur.values.tolist()]
    return "\n".join([header, sep] + rows)


def run_network_neighborhood_analysis(args: argparse.Namespace) -> None:
    analysis_input_dir = args.analysis_input_dir if args.analysis_input_dir is not None else DEFAULT_OUT_DIR
    if not analysis_input_dir.exists():
        raise FileNotFoundError(f"Analysis input directory not found: {analysis_input_dir}")

    analysis_out_dir = (
        args.analysis_out_dir
        if args.analysis_out_dir is not None
        else analysis_input_dir / f"network_neighborhood_analysis_{date.today().isoformat()}"
    )
    analysis_out_dir.mkdir(parents=True, exist_ok=True)

    network_result = load_network(args.network_path)
    score_suffix = "_normalized" if args.analysis_score_type == "normalized" else ""
    model_score_path = analysis_input_dir / f"ppr_scores_model_seeds{score_suffix}.csv"
    clinvar_score_path = analysis_input_dir / f"ppr_scores_clinvar_seeds{score_suffix}.csv"
    if not model_score_path.exists() or not clinvar_score_path.exists():
        raise FileNotFoundError(
            "Could not find propagated score files for requested score type. "
            f"Expected: {model_score_path} and {clinvar_score_path}"
        )

    model_scores = _load_score_series_from_csv(model_score_path)
    clinvar_scores = _load_score_series_from_csv(clinvar_score_path)

    top_ns = sorted(set(int(x) for x in args.analysis_top_ns if int(x) > 0))
    if not top_ns:
        raise ValueError("analysis_top_ns must contain at least one positive integer.")

    model_top_sets: Dict[int, Set[str]] = {}
    clinvar_top_sets: Dict[int, Set[str]] = {}
    for top_n in top_ns:
        model_top_df = model_scores.sort_values(ascending=False).head(top_n).reset_index()
        model_top_df.columns = ["gene_id", "score"]
        model_top_df["rank"] = np.arange(1, len(model_top_df) + 1)
        model_top_df.to_csv(analysis_out_dir / f"top_model_genes_top{top_n}.csv", index=False)
        model_top_sets[top_n] = set(model_top_df["gene_id"].astype(str).tolist())

        clinvar_top_df = clinvar_scores.sort_values(ascending=False).head(top_n).reset_index()
        clinvar_top_df.columns = ["gene_id", "score"]
        clinvar_top_df["rank"] = np.arange(1, len(clinvar_top_df) + 1)
        clinvar_top_df.to_csv(analysis_out_dir / f"top_clinvar_genes_top{top_n}.csv", index=False)
        clinvar_top_sets[top_n] = set(clinvar_top_df["gene_id"].astype(str).tolist())

    model_seed_path = analysis_input_dir / "model_seed_genes_mapped.csv"
    clinvar_seed_path = analysis_input_dir / "clinvar_seed_genes_mapped.csv"
    model_seed_ids, model_seed_df = _load_seed_ids_from_mapped_csv(model_seed_path)
    clinvar_seed_ids, clinvar_seed_df = _load_seed_ids_from_mapped_csv(clinvar_seed_path)

    seed_model_to_clinvar = _nearest_distance_records(
        graph=network_result.graph,
        source_nodes=model_seed_ids,
        target_nodes=clinvar_seed_ids,
        direction="model_seed_to_nearest_clinvar_seed",
    )
    seed_clinvar_to_model = _nearest_distance_records(
        graph=network_result.graph,
        source_nodes=clinvar_seed_ids,
        target_nodes=model_seed_ids,
        direction="clinvar_seed_to_nearest_model_seed",
    )
    seed_distance_details = pd.concat([seed_model_to_clinvar, seed_clinvar_to_model], ignore_index=True)
    seed_distance_details.to_csv(analysis_out_dir / "seed_shortest_path_details.csv", index=False)

    seed_summary_rows = [
        _summarize_distance_records(seed_model_to_clinvar, "model_seed_to_nearest_clinvar_seed"),
        _summarize_distance_records(seed_clinvar_to_model, "clinvar_seed_to_nearest_model_seed"),
    ]
    seed_summary_df = pd.DataFrame(seed_summary_rows)
    seed_summary_df.to_csv(analysis_out_dir / "seed_shortest_path_summary.csv", index=False)

    top_distance_detail_frames: List[pd.DataFrame] = []
    top_summary_rows: List[Dict[str, object]] = []
    connected_component_rows: List[Dict[str, object]] = []
    connected_component_detail_rows: List[Dict[str, object]] = []

    for top_n in top_ns:
        model_top = model_top_sets[top_n]
        clinvar_top = clinvar_top_sets[top_n]
        shared_top = model_top.intersection(clinvar_top)

        d_model_to_clinvar = _nearest_distance_records(
            graph=network_result.graph,
            source_nodes=model_top,
            target_nodes=clinvar_top,
            direction=f"model_top{top_n}_to_nearest_clinvar_top{top_n}",
            top_n=top_n,
        )
        d_clinvar_to_model = _nearest_distance_records(
            graph=network_result.graph,
            source_nodes=clinvar_top,
            target_nodes=model_top,
            direction=f"clinvar_top{top_n}_to_nearest_model_top{top_n}",
            top_n=top_n,
        )
        top_distance_detail_frames.extend([d_model_to_clinvar, d_clinvar_to_model])

        top_summary_rows.append(_summarize_distance_records(d_model_to_clinvar, d_model_to_clinvar["direction"].iloc[0], top_n))
        top_summary_rows.append(_summarize_distance_records(d_clinvar_to_model, d_clinvar_to_model["direction"].iloc[0], top_n))

        m1 = pd.to_numeric(d_model_to_clinvar["nearest_distance"], errors="coerce")
        m2 = pd.to_numeric(d_clinvar_to_model["nearest_distance"], errors="coerce")
        bidir = pd.concat([m1[np.isfinite(m1)], m2[np.isfinite(m2)]], ignore_index=True)
        top_summary_rows.append(
            {
                "direction": f"bidirectional_top{top_n}",
                "top_n": int(top_n),
                "n_sources": int(len(model_top) + len(clinvar_top)),
                "n_finite": int(len(bidir)),
                "n_infinite": int((~np.isfinite(pd.concat([m1, m2], ignore_index=True))).sum()),
                "mean_distance": float(bidir.mean()) if len(bidir) else np.nan,
                "median_distance": float(bidir.median()) if len(bidir) else np.nan,
                "min_distance": float(bidir.min()) if len(bidir) else np.nan,
                "max_distance": float(bidir.max()) if len(bidir) else np.nan,
                "distance_distribution": np.nan,
                "set_overlap_count": int(len(shared_top)),
                "set_jaccard": float(len(shared_top) / len(model_top.union(clinvar_top))) if model_top.union(clinvar_top) else np.nan,
            }
        )

        union_top = model_top.union(clinvar_top)
        top_subgraph = network_result.graph.subgraph(union_top).copy()
        components = sorted(nx.connected_components(top_subgraph), key=len, reverse=True)
        component_map: Dict[str, int] = {}
        for cid, comp in enumerate(components, start=1):
            for node in comp:
                component_map[str(node)] = cid

        largest_comp = components[0] if components else set()
        connected_component_rows.append(
            {
                "top_n": int(top_n),
                "union_size": int(len(union_top)),
                "model_top_count": int(len(model_top)),
                "clinvar_top_count": int(len(clinvar_top)),
                "shared_top_count": int(len(shared_top)),
                "induced_edges": int(top_subgraph.number_of_edges()),
                "num_components": int(len(components)),
                "largest_component_size": int(len(largest_comp)),
                "model_in_largest_component": int(len(model_top.intersection(largest_comp))),
                "clinvar_in_largest_component": int(len(clinvar_top.intersection(largest_comp))),
                "shared_in_largest_component": int(len(shared_top.intersection(largest_comp))),
                "model_fraction_in_largest_component": (
                    float(len(model_top.intersection(largest_comp)) / len(model_top)) if len(model_top) else np.nan
                ),
                "clinvar_fraction_in_largest_component": (
                    float(len(clinvar_top.intersection(largest_comp)) / len(clinvar_top)) if len(clinvar_top) else np.nan
                ),
            }
        )

        for cid, comp in enumerate(components, start=1):
            comp = set(str(x) for x in comp)
            comp_shared = sorted(list(comp.intersection(shared_top)))
            connected_component_detail_rows.append(
                {
                    "top_n": int(top_n),
                    "component_id": int(cid),
                    "component_size": int(len(comp)),
                    "model_top_genes_count": int(len(comp.intersection(model_top))),
                    "clinvar_top_genes_count": int(len(comp.intersection(clinvar_top))),
                    "shared_top_genes_count": int(len(comp_shared)),
                    "shared_top_genes_sample": ";".join(comp_shared[:20]),
                }
            )

    top_distance_details = pd.concat(top_distance_detail_frames, ignore_index=True)
    top_distance_details.to_csv(analysis_out_dir / "top_gene_shortest_path_details.csv", index=False)
    top_summary_df = pd.DataFrame(top_summary_rows)
    top_summary_df.to_csv(analysis_out_dir / "top_gene_shortest_path_summary.csv", index=False)

    connected_component_summary_df = pd.DataFrame(connected_component_rows)
    connected_component_summary_df.to_csv(analysis_out_dir / "top_gene_connected_component_summary.csv", index=False)
    connected_component_detail_df = pd.DataFrame(connected_component_detail_rows)
    connected_component_detail_df.to_csv(analysis_out_dir / "top_gene_connected_component_detailed.csv", index=False)

    module_top_n = max(top_ns)
    model_top_module = model_top_sets[module_top_n]
    clinvar_top_module = clinvar_top_sets[module_top_n]
    shared_top_module = model_top_module.intersection(clinvar_top_module)
    community_nodes = model_top_module.union(clinvar_top_module).union(model_seed_ids).union(clinvar_seed_ids)
    community_subgraph = network_result.graph.subgraph(community_nodes).copy()

    if community_subgraph.number_of_edges() > 0:
        communities = list(nx.algorithms.community.greedy_modularity_communities(community_subgraph))
    else:
        communities = [set([node]) for node in community_subgraph.nodes()]
    communities = sorted([set(str(g) for g in c) for c in communities], key=len, reverse=True)

    symbol_map = _load_ensembl_to_symbol_map(args.hgnc_path)
    community_summary_rows: List[Dict[str, object]] = []
    community_detail_rows: List[Dict[str, object]] = []
    for idx, comm in enumerate(communities, start=1):
        module_id = f"M{idx}"
        model_count = len(comm.intersection(model_top_module))
        clinvar_count = len(comm.intersection(clinvar_top_module))
        shared_count = len(comm.intersection(shared_top_module))
        model_seed_count = len(comm.intersection(model_seed_ids))
        clinvar_seed_count = len(comm.intersection(clinvar_seed_ids))

        if model_count > 0 and clinvar_count > 0:
            module_type = "mixed_model_clinvar"
        elif model_count > 0:
            module_type = "model_only_top"
        elif clinvar_count > 0:
            module_type = "clinvar_only_top"
        else:
            module_type = "seed_only_or_other"

        community_summary_rows.append(
            {
                "module_id": module_id,
                "module_size": int(len(comm)),
                "model_top_genes_count": int(model_count),
                "clinvar_top_genes_count": int(clinvar_count),
                "shared_top_genes_count": int(shared_count),
                "model_seed_count": int(model_seed_count),
                "clinvar_seed_count": int(clinvar_seed_count),
                "module_type": module_type,
                "shared_genes_sample": ";".join(sorted(list(comm.intersection(shared_top_module)))[:20]),
            }
        )

        for gene_id in sorted(comm):
            in_model_top = gene_id in model_top_module
            in_clinvar_top = gene_id in clinvar_top_module
            in_model_seed = gene_id in model_seed_ids
            in_clinvar_seed = gene_id in clinvar_seed_ids

            if in_model_top and in_clinvar_top:
                node_category = "shared_top"
            elif in_model_top:
                node_category = "model_only_top"
            elif in_clinvar_top:
                node_category = "clinvar_only_top"
            elif in_model_seed or in_clinvar_seed:
                node_category = "seed_only"
            else:
                node_category = "other"

            community_detail_rows.append(
                {
                    "module_id": module_id,
                    "gene_id": gene_id,
                    "gene_symbol": symbol_map.get(gene_id, ""),
                    "node_category": node_category,
                    "is_model_top": int(in_model_top),
                    "is_clinvar_top": int(in_clinvar_top),
                    "is_model_seed": int(in_model_seed),
                    "is_clinvar_seed": int(in_clinvar_seed),
                    "model_score": float(model_scores.get(gene_id, np.nan)),
                    "clinvar_score": float(clinvar_scores.get(gene_id, np.nan)),
                }
            )

    community_summary_df = pd.DataFrame(community_summary_rows)
    community_detail_df = pd.DataFrame(community_detail_rows)
    community_summary_df.to_csv(analysis_out_dir / "community_membership_summary.csv", index=False)
    community_detail_df.to_csv(analysis_out_dir / "community_membership_detailed.csv", index=False)

    _plot_distance_distribution(
        distance_df=seed_distance_details,
        group_col="direction",
        title="Seed-to-Seed Nearest Shortest-Path Distances",
        out_path=analysis_out_dir / "seed_shortest_path_distribution.png",
    )

    top_dist_plot_df = top_distance_details.copy()
    top_dist_plot_df["top_group"] = top_dist_plot_df["top_n"].map(lambda x: f"top{int(x)}")
    _plot_distance_distribution(
        distance_df=top_dist_plot_df,
        group_col="top_group",
        title="Top-Gene Nearest Shortest-Path Distances",
        out_path=analysis_out_dir / "top_gene_shortest_path_distribution.png",
    )

    _plot_top_induced_subnetwork(
        subgraph=community_subgraph,
        model_top=model_top_module,
        clinvar_top=clinvar_top_module,
        model_seeds=model_seed_ids,
        clinvar_seeds=clinvar_seed_ids,
        model_scores=model_scores,
        clinvar_scores=clinvar_scores,
        symbol_map=symbol_map,
        out_path=analysis_out_dir / "top_gene_induced_subnetwork.png",
    )

    _plot_community_summary(
        community_summary_df=community_summary_df,
        out_path=analysis_out_dir / "community_summary.png",
    )

    seed_mean_model_to_clinvar = float(seed_summary_df.loc[seed_summary_df["direction"] == "model_seed_to_nearest_clinvar_seed", "mean_distance"].iloc[0])
    seed_mean_clinvar_to_model = float(seed_summary_df.loc[seed_summary_df["direction"] == "clinvar_seed_to_nearest_model_seed", "mean_distance"].iloc[0])
    best_top_row = top_summary_df.loc[top_summary_df["direction"] == f"bidirectional_top{module_top_n}"].iloc[0]
    mixed_module_count = int((community_summary_df["module_type"] == "mixed_model_clinvar").sum())

    report_lines = [
        "# Network Neighborhood / Module-Level Analysis Report",
        "",
        "## Inputs",
        f"- Input directory: `{analysis_input_dir}`",
        f"- Graph file: `{args.network_path}`",
        f"- Propagated score type used: `{args.analysis_score_type}` (preferred for comparison)",
        f"- Top-N sets analyzed: `{', '.join(str(n) for n in top_ns)}`",
        "",
        "## Seed Distance Summary",
        _markdown_table_from_df(seed_summary_df[["direction", "n_sources", "mean_distance", "median_distance", "min_distance", "max_distance", "distance_distribution"]]),
        "",
        "## Top-Gene Distance Summary",
        _markdown_table_from_df(
            top_summary_df[
                [
                    "direction",
                    "top_n",
                    "n_sources",
                    "mean_distance",
                    "median_distance",
                    "min_distance",
                    "max_distance",
                    "set_overlap_count",
                    "set_jaccard",
                ]
            ].fillna(np.nan)
        ),
        "",
        "## Connected Component Summary",
        _markdown_table_from_df(
            connected_component_summary_df[
                [
                    "top_n",
                    "union_size",
                    "num_components",
                    "largest_component_size",
                    "model_fraction_in_largest_component",
                    "clinvar_fraction_in_largest_component",
                ]
            ]
        ),
        "",
        "## Community / Module Summary",
        f"- Communities detected (Top-{module_top_n} union + seeds): `{len(community_summary_df)}`",
        f"- Mixed model+ClinVar modules: `{mixed_module_count}`",
        _markdown_table_from_df(
            community_summary_df[
                [
                    "module_id",
                    "module_size",
                    "model_top_genes_count",
                    "clinvar_top_genes_count",
                    "shared_top_genes_count",
                    "model_seed_count",
                    "clinvar_seed_count",
                    "module_type",
                ]
            ]
        ),
        "",
        "## Plots",
        "![Seed shortest-path distance distribution](seed_shortest_path_distribution.png)",
        "",
        "![Top-gene shortest-path distance distribution](top_gene_shortest_path_distribution.png)",
        "",
        "![Top-gene induced subnetwork](top_gene_induced_subnetwork.png)",
        "",
        "![Community summary](community_summary.png)",
        "",
        "## Interpretation",
        f"- Mean nearest seed distance (model->ClinVar): `{seed_mean_model_to_clinvar:.3f}`; (ClinVar->model): `{seed_mean_clinvar_to_model:.3f}`.",
        f"- For Top-{module_top_n}, bidirectional mean nearest distance is `{float(best_top_row['mean_distance']):.3f}` with overlap `{int(best_top_row.get('set_overlap_count', 0))}`.",
        f"- Largest-component coverage remains high if model/ClinVar fractions are close to 1.0; see component table for exact values.",
        f"- Presence of `{mixed_module_count}` mixed modules indicates local convergence between methods, while method-dominant modules indicate distinct neighborhoods.",
        "- Findings are descriptive graph-structure comparisons and do not by themselves imply biological causality.",
    ]
    _write_text(analysis_out_dir / "network_neighborhood_analysis_report.md", "\n".join(report_lines) + "\n")

    summary = {
        "script": "intact_ppr_pipeline.py",
        "status": "completed_network_neighborhood_analysis",
        "analysis_input_dir": str(analysis_input_dir),
        "analysis_out_dir": str(analysis_out_dir),
        "score_type": args.analysis_score_type,
        "top_ns": top_ns,
        "seed_summary": seed_summary_rows,
        "top_gene_summary": top_summary_rows,
        "connected_component_summary": connected_component_rows,
        "community_summary": community_summary_rows,
        "outputs": {
            "seed_shortest_path_summary": str(analysis_out_dir / "seed_shortest_path_summary.csv"),
            "top_gene_shortest_path_summary": str(analysis_out_dir / "top_gene_shortest_path_summary.csv"),
            "community_membership_summary": str(analysis_out_dir / "community_membership_summary.csv"),
            "community_membership_detailed": str(analysis_out_dir / "community_membership_detailed.csv"),
            "report_md": str(analysis_out_dir / "network_neighborhood_analysis_report.md"),
            "seed_shortest_path_distribution_png": str(analysis_out_dir / "seed_shortest_path_distribution.png"),
            "top_gene_shortest_path_distribution_png": str(analysis_out_dir / "top_gene_shortest_path_distribution.png"),
            "top_gene_induced_subnetwork_png": str(analysis_out_dir / "top_gene_induced_subnetwork.png"),
            "community_summary_png": str(analysis_out_dir / "community_summary.png"),
        },
    }
    (analysis_out_dir / "network_neighborhood_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("INTACT PPR network-neighborhood analysis completed.")
    print(f"Input directory: {analysis_input_dir}")
    print(f"Output directory: {analysis_out_dir}")
    print(f"Score type used: {args.analysis_score_type}")
    print(f"Top-N sets analyzed: {', '.join(str(n) for n in top_ns)}")


def _pairwise_metrics_row(source_a: str, source_b: str, comparison_metrics: Dict[str, object]) -> Dict[str, object]:
    return {
        "source_a": source_a,
        "source_b": source_b,
        "pair": f"{source_a}_vs_{source_b}",
        "raw_pearson": float(comparison_metrics.get("raw_score_pearson", np.nan)),
        "raw_spearman": float(comparison_metrics.get("raw_score_spearman", np.nan)),
        "raw_top50_overlap": int(comparison_metrics.get("raw_top_overlap_top50", {}).get("overlap_count", 0)),
        "raw_top100_overlap": int(comparison_metrics.get("raw_top_overlap_top100", {}).get("overlap_count", 0)),
        "norm_pearson": float(comparison_metrics.get("normalized_score_pearson", np.nan)),
        "norm_spearman": float(comparison_metrics.get("normalized_score_spearman", np.nan)),
        "norm_top50_overlap": int(comparison_metrics.get("normalized_top_overlap_top50", {}).get("overlap_count", 0)),
        "norm_top100_overlap": int(comparison_metrics.get("normalized_top_overlap_top100", {}).get("overlap_count", 0)),
    }


def _three_way_node_category(gene_id: str, model_top: Set[str], clinvar_top: Set[str], gwas_top: Set[str], seed_union: Set[str]) -> str:
    in_m = gene_id in model_top
    in_c = gene_id in clinvar_top
    in_g = gene_id in gwas_top

    if in_m and in_c and in_g:
        return "all_three_shared"
    if in_m and in_c:
        return "model_clinvar_shared"
    if in_m and in_g:
        return "model_gwas_shared"
    if in_c and in_g:
        return "clinvar_gwas_shared"
    if in_m:
        return "model_only"
    if in_c:
        return "clinvar_only"
    if in_g:
        return "gwas_only"
    if gene_id in seed_union:
        return "seed_only"
    return "other"


def _plot_three_way_induced_subnetwork(
    subgraph: nx.Graph,
    model_top: Set[str],
    clinvar_top: Set[str],
    gwas_top: Set[str],
    model_seeds: Set[str],
    clinvar_seeds: Set[str],
    gwas_seeds: Set[str],
    model_scores: pd.Series,
    clinvar_scores: pd.Series,
    gwas_scores: pd.Series,
    symbol_map: Dict[str, str],
    out_path: Path,
) -> None:
    if subgraph.number_of_nodes() == 0:
        plt.figure(figsize=(7, 5))
        plt.text(0.5, 0.5, "No nodes available for three-way subnetwork plot", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    seed_union = model_seeds.union(clinvar_seeds).union(gwas_seeds)
    node_categories = {
        n: _three_way_node_category(str(n), model_top, clinvar_top, gwas_top, seed_union) for n in subgraph.nodes()
    }

    category_colors = {
        "model_only": "#1f77b4",
        "clinvar_only": "#ff7f0e",
        "gwas_only": "#d62728",
        "model_clinvar_shared": "#2ca02c",
        "model_gwas_shared": "#17becf",
        "clinvar_gwas_shared": "#bcbd22",
        "all_three_shared": "#8c564b",
        "seed_only": "#bdbdbd",
    }

    pos = nx.spring_layout(subgraph, seed=42, k=1.0 / np.sqrt(max(subgraph.number_of_nodes(), 1)))
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(subgraph, pos=pos, width=0.8, alpha=0.25, edge_color="#999999")

    for cat, color in category_colors.items():
        nodes = [n for n, c in node_categories.items() if c == cat and n not in seed_union]
        if not nodes:
            continue
        nx.draw_networkx_nodes(subgraph, pos, nodelist=nodes, node_size=145, node_color=color, linewidths=0.3)

    for cat, color in category_colors.items():
        nodes = [n for n, c in node_categories.items() if c == cat and n in seed_union]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            nodelist=nodes,
            node_size=240,
            node_color=color,
            edgecolors="black",
            linewidths=1.0,
        )

    union_top = model_top.union(clinvar_top).union(gwas_top)
    ranked_nodes = sorted(
        list(union_top),
        key=lambda g: max(
            float(model_scores.get(g, -np.inf)),
            float(clinvar_scores.get(g, -np.inf)),
            float(gwas_scores.get(g, -np.inf)),
        ),
        reverse=True,
    )
    label_nodes = ranked_nodes[:24]
    labels = {g: symbol_map.get(g, g) for g in label_nodes}
    nx.draw_networkx_labels(subgraph, pos=pos, labels=labels, font_size=7)

    legend_items = [
        Patch(facecolor=category_colors["model_only"], edgecolor="none", label="PCA-only top"),
        Patch(facecolor=category_colors["clinvar_only"], edgecolor="none", label="ClinVar-only top"),
        Patch(facecolor=category_colors["gwas_only"], edgecolor="none", label="GWAS-only top"),
        Patch(facecolor=category_colors["model_clinvar_shared"], edgecolor="none", label="PCA+ClinVar"),
        Patch(facecolor=category_colors["model_gwas_shared"], edgecolor="none", label="PCA+GWAS"),
        Patch(facecolor=category_colors["clinvar_gwas_shared"], edgecolor="none", label="ClinVar+GWAS"),
        Patch(facecolor=category_colors["all_three_shared"], edgecolor="none", label="Shared by all three"),
        Patch(facecolor=category_colors["seed_only"], edgecolor="none", label="Seed-only"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777", markeredgecolor="black", markersize=8, label="Seed node"),
    ]
    plt.legend(handles=legend_items, loc="best", frameon=True, fontsize=8)
    plt.title("Three-way top-gene subnetwork (top-100 each + seeds)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_three_way_community_summary(community_summary_df: pd.DataFrame, out_path: Path) -> None:
    if community_summary_df.empty:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "No communities available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    cur = community_summary_df.sort_values("module_size", ascending=False).reset_index(drop=True).copy()
    x = np.arange(len(cur))
    width = 0.25
    plt.figure(figsize=(max(8, len(cur) * 0.55), 5))
    plt.bar(x - width, cur["model_top_genes_count"], width=width, label="PCA top genes", color="#1f77b4")
    plt.bar(x, cur["clinvar_top_genes_count"], width=width, label="ClinVar top genes", color="#ff7f0e")
    plt.bar(x + width, cur["gwas_top_genes_count"], width=width, label="GWAS top genes", color="#d62728")
    plt.xticks(x, cur["module_id"], rotation=0)
    plt.xlabel("Detected module")
    plt.ylabel("Top-gene count")
    plt.title("Three-way community summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_three_way_pipeline(args: argparse.Namespace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    network_result = load_network(args.network_path)
    background_result = load_background_ppr(args.background_ppr_path, network_result.node_set)
    project_gene_table_path = choose_project_gene_table(args.project_gene_table, args.gene_table_dir)
    project_genes_result = load_project_genes(project_gene_table_path)
    project_mapping_result = map_project_genes_to_network(project_genes_result.unique_genes_df, network_result.node_set)

    model_run_report = resolve_model_run_from_args(args)
    model_seed_df, model_seed_report = select_model_seed_genes(
        model_predictions_path=Path(model_run_report["predictions_path"]),
        threshold_override=args.model_seed_threshold,
    )

    clinvar_seed_df, clinvar_seed_report = select_l2g_seed_genes_from_existing_logic(
        ot_json_path=args.ot_json_path,
        hgnc_path=args.hgnc_path,
        source_column=args.clinvar_source_column,
        score_threshold=float(args.clinvar_threshold),
        seed_source_label="clinvar",
    )
    gwas_seed_df, gwas_seed_report = select_l2g_seed_genes_from_existing_logic(
        ot_json_path=args.ot_json_path,
        hgnc_path=args.hgnc_path,
        source_column=args.gwas_source_column,
        score_threshold=float(args.gwas_threshold),
        seed_source_label="gwas",
    )

    model_seed_map = map_seed_genes_to_network(model_seed_df, network_result.node_set, source_name="model")
    clinvar_seed_map = map_seed_genes_to_network(clinvar_seed_df, network_result.node_set, source_name="clinvar")
    gwas_seed_map = map_seed_genes_to_network(gwas_seed_df, network_result.node_set, source_name="gwas")

    model_seed_df.to_csv(out_dir / "model_seed_genes.csv", index=False)
    clinvar_seed_df.to_csv(out_dir / "clinvar_seed_genes.csv", index=False)
    gwas_seed_df.to_csv(out_dir / "gwas_seed_genes.csv", index=False)

    model_seed_map.mapped_df.to_csv(out_dir / "model_seed_genes_mapped.csv", index=False)
    clinvar_seed_map.mapped_df.to_csv(out_dir / "clinvar_seed_genes_mapped.csv", index=False)
    gwas_seed_map.mapped_df.to_csv(out_dir / "gwas_seed_genes_mapped.csv", index=False)
    model_seed_map.unmapped_df.to_csv(out_dir / "model_seed_genes_unmapped.csv", index=False)
    clinvar_seed_map.unmapped_df.to_csv(out_dir / "clinvar_seed_genes_unmapped.csv", index=False)
    gwas_seed_map.unmapped_df.to_csv(out_dir / "gwas_seed_genes_unmapped.csv", index=False)

    model_seed_ids = model_seed_map.mapped_df["gene_id_normalized"].dropna().astype(str).tolist()
    clinvar_seed_ids = clinvar_seed_map.mapped_df["gene_id_normalized"].dropna().astype(str).tolist()
    gwas_seed_ids = gwas_seed_map.mapped_df["gene_id_normalized"].dropna().astype(str).tolist()

    if len(model_seed_ids) == 0:
        raise ValueError("No mapped model seeds available for PPR.")
    if len(clinvar_seed_ids) == 0:
        raise ValueError("No mapped ClinVar seeds available for PPR.")
    if len(gwas_seed_ids) == 0:
        raise ValueError("No mapped GWAS seeds available for PPR.")

    ppr_model = run_ppr(
        graph=network_result.graph,
        seed_genes=model_seed_ids,
        genes_in_network=network_result.node_set,
        alpha=float(args.ppr_alpha),
        max_iter=int(args.ppr_max_iter),
        tol=float(args.ppr_tol),
    )
    ppr_clinvar = run_ppr(
        graph=network_result.graph,
        seed_genes=clinvar_seed_ids,
        genes_in_network=network_result.node_set,
        alpha=float(args.ppr_alpha),
        max_iter=int(args.ppr_max_iter),
        tol=float(args.ppr_tol),
    )
    ppr_gwas = run_ppr(
        graph=network_result.graph,
        seed_genes=gwas_seed_ids,
        genes_in_network=network_result.node_set,
        alpha=float(args.ppr_alpha),
        max_iter=int(args.ppr_max_iter),
        tol=float(args.ppr_tol),
    )

    _save_scored_series(ppr_model, out_dir / "ppr_scores_model_seeds.csv")
    _save_scored_series(ppr_clinvar, out_dir / "ppr_scores_clinvar_seeds.csv")
    _save_scored_series(ppr_gwas, out_dir / "ppr_scores_gwas_seeds.csv")

    ppr_model_norm = normalize_propagated_scores(ppr_model, background_result.background_df)
    ppr_clinvar_norm = normalize_propagated_scores(ppr_clinvar, background_result.background_df)
    ppr_gwas_norm = normalize_propagated_scores(ppr_gwas, background_result.background_df)

    _save_scored_series(ppr_model_norm, out_dir / "ppr_scores_model_seeds_normalized.csv")
    _save_scored_series(ppr_clinvar_norm, out_dir / "ppr_scores_clinvar_seeds_normalized.csv")
    _save_scored_series(ppr_gwas_norm, out_dir / "ppr_scores_gwas_seeds_normalized.csv")

    score_series_raw = {"model": ppr_model, "clinvar": ppr_clinvar, "gwas": ppr_gwas}
    score_series_norm = {"model": ppr_model_norm, "clinvar": ppr_clinvar_norm, "gwas": ppr_gwas_norm}
    pair_list = [("model", "clinvar"), ("model", "gwas"), ("clinvar", "gwas")]

    pairwise_metrics_json: Dict[str, Dict[str, object]] = {}
    pairwise_rows: List[Dict[str, object]] = []
    for a, b in pair_list:
        m = compute_comparison_metrics(
            ppr_model=score_series_raw[a],
            ppr_clinvar=score_series_raw[b],
            norm_model=score_series_norm[a],
            norm_clinvar=score_series_norm[b],
        )
        pairwise_metrics_json[f"{a}_vs_{b}"] = m
        pairwise_rows.append(_pairwise_metrics_row(a, b, m))

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(out_dir / "pairwise_comparison_metrics.csv", index=False)
    (out_dir / "pairwise_comparison_metrics.json").write_text(json.dumps(pairwise_metrics_json, indent=2), encoding="utf-8")

    top_ns = sorted(set(int(x) for x in args.analysis_top_ns if int(x) > 0))
    if not top_ns:
        top_ns = [20, 50, 100]

    top_sets: Dict[str, Dict[int, Set[str]]] = {"model": {}, "clinvar": {}, "gwas": {}}
    for source_name, ser in score_series_norm.items():
        for top_n in top_ns:
            top_df = ser.sort_values(ascending=False).head(top_n).reset_index()
            top_df.columns = ["gene_id", "score"]
            top_df["rank"] = np.arange(1, len(top_df) + 1)
            top_df.to_csv(out_dir / f"top_{source_name}_genes_top{top_n}.csv", index=False)
            top_sets[source_name][top_n] = set(top_df["gene_id"].astype(str).tolist())

    seed_sets = {"model": set(model_seed_ids), "clinvar": set(clinvar_seed_ids), "gwas": set(gwas_seed_ids)}

    seed_dist_detail_frames: List[pd.DataFrame] = []
    seed_dist_summary_rows: List[Dict[str, object]] = []
    for a, b in pair_list:
        d_ab = _nearest_distance_records(network_result.graph, seed_sets[a], seed_sets[b], direction=f"{a}_seed_to_{b}_seed")
        d_ba = _nearest_distance_records(network_result.graph, seed_sets[b], seed_sets[a], direction=f"{b}_seed_to_{a}_seed")
        seed_dist_detail_frames.extend([d_ab, d_ba])
        seed_dist_summary_rows.append(_summarize_distance_records(d_ab, f"{a}_seed_to_{b}_seed"))
        seed_dist_summary_rows.append(_summarize_distance_records(d_ba, f"{b}_seed_to_{a}_seed"))

    seed_dist_details = pd.concat(seed_dist_detail_frames, ignore_index=True)
    seed_dist_summary_df = pd.DataFrame(seed_dist_summary_rows)
    seed_dist_details.to_csv(out_dir / "three_way_seed_shortest_path_details.csv", index=False)
    seed_dist_summary_df.to_csv(out_dir / "three_way_seed_shortest_path_summary.csv", index=False)

    top_dist_detail_frames: List[pd.DataFrame] = []
    top_dist_summary_rows: List[Dict[str, object]] = []
    cc_rows: List[Dict[str, object]] = []
    for top_n in top_ns:
        for a, b in pair_list:
            set_a = top_sets[a][top_n]
            set_b = top_sets[b][top_n]
            d_ab = _nearest_distance_records(
                network_result.graph, set_a, set_b, direction=f"{a}_top{top_n}_to_{b}_top{top_n}", top_n=top_n
            )
            d_ba = _nearest_distance_records(
                network_result.graph, set_b, set_a, direction=f"{b}_top{top_n}_to_{a}_top{top_n}", top_n=top_n
            )
            top_dist_detail_frames.extend([d_ab, d_ba])
            top_dist_summary_rows.append(_summarize_distance_records(d_ab, d_ab["direction"].iloc[0], top_n))
            top_dist_summary_rows.append(_summarize_distance_records(d_ba, d_ba["direction"].iloc[0], top_n))

            shared = set_a.intersection(set_b)
            both = pd.concat(
                [
                    pd.to_numeric(d_ab["nearest_distance"], errors="coerce"),
                    pd.to_numeric(d_ba["nearest_distance"], errors="coerce"),
                ],
                ignore_index=True,
            )
            finite_both = both[np.isfinite(both)]
            top_dist_summary_rows.append(
                {
                    "direction": f"bidirectional_{a}_vs_{b}_top{top_n}",
                    "top_n": int(top_n),
                    "n_sources": int(len(set_a) + len(set_b)),
                    "n_finite": int(len(finite_both)),
                    "n_infinite": int((~np.isfinite(both)).sum()),
                    "mean_distance": float(finite_both.mean()) if len(finite_both) else np.nan,
                    "median_distance": float(finite_both.median()) if len(finite_both) else np.nan,
                    "min_distance": float(finite_both.min()) if len(finite_both) else np.nan,
                    "max_distance": float(finite_both.max()) if len(finite_both) else np.nan,
                    "distance_distribution": np.nan,
                    "set_overlap_count": int(len(shared)),
                    "set_jaccard": float(len(shared) / len(set_a.union(set_b))) if set_a.union(set_b) else np.nan,
                }
            )

        union_top = top_sets["model"][top_n].union(top_sets["clinvar"][top_n]).union(top_sets["gwas"][top_n])
        sub_top = network_result.graph.subgraph(union_top).copy()
        components = sorted(nx.connected_components(sub_top), key=len, reverse=True)
        largest = components[0] if components else set()
        cc_rows.append(
            {
                "top_n": int(top_n),
                "union_size": int(len(union_top)),
                "num_components": int(len(components)),
                "largest_component_size": int(len(largest)),
                "model_fraction_in_largest_component": (
                    float(len(top_sets["model"][top_n].intersection(largest)) / len(top_sets["model"][top_n]))
                    if len(top_sets["model"][top_n])
                    else np.nan
                ),
                "clinvar_fraction_in_largest_component": (
                    float(len(top_sets["clinvar"][top_n].intersection(largest)) / len(top_sets["clinvar"][top_n]))
                    if len(top_sets["clinvar"][top_n])
                    else np.nan
                ),
                "gwas_fraction_in_largest_component": (
                    float(len(top_sets["gwas"][top_n].intersection(largest)) / len(top_sets["gwas"][top_n]))
                    if len(top_sets["gwas"][top_n])
                    else np.nan
                ),
            }
        )

    top_dist_details = pd.concat(top_dist_detail_frames, ignore_index=True)
    top_dist_summary_df = pd.DataFrame(top_dist_summary_rows)
    cc_df = pd.DataFrame(cc_rows)
    top_dist_details.to_csv(out_dir / "three_way_top_gene_shortest_path_details.csv", index=False)
    top_dist_summary_df.to_csv(out_dir / "three_way_top_gene_shortest_path_summary.csv", index=False)
    cc_df.to_csv(out_dir / "three_way_top_gene_connected_component_summary.csv", index=False)

    module_top_n = max(top_ns)
    model_top_module = top_sets["model"][module_top_n]
    clinvar_top_module = top_sets["clinvar"][module_top_n]
    gwas_top_module = top_sets["gwas"][module_top_n]
    union_module = model_top_module.union(clinvar_top_module).union(gwas_top_module)
    seed_union = seed_sets["model"].union(seed_sets["clinvar"]).union(seed_sets["gwas"])
    community_nodes = union_module.union(seed_union)
    community_subgraph = network_result.graph.subgraph(community_nodes).copy()

    if community_subgraph.number_of_edges() > 0:
        communities = list(nx.algorithms.community.greedy_modularity_communities(community_subgraph))
    else:
        communities = [set([node]) for node in community_subgraph.nodes()]
    communities = sorted([set(str(g) for g in c) for c in communities], key=len, reverse=True)

    symbol_map = _load_ensembl_to_symbol_map(args.hgnc_path)
    comm_summary_rows: List[Dict[str, object]] = []
    comm_detail_rows: List[Dict[str, object]] = []
    for idx, comm in enumerate(communities, start=1):
        module_id = f"M{idx}"
        m = len(comm.intersection(model_top_module))
        c = len(comm.intersection(clinvar_top_module))
        g = len(comm.intersection(gwas_top_module))
        shared_mc = len(comm.intersection(model_top_module.intersection(clinvar_top_module)))
        shared_mg = len(comm.intersection(model_top_module.intersection(gwas_top_module)))
        shared_cg = len(comm.intersection(clinvar_top_module.intersection(gwas_top_module)))
        shared_all = len(comm.intersection(model_top_module.intersection(clinvar_top_module).intersection(gwas_top_module)))

        if m > 0 and c > 0 and g > 0:
            module_type = "all_three_mixed"
        elif m > 0 and c > 0:
            module_type = "model_clinvar_mixed"
        elif m > 0 and g > 0:
            module_type = "model_gwas_mixed"
        elif c > 0 and g > 0:
            module_type = "clinvar_gwas_mixed"
        elif m > 0:
            module_type = "model_only_top"
        elif c > 0:
            module_type = "clinvar_only_top"
        elif g > 0:
            module_type = "gwas_only_top"
        else:
            module_type = "seed_only_or_other"

        comm_summary_rows.append(
            {
                "module_id": module_id,
                "module_size": int(len(comm)),
                "model_top_genes_count": int(m),
                "clinvar_top_genes_count": int(c),
                "gwas_top_genes_count": int(g),
                "shared_model_clinvar_count": int(shared_mc),
                "shared_model_gwas_count": int(shared_mg),
                "shared_clinvar_gwas_count": int(shared_cg),
                "shared_all_three_count": int(shared_all),
                "model_seed_count": int(len(comm.intersection(seed_sets["model"]))),
                "clinvar_seed_count": int(len(comm.intersection(seed_sets["clinvar"]))),
                "gwas_seed_count": int(len(comm.intersection(seed_sets["gwas"]))),
                "module_type": module_type,
            }
        )

        for gene_id in sorted(comm):
            comm_detail_rows.append(
                {
                    "module_id": module_id,
                    "gene_id": gene_id,
                    "gene_symbol": symbol_map.get(gene_id, ""),
                    "node_category": _three_way_node_category(gene_id, model_top_module, clinvar_top_module, gwas_top_module, seed_union),
                    "is_model_top": int(gene_id in model_top_module),
                    "is_clinvar_top": int(gene_id in clinvar_top_module),
                    "is_gwas_top": int(gene_id in gwas_top_module),
                    "is_model_seed": int(gene_id in seed_sets["model"]),
                    "is_clinvar_seed": int(gene_id in seed_sets["clinvar"]),
                    "is_gwas_seed": int(gene_id in seed_sets["gwas"]),
                    "model_score": float(ppr_model_norm.get(gene_id, np.nan)),
                    "clinvar_score": float(ppr_clinvar_norm.get(gene_id, np.nan)),
                    "gwas_score": float(ppr_gwas_norm.get(gene_id, np.nan)),
                }
            )

    comm_summary_df = pd.DataFrame(comm_summary_rows)
    comm_detail_df = pd.DataFrame(comm_detail_rows)
    comm_summary_df.to_csv(out_dir / "three_way_community_summary.csv", index=False)
    comm_detail_df.to_csv(out_dir / "three_way_community_detailed.csv", index=False)

    _plot_three_way_induced_subnetwork(
        subgraph=community_subgraph,
        model_top=model_top_module,
        clinvar_top=clinvar_top_module,
        gwas_top=gwas_top_module,
        model_seeds=seed_sets["model"],
        clinvar_seeds=seed_sets["clinvar"],
        gwas_seeds=seed_sets["gwas"],
        model_scores=ppr_model_norm,
        clinvar_scores=ppr_clinvar_norm,
        gwas_scores=ppr_gwas_norm,
        symbol_map=symbol_map,
        out_path=out_dir / "three_way_induced_subnetwork.png",
    )
    _plot_three_way_community_summary(
        community_summary_df=comm_summary_df,
        out_path=out_dir / "three_way_community_summary.png",
    )

    model_label = str(args.model_source_label)
    model_prior_desc = str(args.model_prior_description)

    model_vs_gwas = pairwise_df.loc[pairwise_df["pair"] == "model_vs_gwas"].iloc[0]
    clinvar_vs_gwas = pairwise_df.loc[pairwise_df["pair"] == "clinvar_vs_gwas"].iloc[0]
    if float(model_vs_gwas["norm_spearman"]) > float(clinvar_vs_gwas["norm_spearman"]):
        gwas_closeness = f"GWAS appears closer to {model_label} (by normalized Spearman)."
    elif float(model_vs_gwas["norm_spearman"]) < float(clinvar_vs_gwas["norm_spearman"]):
        gwas_closeness = "GWAS appears closer to ClinVar (by normalized Spearman)."
    else:
        gwas_closeness = f"GWAS shows similar closeness to {model_label} and ClinVar (by normalized Spearman)."

    report_lines = [
        f"# INTACT PPR Three-Way Report ({model_label} vs ClinVar vs GWAS)",
        "",
        "## Source Framing",
        f"- Functional prior ({model_label}): `{model_prior_desc}`",
        "- Genetic prior: GWAS-derived seeds (`gwasCredibleSets`).",
        "- Clinical prior: ClinVar/EVA-derived seeds (`eva`).",
        "",
        "## GWAS Seed Definition",
        f"- Selected GWAS source column: `{args.gwas_source_column}`",
        "- Rationale: `gwasCredibleSets` is the explicit GWAS-derived Open Targets evidence column already used in project evaluation/source exports.",
        f"- Threshold used: `{float(args.gwas_threshold):.2f}`",
        "- Alternative considered: `geneBurden` (also genetic evidence), but we kept `gwasCredibleSets` for direct GWAS consistency.",
        "",
        "## Seed Counts and Mapping",
        f"- {model_label} seeds selected/mapped/unmapped: `{model_seed_report['selected_seed_count']}` / `{model_seed_map.report['mapped_seed_count']}` / `{model_seed_map.report['unmapped_seed_count']}`",
        f"- ClinVar seeds selected/mapped/unmapped: `{clinvar_seed_report['selected_seed_count_after_dedup']}` / `{clinvar_seed_map.report['mapped_seed_count']}` / `{clinvar_seed_map.report['unmapped_seed_count']}`",
        f"- GWAS seeds selected/mapped/unmapped: `{gwas_seed_report['selected_seed_count_after_dedup']}` / `{gwas_seed_map.report['mapped_seed_count']}` / `{gwas_seed_map.report['unmapped_seed_count']}`",
        "",
        "## Pairwise Comparison Metrics",
        _markdown_table_from_df(pairwise_df),
        "",
        "## Three-way Neighborhood/Module Summary",
        f"- Top-N sets analyzed: `{', '.join(str(n) for n in top_ns)}`",
        _markdown_table_from_df(
            cc_df[
                [
                    "top_n",
                    "union_size",
                    "num_components",
                    "largest_component_size",
                    "model_fraction_in_largest_component",
                    "clinvar_fraction_in_largest_component",
                    "gwas_fraction_in_largest_component",
                ]
            ]
        ),
        "",
        f"- Communities detected (Top-{module_top_n} union + seeds): `{len(comm_summary_df)}`",
        _markdown_table_from_df(
            comm_summary_df[
                [
                    "module_id",
                    "module_size",
                    "model_top_genes_count",
                    "clinvar_top_genes_count",
                    "gwas_top_genes_count",
                    "shared_all_three_count",
                    "module_type",
                ]
            ]
        ),
        "",
        "## Plots",
        "![Three-way induced subnetwork](three_way_induced_subnetwork.png)",
        "",
        "![Three-way community summary](three_way_community_summary.png)",
        "",
        "## Interpretation",
        f"- {gwas_closeness}",
        "- Interpretation is descriptive and graph-structure based; no biological causality is implied.",
    ]
    _write_text(out_dir / "three_way_report.md", "\n".join(report_lines) + "\n")

    project_mapped_out = project_mapping_result.mapped_df[
        ["gene_id_raw", "gene_id_normalized", "gene_symbol", "mapped_to_network"]
    ].rename(columns={"gene_id_normalized": "network_gene_id"})
    project_unmapped_out = project_mapping_result.unmapped_df[
        ["gene_id_raw", "gene_id_normalized", "gene_symbol", "mapped_to_network"]
    ].rename(columns={"gene_id_normalized": "network_gene_id"})
    project_mapped_out.to_csv(out_dir / "project_genes_mapped.csv", index=False)
    project_unmapped_out.to_csv(out_dir / "project_genes_unmapped.csv", index=False)

    summary = {
        "script": "intact_ppr_pipeline.py",
        "status": "completed_three_way_comparison",
        "inputs": {
            "network_path": str(args.network_path),
            "background_ppr_path": str(args.background_ppr_path),
            "project_gene_table_path": str(project_gene_table_path),
            "comparison_root": str(args.comparison_root),
            "model_predictions_path": str(args.model_predictions_path) if args.model_predictions_path is not None else None,
            "ot_json_path": str(args.ot_json_path),
            "hgnc_path": str(args.hgnc_path),
        },
        "gwas_source_selection": {
            "selected_source_column": args.gwas_source_column,
            "threshold": float(args.gwas_threshold),
            "rationale": "Open Targets gwasCredibleSets column used as GWAS proxy, with threshold style aligned to existing 0.5 convention.",
            "alternatives_considered": ["geneBurden"],
        },
        "model_run_selection": model_run_report,
        "model_seed_selection": model_seed_report,
        "clinvar_seed_selection": clinvar_seed_report,
        "gwas_seed_selection": gwas_seed_report,
        "model_seed_mapping": model_seed_map.report,
        "clinvar_seed_mapping": clinvar_seed_map.report,
        "gwas_seed_mapping": gwas_seed_map.report,
        "ppr_parameters": {
            "alpha": float(args.ppr_alpha),
            "max_iter": int(args.ppr_max_iter),
            "tol": float(args.ppr_tol),
        },
        "pairwise_comparison_metrics": pairwise_metrics_json,
        "three_way_connected_components": cc_rows,
        "three_way_community_summary": comm_summary_rows,
        "outputs": {
            "out_dir": str(out_dir),
            "pairwise_comparison_metrics_csv": str(out_dir / "pairwise_comparison_metrics.csv"),
            "pairwise_comparison_metrics_json": str(out_dir / "pairwise_comparison_metrics.json"),
            "three_way_report_md": str(out_dir / "three_way_report.md"),
            "three_way_plot_png": str(out_dir / "three_way_induced_subnetwork.png"),
            "three_way_community_png": str(out_dir / "three_way_community_summary.png"),
        },
    }
    (out_dir / "three_way_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("INTACT PPR three-way pipeline completed.")
    print(f"Output directory: {out_dir}")
    print(
        "Seeds mapped (model/clinvar/gwas): "
        f"{model_seed_map.report['mapped_seed_count']}/"
        f"{clinvar_seed_map.report['mapped_seed_count']}/"
        f"{gwas_seed_map.report['mapped_seed_count']}"
    )


def _benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    adj = np.full_like(p, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    if finite_mask.sum() == 0:
        return adj

    p_finite = p[finite_mask]
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    m = float(len(ranked))
    bh = ranked * m / (np.arange(1, len(ranked) + 1))
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    mapped = bh[inv_order]
    adj[finite_mask] = mapped
    return adj


def _load_consensus_modules(
    module_clusters_path: Path,
    min_module_size: int,
) -> Tuple[Dict[str, Set[str]], pd.DataFrame, Dict[str, object]]:
    module_df = pd.read_csv(module_clusters_path)
    if "Gene" not in module_df.columns or "Cluster" not in module_df.columns:
        raise ValueError(f"Module file must contain Gene and Cluster columns: {module_clusters_path}")

    cur = module_df.copy()
    cur["Gene"] = cur["Gene"].map(normalize_gene_id)
    cur = cur.dropna(subset=["Gene", "Cluster"]).copy()
    cur["Cluster"] = cur["Cluster"].astype(str)
    cur = cur.drop_duplicates(subset=["Cluster", "Gene"], keep="first")

    size_all = cur.groupby("Cluster", as_index=False)["Gene"].nunique().rename(columns={"Gene": "module_size_all"})
    size_all["kept_for_enrichment"] = size_all["module_size_all"] >= int(min_module_size)

    kept_ids = set(size_all.loc[size_all["kept_for_enrichment"], "Cluster"].astype(str))
    kept = cur.loc[cur["Cluster"].isin(kept_ids)].copy()
    module_dict: Dict[str, Set[str]] = {
        str(cluster): set(df["Gene"].astype(str).tolist()) for cluster, df in kept.groupby("Cluster")
    }

    module_summary = size_all.sort_values("module_size_all", ascending=False, kind="stable").reset_index(drop=True)
    report = {
        "module_clusters_path": str(module_clusters_path),
        "rows_total": int(len(module_df)),
        "rows_after_normalization_dedup": int(len(cur)),
        "modules_total": int(size_all.shape[0]),
        "modules_kept": int(len(module_dict)),
        "min_module_size_filter": int(min_module_size),
        "module_size_min_all": int(size_all["module_size_all"].min()) if not size_all.empty else 0,
        "module_size_median_all": float(size_all["module_size_all"].median()) if not size_all.empty else 0.0,
        "module_size_max_all": int(size_all["module_size_all"].max()) if not size_all.empty else 0,
    }
    return module_dict, module_summary, report


def _plot_module_size_histogram(module_summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    vals = pd.to_numeric(module_summary_df["module_size_all"], errors="coerce").dropna().values
    if len(vals) == 0:
        plt.text(0.5, 0.5, "No modules available", ha="center", va="center")
        plt.axis("off")
    else:
        bins = min(50, max(10, int(np.sqrt(len(vals)))))
        plt.hist(vals, bins=bins, color="#4c78a8", alpha=0.85, edgecolor="white")
        plt.xlabel("Module size (genes)")
        plt.ylabel("Number of modules")
        plt.title("Consensus Module Size Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _run_fgsea_r_pipeline(
    modules_csv_path: Path,
    rankings_manifest_csv: Path,
    out_dir: Path,
    min_module_size: int,
    micromamba_path: Path,
    micromamba_root: Path,
    fgsea_env_name: str,
) -> None:
    r_script_path = out_dir / "_run_fgsea_modules.R"
    r_script = r"""suppressPackageStartupMessages({
  library(data.table)
  library(fgsea)
  library(BiocParallel)
})
register(SerialParam(), default = TRUE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript _run_fgsea_modules.R <modules_csv> <manifest_csv> <out_dir> <min_size>")
}

modules_csv <- args[1]
manifest_csv <- args[2]
out_dir <- args[3]
min_size <- as.integer(args[4])

mod <- fread(modules_csv, data.table = TRUE)
if (!all(c("Gene", "Cluster") %in% colnames(mod))) {
  stop("modules_csv must contain Gene and Cluster columns.")
}
mod <- mod[!is.na(Gene) & !is.na(Cluster)]
mod[, Gene := sub("\\..*$", "", as.character(Gene))]
mod[, Cluster := as.character(Cluster)]
mod <- unique(mod[, .(Cluster, Gene)])

path_df <- mod[, .(genes = list(unique(Gene))), by = Cluster]
path_df <- path_df[lengths(genes) >= min_size]
if (nrow(path_df) == 0) {
  stop("No modules passed min_size filter.")
}
pathways <- setNames(path_df$genes, path_df$Cluster)

manifest <- fread(manifest_csv, data.table = FALSE)
if (!all(c("method", "rank_path") %in% colnames(manifest))) {
  stop("manifest_csv must contain method and rank_path columns.")
}

for (i in seq_len(nrow(manifest))) {
  method <- as.character(manifest$method[i])
  rank_path <- as.character(manifest$rank_path[i])
  ranks <- fread(rank_path, data.table = FALSE)
  if (!all(c("gene_id", "score") %in% colnames(ranks))) {
    stop(sprintf("Ranking file missing gene_id/score columns: %s", rank_path))
  }
  ranks <- ranks[!is.na(ranks$gene_id) & !is.na(ranks$score), c("gene_id", "score")]
  ranks$gene_id <- sub("\\..*$", "", as.character(ranks$gene_id))
  ranks <- ranks[!duplicated(ranks$gene_id), ]
  ranks <- ranks[order(-ranks$score), ]

  statsVec <- setNames(as.numeric(ranks$score), ranks$gene_id)
  statsVec <- sort(statsVec, decreasing = TRUE)

  fg <- fgsea(
    pathways = pathways,
    stats = statsVec,
    minSize = min_size,
    maxSize = Inf,
    eps = 0,
    nproc = 1,
    BPPARAM = SerialParam()
  )
  fg <- as.data.frame(fg)
  if ("leadingEdge" %in% colnames(fg)) {
    fg$leadingEdge <- vapply(fg$leadingEdge, function(x) paste(x, collapse = ";"), character(1))
  }
  fg$method <- method
  fg <- fg[order(fg$padj, -fg$NES), ]

  out_path <- file.path(out_dir, paste0("module_enrichment_", method, ".csv"))
  fwrite(fg, out_path)
}
"""
    r_script_path.write_text(r_script, encoding="utf-8")

    cmd = [
        str(micromamba_path),
        "run",
        "-n",
        str(fgsea_env_name),
        "Rscript",
        str(r_script_path),
        str(modules_csv_path),
        str(rankings_manifest_csv),
        str(out_dir),
        str(int(min_module_size)),
    ]
    env = dict(**os.environ, MAMBA_ROOT_PREFIX=str(micromamba_root))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            "fgsea R execution failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _compute_preranked_es(
    ranked_genes: np.ndarray,
    ranked_scores: np.ndarray,
    gene_set: Set[str],
) -> Tuple[float, List[str]]:
    if len(gene_set) == 0:
        return float("nan"), []

    hit_mask = np.isin(ranked_genes, np.array(sorted(gene_set), dtype=object))
    nh = int(hit_mask.sum())
    n = int(len(ranked_genes))
    if nh == 0 or nh == n:
        return float("nan"), []

    abs_scores = np.abs(ranked_scores).astype(float)
    hit_weights = abs_scores * hit_mask
    denom_hit = float(hit_weights.sum())
    if denom_hit <= 0:
        hit_weights = hit_mask.astype(float)
        denom_hit = float(hit_weights.sum())

    p_hit = hit_weights / denom_hit
    p_miss = (~hit_mask).astype(float) / float(n - nh)
    running = np.cumsum(p_hit - p_miss)

    max_idx = int(np.argmax(running))
    min_idx = int(np.argmin(running))
    max_es = float(running[max_idx])
    min_es = float(running[min_idx])

    if abs(max_es) >= abs(min_es):
        es = max_es
        leading_hits = ranked_genes[: max_idx + 1][hit_mask[: max_idx + 1]]
    else:
        es = min_es
        leading_hits = ranked_genes[min_idx:][hit_mask[min_idx:]]

    leading_edge = [str(g) for g in leading_hits.tolist()[:50]]
    return float(es), leading_edge


def _run_module_enrichment_for_scores(
    normalized_scores: pd.Series,
    module_dict: Dict[str, Set[str]],
    method_name: str,
) -> pd.DataFrame:
    ser = normalized_scores.copy()
    ser.index = ser.index.map(normalize_gene_id)
    ser = ser[ser.index.notna()].astype(float)
    ser = ser[~ser.index.duplicated(keep="first")]
    ser = ser.sort_values(ascending=False, kind="stable")

    ranked_genes = np.array(ser.index.astype(str).tolist(), dtype=object)
    ranked_scores = ser.values.astype(float)
    universe_set = set(ranked_genes.tolist())
    n = int(len(ranked_genes))
    if n < 5:
        raise ValueError(f"Not enough ranked genes for module enrichment: {method_name}")

    # Rank-based normal approximation (practical Python equivalent to fgsea significance screen).
    ranks = stats.rankdata(ranked_scores, method="average")
    idx_map = {g: i for i, g in enumerate(ranked_genes.tolist())}

    rows: List[Dict[str, object]] = []
    for module_id, genes in module_dict.items():
        genes_present = sorted(list(set(genes).intersection(universe_set)))
        n1 = len(genes_present)
        n2 = n - n1
        if n1 < 2 or n2 < 2:
            continue

        idx = np.array([idx_map[g] for g in genes_present], dtype=int)
        module_scores = ranked_scores[idx]
        mean_score = float(np.mean(module_scores))
        median_score = float(np.median(module_scores))

        # One-sided Mann-Whitney style z/p without materializing complement each time.
        rank_sum = float(np.sum(ranks[idx]))
        u_stat = rank_sum - n1 * (n1 + 1) / 2.0
        mu_u = n1 * n2 / 2.0
        sigma_u = np.sqrt(n1 * n2 * (n + 1) / 12.0)
        if sigma_u > 0:
            z_score = float((u_stat - mu_u) / sigma_u)
            p_value = float(stats.norm.sf(z_score))
        else:
            z_score = float("nan")
            p_value = float("nan")

        es, leading_edge = _compute_preranked_es(ranked_genes, ranked_scores, set(genes_present))
        top_module_genes = (
            ser.loc[genes_present].sort_values(ascending=False, kind="stable").head(20).index.astype(str).tolist()
        )

        rows.append(
            {
                "method": method_name,
                "module_id": str(module_id),
                "module_size": int(len(genes)),
                "module_size_scored": int(n1),
                "ES": float(es) if np.isfinite(es) else np.nan,
                "NES": float(z_score) if np.isfinite(z_score) else np.nan,
                "pval": float(p_value) if np.isfinite(p_value) else np.nan,
                "mean_module_score": mean_score,
                "median_module_score": median_score,
                "leading_edge_genes": ";".join(leading_edge),
                "top_module_genes_by_score": ";".join(top_module_genes),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["padj"] = _benjamini_hochberg(out["pval"].tolist())
    out = out.sort_values(["padj", "NES", "ES"], ascending=[True, False, False], kind="stable").reset_index(drop=True)
    return out


def _plot_top_enriched_modules(
    enrichment_df: pd.DataFrame,
    method_label: str,
    fdr_threshold: float,
    top_n: int,
    out_path: Path,
) -> None:
    cur = enrichment_df.copy()
    cur = cur.loc[(cur["padj"] <= float(fdr_threshold)) & (cur["ES"] > 0)].copy()
    if cur.empty:
        cur = enrichment_df.head(int(top_n)).copy()
    else:
        cur = cur.sort_values("NES", ascending=False, kind="stable").head(int(top_n))

    plt.figure(figsize=(8, max(4, 0.38 * len(cur))))
    if cur.empty:
        plt.text(0.5, 0.5, "No enriched modules to show", ha="center", va="center")
        plt.axis("off")
    else:
        y = np.arange(len(cur))
        plt.barh(y, cur["NES"].astype(float).values, color="#4c78a8")
        plt.yticks(y, cur["module_id"].astype(str).tolist())
        plt.xlabel("NES (rank-based z-score)")
        plt.ylabel("Module")
        plt.title(f"Top Enriched Modules - {method_label}")
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_module_enrichment_heatmap(
    comparison_df: pd.DataFrame,
    out_path: Path,
    top_rows: int = 60,
) -> None:
    if comparison_df.empty:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "No enriched modules for heatmap", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    cur = comparison_df.copy()
    cur["n_methods_sig"] = (
        cur["sig_pca"].astype(int) + cur["sig_clinvar"].astype(int) + cur["sig_gwas"].astype(int)
    )
    cur["max_abs_nes"] = (
        cur[["NES_pca", "NES_clinvar", "NES_gwas"]].abs().max(axis=1)
    )
    cur = cur.sort_values(["n_methods_sig", "max_abs_nes"], ascending=[False, False], kind="stable").head(int(top_rows))

    mat = cur[["NES_pca", "NES_clinvar", "NES_gwas"]].fillna(0.0).values.astype(float)
    labels = cur["module_id"].astype(str).tolist()

    plt.figure(figsize=(7, max(6, 0.20 * len(labels))))
    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 1e-6)
    plt.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="NES (rank-based z-score)")
    plt.xticks([0, 1, 2], ["PCA", "ClinVar", "GWAS"])
    plt.yticks(np.arange(len(labels)), labels)
    plt.title("Cross-method Module Enrichment Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def _categorize_module_overlap(sig_pca: bool, sig_clinvar: bool, sig_gwas: bool) -> str:
    if sig_pca and sig_clinvar and sig_gwas:
        return "all_three"
    if sig_pca and sig_clinvar:
        return "pca_clinvar_only"
    if sig_pca and sig_gwas:
        return "pca_gwas_only"
    if sig_clinvar and sig_gwas:
        return "clinvar_gwas_only"
    if sig_pca:
        return "pca_only"
    if sig_clinvar:
        return "clinvar_only"
    if sig_gwas:
        return "gwas_only"
    return "none"


def run_module_enrichment_pipeline(args: argparse.Namespace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dir = args.module_enrichment_input_dir
    if input_dir is None:
        input_dir = DEFAULT_THREE_WAY_OUT_DIR
    if not input_dir.exists():
        raise FileNotFoundError(f"Module-enrichment input directory not found: {input_dir}")

    modules_path = args.module_clusters_path
    _module_dict, module_summary_df, module_load_report = _load_consensus_modules(
        module_clusters_path=modules_path,
        min_module_size=int(args.module_min_size),
    )
    module_summary_df.to_csv(out_dir / "module_summary.csv", index=False)
    _plot_module_size_histogram(module_summary_df, out_dir / "module_size_distribution.png")

    model_norm = _load_score_series_from_csv(input_dir / "ppr_scores_model_seeds_normalized.csv")
    clinvar_norm = _load_score_series_from_csv(input_dir / "ppr_scores_clinvar_seeds_normalized.csv")
    gwas_norm = _load_score_series_from_csv(input_dir / "ppr_scores_gwas_seeds_normalized.csv")

    # Prepare preranked inputs for faithful fgsea in R.
    pca_rank_path = out_dir / "_rank_pca.csv"
    clinvar_rank_path = out_dir / "_rank_clinvar.csv"
    gwas_rank_path = out_dir / "_rank_gwas.csv"
    _save_scored_series(model_norm, pca_rank_path)
    _save_scored_series(clinvar_norm, clinvar_rank_path)
    _save_scored_series(gwas_norm, gwas_rank_path)

    manifest = pd.DataFrame(
        {
            "method": ["pca", "clinvar", "gwas"],
            "rank_path": [str(pca_rank_path), str(clinvar_rank_path), str(gwas_rank_path)],
        }
    )
    manifest_path = out_dir / "_fgsea_rank_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    _run_fgsea_r_pipeline(
        modules_csv_path=modules_path,
        rankings_manifest_csv=manifest_path,
        out_dir=out_dir,
        min_module_size=int(args.module_min_size),
        micromamba_path=args.fgsea_micromamba_path,
        micromamba_root=args.fgsea_mamba_root_prefix,
        fgsea_env_name=args.fgsea_env_name,
    )

    # Read fgsea outputs and standardize schema used downstream.
    def _read_fgsea_standardized(path: Path, method: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Expected fgsea output not found: {path}")
        df = pd.read_csv(path)
        rename_map = {}
        if "pathway" in df.columns:
            rename_map["pathway"] = "module_id"
        if "size" in df.columns:
            rename_map["size"] = "module_size"
        if rename_map:
            df = df.rename(columns=rename_map)
        req = {"module_id", "module_size", "ES", "NES", "pval", "padj"}
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"fgsea output missing columns {missing}: {path}")

        df["module_id"] = df["module_id"].astype(str)
        df["module_size"] = pd.to_numeric(df["module_size"], errors="coerce")
        df["ES"] = pd.to_numeric(df["ES"], errors="coerce")
        df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
        df["pval"] = pd.to_numeric(df["pval"], errors="coerce")
        df["padj"] = pd.to_numeric(df["padj"], errors="coerce")
        df["method"] = method
        if "leadingEdge" in df.columns:
            df["leading_edge_genes"] = df["leadingEdge"].astype(str)
        elif "leading_edge_genes" not in df.columns:
            df["leading_edge_genes"] = ""
        return df.sort_values(["padj", "NES"], ascending=[True, False], kind="stable").reset_index(drop=True)

    pca_enrich = _read_fgsea_standardized(out_dir / "module_enrichment_pca.csv", method="pca")
    clinvar_enrich = _read_fgsea_standardized(out_dir / "module_enrichment_clinvar.csv", method="clinvar")
    gwas_enrich = _read_fgsea_standardized(out_dir / "module_enrichment_gwas.csv", method="gwas")

    pca_enrich.to_csv(out_dir / "module_enrichment_pca.csv", index=False)
    clinvar_enrich.to_csv(out_dir / "module_enrichment_clinvar.csv", index=False)
    gwas_enrich.to_csv(out_dir / "module_enrichment_gwas.csv", index=False)

    _plot_top_enriched_modules(
        enrichment_df=pca_enrich,
        method_label="PCA-model",
        fdr_threshold=float(args.module_fdr_threshold),
        top_n=int(args.module_top_plot_n),
        out_path=out_dir / "top_enriched_modules_pca.png",
    )
    _plot_top_enriched_modules(
        enrichment_df=clinvar_enrich,
        method_label="ClinVar/EVA",
        fdr_threshold=float(args.module_fdr_threshold),
        top_n=int(args.module_top_plot_n),
        out_path=out_dir / "top_enriched_modules_clinvar.png",
    )
    _plot_top_enriched_modules(
        enrichment_df=gwas_enrich,
        method_label="GWAS",
        fdr_threshold=float(args.module_fdr_threshold),
        top_n=int(args.module_top_plot_n),
        out_path=out_dir / "top_enriched_modules_gwas.png",
    )

    # Build three-way module comparison table.
    pca_cmp = pca_enrich[["module_id", "module_size", "ES", "NES", "pval", "padj"]].rename(
        columns={"ES": "ES_pca", "NES": "NES_pca", "pval": "pval_pca", "padj": "padj_pca"}
    )
    clin_cmp = clinvar_enrich[["module_id", "ES", "NES", "pval", "padj"]].rename(
        columns={"ES": "ES_clinvar", "NES": "NES_clinvar", "pval": "pval_clinvar", "padj": "padj_clinvar"}
    )
    gwas_cmp = gwas_enrich[["module_id", "ES", "NES", "pval", "padj"]].rename(
        columns={"ES": "ES_gwas", "NES": "NES_gwas", "pval": "pval_gwas", "padj": "padj_gwas"}
    )

    comparison = pca_cmp.merge(clin_cmp, on="module_id", how="outer").merge(gwas_cmp, on="module_id", how="outer")
    comparison["module_size"] = pd.to_numeric(comparison["module_size"], errors="coerce")
    comparison["sig_pca"] = (comparison["padj_pca"] <= float(args.module_fdr_threshold)) & (comparison["ES_pca"] > 0)
    comparison["sig_clinvar"] = (comparison["padj_clinvar"] <= float(args.module_fdr_threshold)) & (
        comparison["ES_clinvar"] > 0
    )
    comparison["sig_gwas"] = (comparison["padj_gwas"] <= float(args.module_fdr_threshold)) & (comparison["ES_gwas"] > 0)
    comparison["overlap_category"] = [
        _categorize_module_overlap(bool(a), bool(b), bool(c))
        for a, b, c in zip(comparison["sig_pca"], comparison["sig_clinvar"], comparison["sig_gwas"])
    ]
    comparison = comparison.sort_values(
        ["sig_pca", "sig_clinvar", "sig_gwas", "NES_pca", "NES_clinvar", "NES_gwas"],
        ascending=[False, False, False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    comparison.to_csv(out_dir / "module_enrichment_comparison_summary.csv", index=False)

    overlap_counts = (
        comparison.loc[comparison["overlap_category"] != "none", "overlap_category"]
        .value_counts(dropna=False)
        .rename_axis("overlap_category")
        .reset_index(name="module_count")
    )
    overlap_counts.to_csv(out_dir / "module_enrichment_overlap_counts.csv", index=False)

    _plot_module_enrichment_heatmap(
        comparison_df=comparison.loc[comparison["overlap_category"] != "none"].copy(),
        out_path=out_dir / "module_enrichment_overlap_heatmap.png",
        top_rows=int(args.module_heatmap_top_rows),
    )

    # Top summaries for report.
    def _top_sig(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        cur = df.loc[(df["padj"] <= float(args.module_fdr_threshold)) & (df["ES"] > 0)].copy()
        if cur.empty:
            cur = df.head(n).copy()
        else:
            cur = cur.sort_values("NES", ascending=False, kind="stable").head(n)
        return cur

    top_pca = _top_sig(pca_enrich, n=10)
    top_clin = _top_sig(clinvar_enrich, n=10)
    top_gwas = _top_sig(gwas_enrich, n=10)

    report_lines = [
        "# INTACT Module Enrichment Report",
        "",
        "## Inputs Used",
        f"- Consensus modules: `{modules_path}`",
        f"- Reference logic inspected: `step21_network_modules_fgsea_network_scores.rmd`, `step22_link_modules_with_traits.ipynb`",
        f"- Three-way normalized score files from: `{input_dir}`",
        "  - `ppr_scores_model_seeds_normalized.csv`",
        "  - `ppr_scores_clinvar_seeds_normalized.csv`",
        "  - `ppr_scores_gwas_seeds_normalized.csv`",
        "",
        "## Module Definition and Filtering",
        f"- Total modules in consensus file: `{module_load_report['modules_total']}`",
        f"- Filtering rule: keep modules with size >= `{module_load_report['min_module_size_filter']}` genes",
        f"- Modules analyzed after filter: `{module_load_report['modules_kept']}`",
        "- Note: this follows the step21 FGSEA spirit (`minSize=3` / clusters with >2 genes).",
        "",
        "## Method Note",
        "- Enrichment is now run with **R fgsea** (same method family as `step21_network_modules_fgsea_network_scores.rmd`).",
        "- Parameters aligned to step21 spirit: preranked stats, `minSize=3`, `maxSize=Inf`, `eps=0`.",
        "- Reported columns (`ES`, `NES`, `pval`, `padj`, `leadingEdge`) come directly from fgsea outputs.",
        "",
        "## Top Enriched Modules - PCA",
        _markdown_table_from_df(top_pca[["module_id", "module_size", "ES", "NES", "pval", "padj"]]),
        "",
        "## Top Enriched Modules - ClinVar",
        _markdown_table_from_df(top_clin[["module_id", "module_size", "ES", "NES", "pval", "padj"]]),
        "",
        "## Top Enriched Modules - GWAS",
        _markdown_table_from_df(top_gwas[["module_id", "module_size", "ES", "NES", "pval", "padj"]]),
        "",
        "## Cross-method Overlap",
        _markdown_table_from_df(overlap_counts if not overlap_counts.empty else pd.DataFrame({"overlap_category": [], "module_count": []})),
        "",
        "## Plots",
        "![Module size distribution](module_size_distribution.png)",
        "",
        "![Top enriched modules - PCA](top_enriched_modules_pca.png)",
        "",
        "![Top enriched modules - ClinVar](top_enriched_modules_clinvar.png)",
        "",
        "![Top enriched modules - GWAS](top_enriched_modules_gwas.png)",
        "",
        "![Enriched-module overlap heatmap](module_enrichment_overlap_heatmap.png)",
        "",
        "## Interpretation",
        "- We observe overlap and divergence in enriched consensus modules across PCA, ClinVar, and GWAS.",
        "- This addresses module-level convergence in network architecture using normalized propagation scores.",
        "- GO-term linking from step22 was not required for this pass and is left as a future extension.",
        "- Interpretation remains descriptive and graph-based, without causal biological claims.",
    ]
    _write_text(out_dir / "module_enrichment_report.md", "\n".join(report_lines) + "\n")

    summary = {
        "script": "intact_ppr_pipeline.py",
        "status": "completed_module_enrichment",
        "inputs": {
            "module_clusters_path": str(modules_path),
            "input_dir": str(input_dir),
            "model_norm_scores": str(input_dir / "ppr_scores_model_seeds_normalized.csv"),
            "clinvar_norm_scores": str(input_dir / "ppr_scores_clinvar_seeds_normalized.csv"),
            "gwas_norm_scores": str(input_dir / "ppr_scores_gwas_seeds_normalized.csv"),
        },
        "module_filter_rule": {
            "min_module_size": int(args.module_min_size),
            "modules_total": int(module_load_report["modules_total"]),
            "modules_analyzed": int(module_load_report["modules_kept"]),
        },
        "significance_rule": {
            "fdr_threshold": float(args.module_fdr_threshold),
            "direction": "ES > 0 (top-enriched)",
        },
        "fgsea_runtime": {
            "micromamba_path": str(args.fgsea_micromamba_path),
            "mamba_root_prefix": str(args.fgsea_mamba_root_prefix),
            "env_name": str(args.fgsea_env_name),
            "params": {"minSize": int(args.module_min_size), "maxSize": "Inf", "eps": 0},
        },
        "overlap_counts": overlap_counts.to_dict(orient="records"),
        "outputs": {
            "out_dir": str(out_dir),
            "module_enrichment_pca_csv": str(out_dir / "module_enrichment_pca.csv"),
            "module_enrichment_clinvar_csv": str(out_dir / "module_enrichment_clinvar.csv"),
            "module_enrichment_gwas_csv": str(out_dir / "module_enrichment_gwas.csv"),
            "comparison_summary_csv": str(out_dir / "module_enrichment_comparison_summary.csv"),
            "report_md": str(out_dir / "module_enrichment_report.md"),
        },
    }
    (out_dir / "module_enrichment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("INTACT module-enrichment pipeline completed.")
    print(f"Input three-way directory: {input_dir}")
    print(f"Output directory: {out_dir}")
    print(
        "Modules analyzed: "
        f"{module_load_report['modules_kept']} (from {module_load_report['modules_total']} total; "
        f"min size {args.module_min_size})"
    )


def run_extract_module_genes(args: argparse.Namespace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    module_id = str(args.extract_module).strip()
    if not module_id:
        raise ValueError("extract-module value is empty.")

    module_df = pd.read_csv(args.module_clusters_path)
    if "Gene" not in module_df.columns or "Cluster" not in module_df.columns:
        raise ValueError(f"Module file must contain Gene and Cluster columns: {args.module_clusters_path}")

    cur = module_df.copy()
    cur["gene_id"] = cur["Gene"].map(normalize_gene_id)
    cur["cluster_id"] = cur["Cluster"].astype(str).str.strip()
    cur = cur.dropna(subset=["gene_id", "cluster_id"]).drop_duplicates(subset=["cluster_id", "gene_id"], keep="first")

    module_genes = sorted(cur.loc[cur["cluster_id"] == module_id, "gene_id"].astype(str).unique().tolist())
    if not module_genes:
        raise ValueError(f"Module '{module_id}' not found (or no valid genes) in {args.module_clusters_path}")

    symbol_map = _load_ensembl_to_symbol_map(args.hgnc_path)
    symbols = [symbol_map.get(g, "") for g in module_genes]
    use_symbols_for_txt = all(bool(s) for s in symbols)

    txt_values = symbols if use_symbols_for_txt else module_genes
    txt_identifier = "gene_symbol" if use_symbols_for_txt else "gene_id"

    txt_path = out_dir / f"module_{module_id}_genes.txt"
    csv_path = out_dir / f"module_{module_id}_genes.csv"

    txt_path.write_text("\n".join(txt_values) + "\n", encoding="utf-8")
    out_df = pd.DataFrame(
        {
            "gene_id": module_genes,
            "gene_symbol": symbols,
        }
    )
    out_df.to_csv(csv_path, index=False)

    summary = {
        "module_id": module_id,
        "module_clusters_path": str(args.module_clusters_path),
        "n_genes": int(len(module_genes)),
        "txt_identifier": txt_identifier,
        "txt_path": str(txt_path),
        "csv_path": str(csv_path),
        "symbols_available_for_all_genes": bool(use_symbols_for_txt),
    }
    (out_dir / f"module_{module_id}_genes_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("INTACT module gene extraction completed.")
    print(f"Module ID: {module_id}")
    print(f"Genes exported: {len(module_genes)}")
    print(f"TXT identifier type: {txt_identifier}")
    print(f"TXT path: {txt_path}")
    print(f"CSV path: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="INTACT PPR pipeline: setup + model/ClinVar seeds + PPR + normalization + comparisons."
    )
    parser.add_argument("--network-path", type=Path, default=DEFAULT_NETWORK_PATH)
    parser.add_argument("--background-ppr-path", type=Path, default=DEFAULT_BACKGROUND_PPR_PATH)
    parser.add_argument("--project-gene-table", type=Path, default=None)
    parser.add_argument("--gene-table-dir", type=Path, default=DEFAULT_GENE_TABLE_DIR)

    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT)
    parser.add_argument("--model-penalty", type=str, default="l1")
    parser.add_argument("--model-mode", type=str, default="pca")
    parser.add_argument(
        "--model-predictions-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit model predictions CSV path. "
            "When provided, comparison-root auto-selection is skipped."
        ),
    )
    parser.add_argument(
        "--model-source-label",
        type=str,
        default="PCA-model",
        help="Human-readable label used in reports for the model-derived seed source.",
    )
    parser.add_argument(
        "--model-prior-description",
        type=str,
        default="Model-derived functional prior.",
        help="Short description of the model prior used in narrative reports.",
    )
    parser.add_argument(
        "--model-seed-threshold",
        type=float,
        default=None,
        help="Optional override for model seed threshold on gene_max_predicted_score.",
    )

    parser.add_argument("--ot-json-path", type=Path, default=DEFAULT_OT_ASSOC_JSON)
    parser.add_argument("--hgnc-path", type=Path, default=DEFAULT_HGNC_PATH)
    parser.add_argument(
        "--clinvar-source-column",
        type=str,
        default="eva",
        help="Open Targets disease-association source column used for ClinVar/EVA seeds.",
    )
    parser.add_argument("--clinvar-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gwas-source-column",
        type=str,
        default="gwasCredibleSets",
        help="Open Targets disease-association source column used as GWAS proxy seeds.",
    )
    parser.add_argument("--gwas-threshold", type=float, default=0.5)

    parser.add_argument("--ppr-alpha", type=float, default=0.85)
    parser.add_argument("--ppr-max-iter", type=int, default=500)
    parser.add_argument("--ppr-tol", type=float, default=1e-9)

    parser.add_argument(
        "--run-three-way-comparison",
        action="store_true",
        help="Run three-way comparison pipeline: model-derived functional prior vs ClinVar vs GWAS.",
    )
    parser.add_argument(
        "--run-module-enrichment",
        action="store_true",
        help="Run consensus-module enrichment analysis on normalized three-way propagation scores.",
    )

    parser.add_argument(
        "--run-threshold-sweep",
        action="store_true",
        help="Run model-seed threshold sensitivity sweep against fixed ClinVar seeds.",
    )
    parser.add_argument(
        "--sweep-thresholds",
        type=float,
        nargs="+",
        default=[0.70, 0.65, 0.60, 0.55, 0.50],
        help="Model seed thresholds to sweep when --run-threshold-sweep is enabled.",
    )
    parser.add_argument(
        "--run-network-neighborhood-analysis",
        action="store_true",
        help="Analyze network-neighborhood/module convergence from existing model-vs-ClinVar propagated outputs.",
    )
    parser.add_argument(
        "--analysis-input-dir",
        type=Path,
        default=None,
        help="Directory containing existing propagated outputs (model-vs-ClinVar run).",
    )
    parser.add_argument(
        "--analysis-out-dir",
        type=Path,
        default=None,
        help="Output directory for network-neighborhood analysis artifacts.",
    )
    parser.add_argument(
        "--analysis-score-type",
        type=str,
        choices=["normalized", "raw"],
        default="normalized",
        help="Which propagated scores to use for top-gene and module analysis.",
    )
    parser.add_argument(
        "--analysis-top-ns",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="Top-N propagated gene set sizes used in neighborhood analysis.",
    )

    parser.add_argument(
        "--module-enrichment-input-dir",
        type=Path,
        default=None,
        help="Input directory containing normalized three-way propagated scores.",
    )
    parser.add_argument(
        "--module-clusters-path",
        type=Path,
        default=DEFAULT_REFERENCE_DIR / "intact_netw_consensus_clusters.csv",
        help="Consensus module definition CSV (expects Gene, Cluster columns).",
    )
    parser.add_argument(
        "--module-min-size",
        type=int,
        default=3,
        help="Minimum module size for enrichment analysis (step21-style default: 3).",
    )
    parser.add_argument(
        "--module-fdr-threshold",
        type=float,
        default=0.01,
        help="Adjusted p-value threshold for calling enriched modules.",
    )
    parser.add_argument(
        "--module-top-plot-n",
        type=int,
        default=15,
        help="Number of top modules shown in each per-method enrichment plot.",
    )
    parser.add_argument(
        "--module-heatmap-top-rows",
        type=int,
        default=60,
        help="Maximum number of modules shown in the overlap heatmap.",
    )
    parser.add_argument(
        "--fgsea-micromamba-path",
        type=Path,
        default=Path("/home/viguinijpv/.local/bin/micromamba"),
        help="Path to micromamba executable used to run R/fgsea.",
    )
    parser.add_argument(
        "--fgsea-mamba-root-prefix",
        type=Path,
        default=Path("/home/viguinijpv/.micromamba"),
        help="MAMBA_ROOT_PREFIX that contains the fgsea environment.",
    )
    parser.add_argument(
        "--fgsea-env-name",
        type=str,
        default="fgsea_env",
        help="Micromamba environment name containing R + fgsea.",
    )
    parser.add_argument(
        "--extract-module",
        type=str,
        default=None,
        help="Extract genes for one specific module ID and export TXT/CSV.",
    )

    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_flags = [
        bool(args.run_threshold_sweep),
        bool(args.run_network_neighborhood_analysis),
        bool(args.run_three_way_comparison),
        bool(args.run_module_enrichment),
        bool(args.extract_module is not None),
    ]
    if sum(run_flags) > 1:
        raise ValueError(
            "Use only one mode at a time: --run-threshold-sweep, --run-network-neighborhood-analysis, "
            "--run-three-way-comparison, --run-module-enrichment, or --extract-module."
        )

    if args.run_network_neighborhood_analysis:
        run_network_neighborhood_analysis(args)
        return

    if args.out_dir is None:
        if args.run_module_enrichment:
            out_dir = DEFAULT_MODULE_ENRICH_OUT_DIR
        elif args.extract_module is not None:
            out_dir = DEFAULT_MODULE_ENRICH_OUT_DIR
        elif args.run_three_way_comparison:
            out_dir = DEFAULT_THREE_WAY_OUT_DIR
        elif args.run_threshold_sweep:
            out_dir = DEFAULT_SWEEP_OUT_DIR
        else:
            out_dir = DEFAULT_OUT_DIR
    else:
        out_dir = args.out_dir

    if args.run_three_way_comparison:
        run_three_way_pipeline(args, out_dir)
        return

    if args.run_module_enrichment:
        run_module_enrichment_pipeline(args, out_dir)
        return

    if args.extract_module is not None:
        run_extract_module_genes(args, out_dir)
        return

    if args.run_threshold_sweep:
        run_threshold_sweep(args, out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup stage (already implemented in previous step).
    network_result = load_network(args.network_path)
    background_result = load_background_ppr(args.background_ppr_path, network_result.node_set)
    project_gene_table_path = choose_project_gene_table(args.project_gene_table, args.gene_table_dir)
    project_genes_result = load_project_genes(project_gene_table_path)
    project_mapping_result = map_project_genes_to_network(project_genes_result.unique_genes_df, network_result.node_set)

    # Model seed selection from requested comparison folder.
    model_run_report = resolve_model_run_from_args(args)
    model_seed_df, model_seed_report = select_model_seed_genes(
        model_predictions_path=Path(model_run_report["predictions_path"]),
        threshold_override=args.model_seed_threshold,
    )

    # ClinVar seed selection using closest existing project logic (OT disease-association source thresholding).
    clinvar_seed_df, clinvar_seed_report = select_l2g_seed_genes_from_existing_logic(
        ot_json_path=args.ot_json_path,
        hgnc_path=args.hgnc_path,
        source_column=args.clinvar_source_column,
        score_threshold=float(args.clinvar_threshold),
        seed_source_label="clinvar",
    )

    # Map seed sets to network nodes.
    model_seed_map = map_seed_genes_to_network(model_seed_df, network_result.node_set, source_name="model")
    clinvar_seed_map = map_seed_genes_to_network(clinvar_seed_df, network_result.node_set, source_name="clinvar")

    # Save seed tables.
    model_seed_df.to_csv(out_dir / "model_seed_genes.csv", index=False)
    clinvar_seed_df.to_csv(out_dir / "clinvar_seed_genes.csv", index=False)

    model_seed_map.mapped_df.to_csv(out_dir / "model_seed_genes_mapped.csv", index=False)
    clinvar_seed_map.mapped_df.to_csv(out_dir / "clinvar_seed_genes_mapped.csv", index=False)
    model_seed_map.unmapped_df.to_csv(out_dir / "model_seed_genes_unmapped.csv", index=False)
    clinvar_seed_map.unmapped_df.to_csv(out_dir / "clinvar_seed_genes_unmapped.csv", index=False)

    # Run PPR for both seed sets (notebook-style settings by default).
    model_seed_ids = model_seed_map.mapped_df["gene_id_normalized"].dropna().astype(str).tolist()
    clinvar_seed_ids = clinvar_seed_map.mapped_df["gene_id_normalized"].dropna().astype(str).tolist()

    if len(model_seed_ids) == 0:
        raise ValueError("No mapped model seeds available for PPR.")
    if len(clinvar_seed_ids) == 0:
        raise ValueError("No mapped ClinVar seeds available for PPR.")

    ppr_model = run_ppr(
        graph=network_result.graph,
        seed_genes=model_seed_ids,
        genes_in_network=network_result.node_set,
        alpha=float(args.ppr_alpha),
        max_iter=int(args.ppr_max_iter),
        tol=float(args.ppr_tol),
    )
    ppr_clinvar = run_ppr(
        graph=network_result.graph,
        seed_genes=clinvar_seed_ids,
        genes_in_network=network_result.node_set,
        alpha=float(args.ppr_alpha),
        max_iter=int(args.ppr_max_iter),
        tol=float(args.ppr_tol),
    )

    ppr_model_out = ppr_model.sort_values(ascending=False).reset_index()
    ppr_model_out.columns = ["gene_id", "score"]
    ppr_clinvar_out = ppr_clinvar.sort_values(ascending=False).reset_index()
    ppr_clinvar_out.columns = ["gene_id", "score"]

    ppr_model_out.to_csv(out_dir / "ppr_scores_model_seeds.csv", index=False)
    ppr_clinvar_out.to_csv(out_dir / "ppr_scores_clinvar_seeds.csv", index=False)

    # Notebook-faithful normalization: log(PPR) - log(background score).
    ppr_model_norm = normalize_propagated_scores(ppr_model, background_result.background_df)
    ppr_clinvar_norm = normalize_propagated_scores(ppr_clinvar, background_result.background_df)

    ppr_model_norm_out = ppr_model_norm.sort_values(ascending=False).reset_index()
    ppr_model_norm_out.columns = ["gene_id", "score"]
    ppr_clinvar_norm_out = ppr_clinvar_norm.sort_values(ascending=False).reset_index()
    ppr_clinvar_norm_out.columns = ["gene_id", "score"]

    ppr_model_norm_out.to_csv(out_dir / "ppr_scores_model_seeds_normalized.csv", index=False)
    ppr_clinvar_norm_out.to_csv(out_dir / "ppr_scores_clinvar_seeds_normalized.csv", index=False)

    comparison_metrics = compute_comparison_metrics(
        ppr_model=ppr_model,
        ppr_clinvar=ppr_clinvar,
        norm_model=ppr_model_norm,
        norm_clinvar=ppr_clinvar_norm,
    )

    # Save previous setup outputs for continuity.
    project_mapped_out = project_mapping_result.mapped_df[
        ["gene_id_raw", "gene_id_normalized", "gene_symbol", "mapped_to_network"]
    ].rename(columns={"gene_id_normalized": "network_gene_id"})
    project_unmapped_out = project_mapping_result.unmapped_df[
        ["gene_id_raw", "gene_id_normalized", "gene_symbol", "mapped_to_network"]
    ].rename(columns={"gene_id_normalized": "network_gene_id"})
    project_mapped_out.to_csv(out_dir / "project_genes_mapped.csv", index=False)
    project_unmapped_out.to_csv(out_dir / "project_genes_unmapped.csv", index=False)

    write_markdown_reports(
        out_dir=out_dir,
        network_report=network_result.report,
        background_report=background_result.report,
        project_report=project_genes_result.report,
        mapping_report=project_mapping_result.report,
        model_run_report=model_run_report,
        model_seed_report=model_seed_report,
        clinvar_seed_report=clinvar_seed_report,
        model_seed_map_report=model_seed_map.report,
        clinvar_seed_map_report=clinvar_seed_map.report,
        comparison_metrics=comparison_metrics,
    )

    summary = {
        "script": "intact_ppr_pipeline.py",
        "status": "completed",
        "inputs": {
            "network_path": str(args.network_path),
            "background_ppr_path": str(args.background_ppr_path),
            "project_gene_table_path": str(project_gene_table_path),
            "comparison_root": str(args.comparison_root),
            "model_predictions_path": str(args.model_predictions_path) if args.model_predictions_path is not None else None,
            "ot_json_path": str(args.ot_json_path),
            "hgnc_path": str(args.hgnc_path),
        },
        "model_run_selection": model_run_report,
        "model_seed_selection": model_seed_report,
        "clinvar_seed_selection": clinvar_seed_report,
        "model_seed_mapping": model_seed_map.report,
        "clinvar_seed_mapping": clinvar_seed_map.report,
        "ppr_parameters": {
            "alpha": float(args.ppr_alpha),
            "max_iter": int(args.ppr_max_iter),
            "tol": float(args.ppr_tol),
        },
        "comparison_metrics": comparison_metrics,
        "outputs": {
            "out_dir": str(out_dir),
            "model_seed_genes": str(out_dir / "model_seed_genes.csv"),
            "clinvar_seed_genes": str(out_dir / "clinvar_seed_genes.csv"),
            "model_seed_genes_mapped": str(out_dir / "model_seed_genes_mapped.csv"),
            "clinvar_seed_genes_mapped": str(out_dir / "clinvar_seed_genes_mapped.csv"),
            "model_seed_genes_unmapped": str(out_dir / "model_seed_genes_unmapped.csv"),
            "clinvar_seed_genes_unmapped": str(out_dir / "clinvar_seed_genes_unmapped.csv"),
            "ppr_scores_model_seeds": str(out_dir / "ppr_scores_model_seeds.csv"),
            "ppr_scores_clinvar_seeds": str(out_dir / "ppr_scores_clinvar_seeds.csv"),
            "ppr_scores_model_seeds_normalized": str(out_dir / "ppr_scores_model_seeds_normalized.csv"),
            "ppr_scores_clinvar_seeds_normalized": str(out_dir / "ppr_scores_clinvar_seeds_normalized.csv"),
            "pipeline_report_md": str(out_dir / "pipeline_report.md"),
            "comparison_metrics_json": str(out_dir / "comparison_metrics.json"),
        },
        "network_report": network_result.report,
        "background_report": background_result.report,
        "project_gene_source_report": project_genes_result.report,
        "project_gene_mapping_report": project_mapping_result.report,
    }

    (out_dir / "comparison_metrics.json").write_text(json.dumps(comparison_metrics, indent=2), encoding="utf-8")
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("INTACT PPR model-vs-ClinVar pipeline completed.")
    print(f"Output directory: {out_dir}")
    print(
        "Model seeds: "
        f"selected={model_seed_report['selected_seed_count']} "
        f"mapped={model_seed_map.report['mapped_seed_count']} "
        f"unmapped={model_seed_map.report['unmapped_seed_count']}"
    )
    print(
        "ClinVar seeds: "
        f"selected={clinvar_seed_report['selected_seed_count_after_dedup']} "
        f"mapped={clinvar_seed_map.report['mapped_seed_count']} "
        f"unmapped={clinvar_seed_map.report['unmapped_seed_count']}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Proof-of-concept global locus-to-gene ranking with strict leakage control.

Core idea:
- Train one global linear model on (locus, gene) rows.
- At evaluation time, rank genes within each locus using predicted scores.

Leakage context:
- This script uses LOLO with gene-exclusion, removing train rows that contain
  any gene appearing in the held-out test locus.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from string_network_propagation import (
    GeneNetworkModel,
    load_or_build_string_gene_network_model,
    normalize_gene_symbol as normalize_network_gene_symbol,
)


DEFAULT_INPUT_TABLE = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/"
    "GCST90027164_cs_gene_candidate_feature_table.csv"
)
DEFAULT_OUT_DIR = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/locus_gene_ranker"
)
DEFAULT_STRING_ALIASES_PATH = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/data/9606.protein.aliases.v12.0.txt"
)
DEFAULT_STRING_LINKS_PATH = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/data/9606.protein.links.v12.0.txt"
)
DEFAULT_STRING_CACHE_PATH = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/string_gene_network_min700.npz"
)
DEFAULT_NETWORK_MIN_COMBINED_SCORE = 700.0
DEFAULT_NETWORK_ALPHA = 0.85
DEFAULT_NETWORK_MAX_ITER = 100
DEFAULT_NETWORK_TOL = 1e-9

LOCUS_COL = "gwas_study_locus_id"
LABEL_COL = "label_positive"
GENE_ID_COL = "gene_id"
GENE_SYMBOL_COL = "gene_symbol"

# Small interpretable genetic baseline (current/default profile).
CURRENT_BASELINE_GENETIC_FEATURES = [
    "variant_inside_gene",
    "dist_variant_to_gene_kb",
    "dist_variant_to_tss_kb",
    "dist_score_500kb_log",
    "has_qtl_evidence",
    "colocalisation_h4_max",
    "colocalisation_clpp_max",
    "qtl_study_locus_count",
    "tissue_count",
]

# Simpler quantitative-only baseline profile (no binary/count indicators by default).
QUANTITATIVE_BASELINE_GENETIC_FEATURES = [
    "dist_variant_to_gene_kb",
    "dist_variant_to_tss_kb",
    "colocalisation_h4_max",
    "colocalisation_clpp_max",
]
# Kept only for optional diagnostics in downstream tables/plots; never used as model input.
EMBEDDING_INDICATOR_FEATURE = "has_gene_embedding"
ABUNDANCE_FEATURE_CANDIDATES = [
    "brain_expression_value",
    "muscle_expression_value",
    "neuron_expression_value",
    "brain_expressed_binary",
    "muscle_expressed_binary",
    "neuron_expressed_binary",
    "max_relevant_expression",
    "mean_relevant_expression",
    "has_expression_evidence",
    "has_neuron_expression_evidence",
    "hpa_brain_expression_value",
    "hpa_muscle_expression_value",
    "hpa_neuron_expression_value",
    "hpa_brain_expressed_binary",
    "hpa_muscle_expressed_binary",
    "hpa_neuron_expressed_binary",
    "hpa_max_relevant_expression_value",
    "hpa_mean_relevant_expression_value",
    "has_hpa_expression_evidence",
    "paxdb_brain_abundance_value",
    "paxdb_muscle_abundance_value",
    "paxdb_neuron_abundance_value",
    "paxdb_brain_abundant_binary",
    "paxdb_muscle_abundant_binary",
    "paxdb_neuron_abundant_binary",
    "paxdb_max_relevant_abundance_value",
    "paxdb_mean_relevant_abundance_value",
    "has_paxdb_abundance_evidence",
]
NETWORK_FEATURE_CANDIDATES = [
    "network_prop_score",
    "has_network_prop_score",
    "network_seed_count",
]
QUANTITATIVE_ABUNDANCE_FEATURE_CANDIDATES = [
    "hpa_brain_expression_value",
    "hpa_muscle_expression_value",
]
DEFAULT_DISTANCE_WINDOW_BP = 500_000
RESIDUAL_PCA_MODE = "residual_pca"
BASELINE_THEN_PCA_MODE = "baseline_then_pca"
SCORE_MODIFIER_PCA_MODE = "score_modifier_pca"

DISTANCE_IMPUTE_SPECS = [
    ("dist_variant_to_gene_bp", "max"),
    ("dist_variant_to_gene_kb", "max"),
    ("dist_variant_to_tss_bp", "max"),
    ("dist_variant_to_tss_kb", "max"),
    ("dist_score_500kb_log", "min"),
]


def _estimate_default_window_bp(df: pd.DataFrame) -> int:
    if all(c in df.columns for c in ["candidate_window_start", "candidate_window_end", "gwas_lead_variant_position"]):
        ws = pd.to_numeric(df["candidate_window_start"], errors="coerce")
        we = pd.to_numeric(df["candidate_window_end"], errors="coerce")
        vp = pd.to_numeric(df["gwas_lead_variant_position"], errors="coerce")
        left = (vp - ws).abs()
        right = (we - vp).abs()
        vals = pd.concat([left, right], axis=0)
        vals = vals[np.isfinite(vals)]
        if not vals.empty:
            return max(int(vals.max()), 0)
    return int(DEFAULT_DISTANCE_WINDOW_BP)


def impute_distance_features_worst_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute distance missings pessimistically:
    - Distance columns: fill with max observed distance (or window fallback).
    - Distance score (higher is better): fill with min observed score (or 0.0).
    """
    if df.empty:
        return df

    out = df.copy()
    window_bp = _estimate_default_window_bp(out)
    window_kb = float(window_bp) / 1000.0
    fallback = {
        "dist_variant_to_gene_bp": float(window_bp),
        "dist_variant_to_gene_kb": float(window_kb),
        "dist_variant_to_tss_bp": float(window_bp),
        "dist_variant_to_tss_kb": float(window_kb),
        "dist_score_500kb_log": 0.0,
    }

    for col, strategy in DISTANCE_IMPUTE_SPECS:
        if col not in out.columns:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        observed = numeric[np.isfinite(numeric)]
        if strategy == "max":
            fill_value = float(observed.max()) if not observed.empty else fallback[col]
            out[col] = numeric.clip(lower=0).fillna(fill_value)
        else:
            fill_value = float(observed.min()) if not observed.empty else fallback[col]
            fill_value = float(np.clip(fill_value, 0.0, 1.0))
            out[col] = numeric.clip(lower=0.0, upper=1.0).fillna(fill_value)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a global locus-to-gene ranker with fixed "
            "lolo_gene_exclusion validation and embedding ablations."
        )
    )
    parser.add_argument(
        "--input-table",
        type=Path,
        default=DEFAULT_INPUT_TABLE,
        help="Input candidate-gene table (CSV or Parquet).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--embedding-mode",
        choices=[
            "none",
            "full",
            "pca",
            RESIDUAL_PCA_MODE,
            BASELINE_THEN_PCA_MODE,
            SCORE_MODIFIER_PCA_MODE,
            "all",
            "all_plus_residual",
        ],
        default="all",
        help=(
            "Embedding mode to run. "
            "Use 'all' for none/full/pca; "
            "use 'all_plus_residual' to also run residual_pca."
        ),
    )
    parser.add_argument(
        "--cv-mode",
        choices=["lolo_gene_exclusion"],
        default="lolo_gene_exclusion",
        help=(
            "Validation mode: only 'lolo_gene_exclusion' "
            "(LOLO + remove train rows with held-out genes)."
        ),
    )
    parser.add_argument(
        "--baseline-profile",
        choices=["current", "quantitative"],
        default="quantitative",
        help=(
            "Baseline non-embedding feature profile. "
            "'current' keeps the original mixed feature set; "
            "'quantitative' keeps a smaller mostly continuous feature set."
        ),
    )
    parser.add_argument(
        "--include-network-score",
        action="store_true",
        help="Include network_score as an extra baseline feature.",
    )
    parser.add_argument(
        "--network-score-column",
        type=str,
        default="network_score",
        help="Column name for scalar network feature.",
    )
    parser.add_argument(
        "--network-aliases-path",
        type=Path,
        default=DEFAULT_STRING_ALIASES_PATH,
        help="STRING aliases file used for fold-wise leakage-safe propagation.",
    )
    parser.add_argument(
        "--network-links-path",
        type=Path,
        default=DEFAULT_STRING_LINKS_PATH,
        help="STRING links file used for fold-wise leakage-safe propagation.",
    )
    parser.add_argument(
        "--network-cache-path",
        type=Path,
        default=DEFAULT_STRING_CACHE_PATH,
        help="Optional cache .npz for STRING gene-level transition matrix.",
    )
    parser.add_argument(
        "--network-min-combined-score",
        type=float,
        default=DEFAULT_NETWORK_MIN_COMBINED_SCORE,
        help="Minimum STRING combined_score to include an edge.",
    )
    parser.add_argument(
        "--network-alpha",
        type=float,
        default=DEFAULT_NETWORK_ALPHA,
        help="PPR alpha restart parameter in [0, 1).",
    )
    parser.add_argument(
        "--network-max-iter",
        type=int,
        default=DEFAULT_NETWORK_MAX_ITER,
        help="Maximum PPR iterations for fold-wise network score.",
    )
    parser.add_argument(
        "--network-tol",
        type=float,
        default=DEFAULT_NETWORK_TOL,
        help="PPR convergence tolerance for fold-wise network score.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=32,
        help="PCA components for embedding_mode='pca'.",
    )
    parser.add_argument(
        "--penalty",
        choices=["none", "l1", "l2", "elasticnet"],
        default="none",
        help="Logistic regression penalty. Use 'none' for no regularization.",
    )
    parser.add_argument(
        "--regularization-strength",
        type=float,
        default=0.1,
        help="Inverse regularization strength C for LogisticRegression.",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=0.5,
        help="l1_ratio only when penalty='elasticnet' (ignored otherwise).",
    )
    parser.add_argument("--max-iter", type=int, default=10000, help="Max iterations.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--top-k-contrib",
        type=int,
        default=3,
        help="Top non-zero feature contributions to export per row.",
    )
    parser.add_argument(
        "--false-positive-rank-threshold",
        type=int,
        default=3,
        help="Keep false positives with rank <= this threshold.",
    )
    parser.add_argument(
        "--run-l1-benchmark",
        action="store_true",
        help="Run L1-C benchmark for PCA mode and select final C automatically.",
    )
    parser.add_argument(
        "--l1-benchmark-c-grid",
        type=str,
        default="1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100",
        help="Comma-separated C values for L1 benchmark.",
    )
    parser.add_argument(
        "--l1-selection-pr-auc-tol",
        type=float,
        default=0.02,
        help="Absolute PR-AUC tolerance from best PR-AUC for sparsity-first L1 selection.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}")


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
        values = _to_numeric_feature(df[col]).fillna(fill_value).to_numpy(dtype=np.float64)
        arr.append(values)
    return np.column_stack(arr)


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def float_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def mean_ignore_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def std_ignore_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1))


def parse_l1_c_grid(text: str) -> List[float]:
    if text is None:
        return []
    raw = str(text).replace(";", ",").split(",")
    vals: List[float] = []
    for chunk in raw:
        s = chunk.strip()
        if not s:
            continue
        try:
            v = float(s)
        except ValueError as exc:
            raise ValueError(f"Invalid C value in --l1-benchmark-c-grid: {s}") from exc
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"C values must be finite and > 0. Found: {v}")
        vals.append(v)
    if not vals:
        raise ValueError("Empty --l1-benchmark-c-grid.")
    return sorted(set(vals))


def positive_topk_counts(pos_df: pd.DataFrame) -> Dict[str, int]:
    if pos_df.empty or "rank_within_locus" not in pos_df.columns:
        return {"n_positive_eval_rows": 0, "positive_top1_count": 0, "positive_top3_count": 0}
    ranks = pd.to_numeric(pos_df["rank_within_locus"], errors="coerce")
    valid = ranks[np.isfinite(ranks)]
    if valid.empty:
        return {"n_positive_eval_rows": 0, "positive_top1_count": 0, "positive_top3_count": 0}
    return {
        "n_positive_eval_rows": int(valid.shape[0]),
        "positive_top1_count": int((valid == 1).sum()),
        "positive_top3_count": int((valid <= 3).sum()),
    }


def select_l1_penalty_from_benchmark(bench_df: pd.DataFrame, pr_auc_tol: float) -> float:
    if bench_df.empty:
        raise ValueError("Cannot select L1 penalty from empty benchmark dataframe.")
    dd = bench_df.copy()
    dd["mean_fold_pr_auc"] = pd.to_numeric(dd["mean_fold_pr_auc"], errors="coerce")
    dd["non_zero_total_coefficients"] = pd.to_numeric(dd["non_zero_total_coefficients"], errors="coerce")
    dd["mean_recall_at_1"] = pd.to_numeric(dd["mean_recall_at_1"], errors="coerce")
    dd["mean_mrr"] = pd.to_numeric(dd["mean_mrr"], errors="coerce")
    dd["C"] = pd.to_numeric(dd["C"], errors="coerce")
    dd = dd.dropna(subset=["mean_fold_pr_auc", "non_zero_total_coefficients", "C"]).copy()
    if dd.empty:
        raise ValueError("No valid benchmark rows to select penalty.")

    best_pr_auc = float(dd["mean_fold_pr_auc"].max())
    threshold = float(best_pr_auc - float(pr_auc_tol))
    candidates = dd.loc[dd["mean_fold_pr_auc"] >= threshold].copy()
    if candidates.empty:
        candidates = dd.copy()

    candidates = candidates.sort_values(
        by=["non_zero_total_coefficients", "mean_recall_at_1", "mean_mrr", "C"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return float(candidates.iloc[0]["C"])


def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(np.asarray(x, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - float(max(threshold, 0.0)), 0.0)


def infer_has_embedding_mask(df: pd.DataFrame, embedding_cols: Sequence[str]) -> np.ndarray:
    if EMBEDDING_INDICATOR_FEATURE in df.columns:
        mask = pd.to_numeric(df[EMBEDDING_INDICATOR_FEATURE], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return mask > 0.5
    if not embedding_cols:
        return np.zeros(len(df), dtype=bool)
    emb = as_numeric_matrix(df, embedding_cols, fill_value=0.0)
    return np.any(np.abs(emb) > 1e-12, axis=1)


def fit_residual_offset_logistic(
    x_emb_pca: np.ndarray,
    baseline_linear: np.ndarray,
    y_true: np.ndarray,
    *,
    penalty: str,
    c_value: float,
    l1_ratio: float,
    max_iter: int,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, object]]:
    x = np.asarray(x_emb_pca, dtype=np.float64)
    offset = np.asarray(baseline_linear, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    n_rows, n_features = x.shape
    if n_rows == 0 or n_features == 0:
        return np.zeros(n_features, dtype=np.float64), {
            "converged": True,
            "n_iter": 0,
            "learning_rate": float("nan"),
            "regularization_alpha": 0.0,
            "regularization_l1_weight": 0.0,
            "regularization_l2_weight": 0.0,
        }

    penalty_l = str(penalty).lower()
    c_safe = float(max(float(c_value), 1e-12))
    alpha = 0.0 if penalty_l == "none" else (1.0 / c_safe)
    if penalty_l == "l1":
        l1_weight = alpha
        l2_weight = 0.0
    elif penalty_l == "l2":
        l1_weight = 0.0
        l2_weight = alpha
    elif penalty_l == "elasticnet":
        l1_weight = alpha * float(l1_ratio)
        l2_weight = alpha * float(max(0.0, 1.0 - float(l1_ratio)))
    else:
        l1_weight = 0.0
        l2_weight = 0.0

    if n_features > 0:
        svals = np.linalg.svd(x, compute_uv=False)
        max_singular = float(svals[0]) if svals.size > 0 else 0.0
    else:
        max_singular = 0.0
    lipschitz = (0.25 * (max_singular ** 2) / float(max(n_rows, 1))) + float(l2_weight)
    learning_rate = float(1.0 / max(lipschitz, 1e-8))

    gamma = np.zeros(n_features, dtype=np.float64)
    converged = False
    n_iter_done = 0
    for it in range(1, int(max_iter) + 1):
        logits = offset + x @ gamma
        probs = sigmoid_stable(logits)
        grad = (x.T @ (probs - y)) / float(max(n_rows, 1))
        if l2_weight > 0.0:
            grad = grad + (l2_weight * gamma)

        updated = gamma - (learning_rate * grad)
        if l1_weight > 0.0:
            updated = soft_threshold(updated, learning_rate * l1_weight)

        max_delta = float(np.max(np.abs(updated - gamma))) if updated.size > 0 else 0.0
        gamma = updated
        n_iter_done = it
        if max_delta <= float(tol):
            converged = True
            break

    return gamma, {
        "converged": bool(converged),
        "n_iter": int(n_iter_done),
        "learning_rate": float(learning_rate),
        "regularization_alpha": float(alpha),
        "regularization_l1_weight": float(l1_weight),
        "regularization_l2_weight": float(l2_weight),
    }


def format_top_contributions(
    contribution_row: np.ndarray,
    coef: np.ndarray,
    feature_names: Sequence[str],
    top_k: int,
) -> str:
    non_zero_coef = np.where(np.abs(coef) > 1e-12)[0]
    if non_zero_coef.size == 0:
        return ""
    vals = contribution_row[non_zero_coef]
    idx_sorted = non_zero_coef[np.argsort(-np.abs(vals))]
    chunks: List[str] = []
    for idx in idx_sorted[:top_k]:
        val = contribution_row[idx]
        if abs(val) <= 1e-12:
            continue
        chunks.append(f"{feature_names[idx]}:{val:.4f}")
    return "; ".join(chunks)


def build_model(penalty: str, c_value: float, l1_ratio: float, max_iter: int, random_state: int) -> Pipeline:
    if penalty == "none":
        clf = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=int(max_iter),
            random_state=int(random_state),
        )
    elif penalty == "l1":
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=float(c_value),
            class_weight="balanced",
            max_iter=int(max_iter),
            random_state=int(random_state),
        )
    elif penalty == "l2":
        # Explicit pure L2 regularization (no L1 / no elastic-net).
        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=float(c_value),
            class_weight="balanced",
            max_iter=int(max_iter),
            random_state=int(random_state),
        )
    else:
        clf = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=float(l1_ratio),
            solver="saga",
            C=float(c_value),
            class_weight="balanced",
            max_iter=int(max_iter),
            random_state=int(random_state),
        )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", clf),
        ]
    )


@dataclass
class ModeFeatureBuilder:
    baseline_cols: List[str]
    embedding_cols: List[str]
    mode: str
    pca_dim: int
    random_state: int

    emb_scaler: Optional[StandardScaler] = None
    pca: Optional[PCA] = None
    feature_names_: Optional[List[str]] = None
    pca_explained_variance_ratio_: Optional[np.ndarray] = None

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        x_base = as_numeric_matrix(train_df, self.baseline_cols, fill_value=0.0)

        if self.mode == "none":
            self.feature_names_ = list(self.baseline_cols)
            self.pca_explained_variance_ratio_ = None
            return x_base

        x_emb = as_numeric_matrix(train_df, self.embedding_cols, fill_value=0.0)

        if self.mode == "full":
            self.feature_names_ = list(self.baseline_cols) + list(self.embedding_cols)
            self.pca_explained_variance_ratio_ = None
            return np.column_stack([x_base, x_emb])

        n_components = min(int(self.pca_dim), x_emb.shape[1], x_emb.shape[0])
        if n_components < 1:
            raise ValueError(
                "Cannot fit PCA for embeddings: not enough rows/features. "
                f"rows={x_emb.shape[0]}, emb_features={x_emb.shape[1]}"
            )
        self.emb_scaler = StandardScaler()
        x_emb_scaled = self.emb_scaler.fit_transform(x_emb)
        self.pca = PCA(n_components=n_components, random_state=int(self.random_state))
        x_emb_pca = self.pca.fit_transform(x_emb_scaled)
        self.pca_explained_variance_ratio_ = self.pca.explained_variance_ratio_.copy()
        pca_names = [f"emb_pca_{i:03d}" for i in range(n_components)]
        self.feature_names_ = list(self.baseline_cols) + pca_names
        return np.column_stack([x_base, x_emb_pca])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_names_ is None:
            raise RuntimeError("ModeFeatureBuilder.transform called before fit_transform.")

        x_base = as_numeric_matrix(df, self.baseline_cols, fill_value=0.0)
        if self.mode == "none":
            return x_base

        x_emb = as_numeric_matrix(df, self.embedding_cols, fill_value=0.0)
        if self.mode == "full":
            return np.column_stack([x_base, x_emb])

        if self.emb_scaler is None or self.pca is None:
            raise RuntimeError("PCA mode requires fitted scaler and PCA before transform.")
        x_emb_scaled = self.emb_scaler.transform(x_emb)
        x_emb_pca = self.pca.transform(x_emb_scaled)
        return np.column_stack([x_base, x_emb_pca])


@dataclass
class FoldSpec:
    fold_index: int
    fold_id: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_rows_removed_due_to_gene_exclusion: int = 0
    n_unique_genes_removed_due_to_gene_exclusion: int = 0


@dataclass
class FoldNetworkStats:
    seed_count: int
    seed_mapped_count: int
    ppr_iterations: int
    ppr_converged: int
    train_rows_nonzero_network_score: int
    test_rows_nonzero_network_score: int
    test_rows_with_network_coverage: int


def resolve_modes(embedding_mode: str) -> List[str]:
    if embedding_mode == SCORE_MODIFIER_PCA_MODE:
        return [BASELINE_THEN_PCA_MODE]
    if embedding_mode == "all":
        return ["none", "full", "pca"]
    if embedding_mode == "all_plus_residual":
        return ["none", "full", "pca", RESIDUAL_PCA_MODE]
    return [embedding_mode]


def resolve_cv_modes(cv_mode: str) -> List[str]:
    if cv_mode != "lolo_gene_exclusion":
        raise ValueError("Only cv_mode='lolo_gene_exclusion' is supported.")
    return ["lolo_gene_exclusion"]


def resolve_available_family_columns(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def resolve_baseline_profile_columns(
    df: pd.DataFrame,
    baseline_profile: str,
    *,
    include_network_score: bool = False,
    network_score_column: str = "network_score",
) -> Dict[str, List[str]]:
    if baseline_profile == "current":
        baseline_genetic_cols = resolve_available_family_columns(df, CURRENT_BASELINE_GENETIC_FEATURES)
        abundance_cols = resolve_available_family_columns(df, ABUNDANCE_FEATURE_CANDIDATES)
        network_cols = (
            [network_score_column]
            if include_network_score and network_score_column in df.columns
            else resolve_available_family_columns(df, NETWORK_FEATURE_CANDIDATES)
        )
        return {
            "baseline_genetic_cols": baseline_genetic_cols,
            "abundance_cols": abundance_cols,
            "network_cols": network_cols,
        }

    if baseline_profile == "quantitative":
        baseline_genetic_cols = resolve_available_family_columns(df, QUANTITATIVE_BASELINE_GENETIC_FEATURES)
        abundance_cols = resolve_available_family_columns(df, QUANTITATIVE_ABUNDANCE_FEATURE_CANDIDATES)
        network_cols = [network_score_column] if include_network_score and network_score_column in df.columns else []
        return {
            "baseline_genetic_cols": baseline_genetic_cols,
            "abundance_cols": abundance_cols,
            "network_cols": network_cols,
        }

    raise ValueError(f"Unsupported baseline profile: {baseline_profile}")


def baseline_columns_for_mode(
    mode: str,
    *,
    baseline_genetic_cols: Sequence[str],
    abundance_cols: Sequence[str],
    network_cols: Sequence[str],
) -> List[str]:
    cols = list(baseline_genetic_cols)
    cols.extend(list(abundance_cols))
    cols.extend(list(network_cols))
    return cols


def apply_fold_network_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    network_model: Optional[GeneNetworkModel],
    network_score_col: str,
    network_alpha: float,
    network_max_iter: int,
    network_tol: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, FoldNetworkStats]:
    out_train = train_df.copy()
    out_test = test_df.copy()
    out_train[network_score_col] = 0.0
    out_test[network_score_col] = 0.0

    if network_model is None:
        return out_train, out_test, FoldNetworkStats(
            seed_count=0,
            seed_mapped_count=0,
            ppr_iterations=0,
            ppr_converged=0,
            train_rows_nonzero_network_score=0,
            test_rows_nonzero_network_score=0,
            test_rows_with_network_coverage=0,
        )

    seed_genes = sorted(
        {
            normalize_network_gene_symbol(g)
            for g in out_train.loc[pd.to_numeric(out_train[LABEL_COL], errors="coerce").fillna(0).astype(int) == 1, GENE_SYMBOL_COL].tolist()
            if normalize_network_gene_symbol(g) is not None
        }
    )

    ppr_scores, ppr_stats = network_model.personalized_pagerank(
        seed_genes=seed_genes,
        alpha=float(network_alpha),
        max_iter=int(network_max_iter),
        tol=float(network_tol),
    )
    train_norm = out_train[GENE_SYMBOL_COL].map(normalize_network_gene_symbol)
    test_norm = out_test[GENE_SYMBOL_COL].map(normalize_network_gene_symbol)
    out_train[network_score_col] = (
        pd.Series(network_model.score_genes(train_norm.tolist(), ppr_scores, default_score=0.0), index=out_train.index)
        .astype(float)
        .fillna(0.0)
    )
    out_test[network_score_col] = (
        pd.Series(network_model.score_genes(test_norm.tolist(), ppr_scores, default_score=0.0), index=out_test.index)
        .astype(float)
        .fillna(0.0)
    )

    graph_genes = set(network_model.gene_to_idx.keys())
    test_cov = test_norm.isin(graph_genes)

    return out_train, out_test, FoldNetworkStats(
        seed_count=int(len(seed_genes)),
        seed_mapped_count=int(ppr_stats.get("seed_mapped_count", 0.0)),
        ppr_iterations=int(ppr_stats.get("iterations", 0.0)),
        ppr_converged=int(ppr_stats.get("converged", 0.0)),
        train_rows_nonzero_network_score=int((out_train[network_score_col] > 0).sum()),
        test_rows_nonzero_network_score=int((out_test[network_score_col] > 0).sum()),
        test_rows_with_network_coverage=int(test_cov.sum()),
    )


def make_gene_group_series(df: pd.DataFrame) -> pd.Series:
    gid = df[GENE_ID_COL].fillna("").astype(str).str.strip()
    sym = df[GENE_SYMBOL_COL].fillna("").astype(str).str.strip()
    gene = gid.where(gid != "", sym)
    missing = gene == ""
    if missing.any():
        gene.loc[missing] = [f"__missing_gene_row_{i}" for i in df.index[missing].tolist()]
    return gene


def validate_input(
    df: pd.DataFrame,
    modes: Sequence[str],
    embedding_cols: Sequence[str],
    required_baseline_cols: Sequence[str],
) -> None:
    required_cols = [LOCUS_COL, LABEL_COL, GENE_SYMBOL_COL, GENE_ID_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input table missing required columns: {missing}")

    missing_baseline = [c for c in required_baseline_cols if c not in df.columns]
    if missing_baseline:
        raise ValueError(
            "Input table missing required baseline features: "
            f"{missing_baseline}"
        )

    if any(mode in {"full", "pca", RESIDUAL_PCA_MODE, BASELINE_THEN_PCA_MODE} for mode in modes) and len(embedding_cols) == 0:
        raise ValueError(
            "Embedding mode requested but no 'gene_emb_*' columns were found."
        )

    y = pd.to_numeric(df[LABEL_COL], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Column '{LABEL_COL}' contains non-numeric values.")
    if not y.isin([0, 1]).all():
        raise ValueError(f"Column '{LABEL_COL}' must be binary 0/1.")


def build_lolo_gene_exclusion_folds(df: pd.DataFrame, gene_series: pd.Series) -> List[FoldSpec]:
    loci = sorted(df[LOCUS_COL].astype(str).unique().tolist())
    folds: List[FoldSpec] = []

    for fold_idx, heldout_locus in enumerate(loci, start=1):
        test_mask = df[LOCUS_COL].astype(str) == heldout_locus
        test_idx = np.where(test_mask.to_numpy())[0]
        train_idx_full = np.where((~test_mask).to_numpy())[0]

        test_genes = set(gene_series.iloc[test_idx].astype(str).tolist())
        train_genes_full = gene_series.iloc[train_idx_full].astype(str)
        to_remove = train_genes_full.isin(test_genes).to_numpy()
        removed_rows = int(to_remove.sum())
        removed_genes = int(train_genes_full[to_remove].nunique()) if removed_rows > 0 else 0
        train_idx = train_idx_full[~to_remove]

        folds.append(
            FoldSpec(
                fold_index=fold_idx,
                fold_id=heldout_locus,
                train_idx=train_idx,
                test_idx=test_idx,
                n_rows_removed_due_to_gene_exclusion=removed_rows,
                n_unique_genes_removed_due_to_gene_exclusion=removed_genes,
            )
        )
    return folds


def build_cv_folds(df: pd.DataFrame, gene_series: pd.Series, cv_mode: str) -> List[FoldSpec]:
    if cv_mode == "lolo_gene_exclusion":
        return build_lolo_gene_exclusion_folds(df=df, gene_series=gene_series)
    raise ValueError(f"Unsupported cv_mode: {cv_mode}")


def assign_rank_within_locus(pred_df: pd.DataFrame, ranking_col: str = "predicted_score") -> pd.DataFrame:
    out = pred_df.copy()
    out["rank_within_locus"] = (
        out.groupby(["fold_index", LOCUS_COL])[ranking_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return out


def run_cv_for_mode(
    df: pd.DataFrame,
    gene_series: pd.Series,
    baseline_profile: str,
    mode: str,
    cv_mode: str,
    baseline_genetic_cols: List[str],
    baseline_cols: List[str],
    abundance_cols: List[str],
    network_cols: List[str],
    embedding_cols: List[str],
    args: argparse.Namespace,
    network_model: Optional[GeneNetworkModel] = None,
    network_score_col: str = "network_score",
) -> Dict[str, object]:
    folds = build_cv_folds(df=df, gene_series=gene_series, cv_mode=cv_mode)

    fold_rows: List[Dict[str, object]] = []
    prediction_rows: List[pd.DataFrame] = []
    positive_rank_rows: List[pd.DataFrame] = []
    fold_network_stats: List[FoldNetworkStats] = []

    for fold in folds:
        train_df = df.iloc[fold.train_idx].reset_index(drop=True)
        test_df = df.iloc[fold.test_idx].reset_index(drop=True)

        train_genes = set(gene_series.iloc[fold.train_idx].astype(str).tolist())
        test_genes = set(gene_series.iloc[fold.test_idx].astype(str).tolist())
        overlap = train_genes.intersection(test_genes)

        n_unique_genes_train = int(len(train_genes))
        n_unique_genes_test = int(len(test_genes))
        n_gene_overlap = int(len(overlap))
        overlap_fraction = float(n_gene_overlap / n_unique_genes_test) if n_unique_genes_test > 0 else float("nan")

        y_train = train_df[LABEL_COL].astype(int).to_numpy()
        y_test = test_df[LABEL_COL].astype(int).to_numpy()

        use_network_score = (network_score_col in baseline_cols) and (network_model is not None)
        if use_network_score:
            train_df, test_df, net_stats = apply_fold_network_score(
                train_df=train_df,
                test_df=test_df,
                network_model=network_model,
                network_score_col=network_score_col,
                network_alpha=float(args.network_alpha),
                network_max_iter=int(args.network_max_iter),
                network_tol=float(args.network_tol),
            )
        else:
            net_stats = FoldNetworkStats(
                seed_count=0,
                seed_mapped_count=0,
                ppr_iterations=0,
                ppr_converged=0,
                train_rows_nonzero_network_score=0,
                test_rows_nonzero_network_score=0,
                test_rows_with_network_coverage=0,
            )
        fold_network_stats.append(net_stats)

        base_fold_info = {
            "validation_mode": cv_mode,
            "embedding_mode": mode,
            "mode": mode,
            "cv_mode": cv_mode,
            "fold_index": int(fold.fold_index),
            "fold_id": str(fold.fold_id),
            "heldout_locus_id": str(fold.fold_id),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "n_unique_genes_train": n_unique_genes_train,
            "n_unique_genes_test": n_unique_genes_test,
            "n_gene_overlap_train_test": n_gene_overlap,
            "gene_overlap_fraction": float_or_nan(overlap_fraction),
            "n_rows_removed_due_to_gene_exclusion": int(fold.n_rows_removed_due_to_gene_exclusion),
            "n_unique_genes_removed_due_to_gene_exclusion": int(fold.n_unique_genes_removed_due_to_gene_exclusion),
            "network_seed_count": int(net_stats.seed_count),
            "network_seed_mapped_count": int(net_stats.seed_mapped_count),
            "network_ppr_iterations": int(net_stats.ppr_iterations),
            "network_ppr_converged": int(net_stats.ppr_converged),
            "train_rows_nonzero_network_score": int(net_stats.train_rows_nonzero_network_score),
            "test_rows_nonzero_network_score": int(net_stats.test_rows_nonzero_network_score),
            "test_rows_with_network_coverage": int(net_stats.test_rows_with_network_coverage),
        }

        if len(train_df) == 0 or len(test_df) == 0:
            fold_rows.append(
                {
                    **base_fold_info,
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                    "recall_at_1": float("nan"),
                    "recall_at_3": float("nan"),
                    "mrr": float("nan"),
                    "status": "skipped_empty_train_or_test",
                }
            )
            continue

        if np.unique(y_train).size < 2:
            fold_rows.append(
                {
                    **base_fold_info,
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                    "recall_at_1": float("nan"),
                    "recall_at_3": float("nan"),
                    "mrr": float("nan"),
                    "status": "skipped_train_single_class",
                }
            )
            continue

        feature_builder = ModeFeatureBuilder(
            baseline_cols=baseline_cols,
            embedding_cols=embedding_cols,
            mode=mode,
            pca_dim=int(args.pca_dim),
            random_state=int(args.random_state),
        )
        x_train = feature_builder.fit_transform(train_df)
        x_test = feature_builder.transform(test_df)
        feature_names = feature_builder.feature_names_ or []

        model = build_model(
            penalty=args.penalty,
            c_value=float(args.regularization_strength),
            l1_ratio=float(args.l1_ratio),
            max_iter=int(args.max_iter),
            random_state=int(args.random_state),
        )
        model.fit(x_train, y_train)
        y_score = model.predict_proba(x_test)[:, 1]
        y_linear = model.decision_function(x_test)

        pred_df = test_df.copy()
        pred_df["validation_mode"] = cv_mode
        pred_df["embedding_mode"] = mode
        pred_df["mode"] = mode
        pred_df["cv_mode"] = cv_mode
        pred_df["fold_index"] = int(fold.fold_index)
        pred_df["fold_id"] = str(fold.fold_id)
        pred_df["heldout_locus_id"] = str(fold.fold_id)
        pred_df["predicted_linear_score"] = y_linear
        pred_df["predicted_score"] = y_score
        pred_df = assign_rank_within_locus(pred_df, ranking_col="predicted_linear_score")

        scaler = model.named_steps["scaler"]
        logreg = model.named_steps["logreg"]
        coef = logreg.coef_.ravel()
        x_test_scaled = scaler.transform(x_test)
        contrib = x_test_scaled * coef.reshape(1, -1)
        pred_df["top_feature_contributions"] = [
            format_top_contributions(contrib[i], coef, feature_names, top_k=int(args.top_k_contrib))
            for i in range(contrib.shape[0])
        ]
        prediction_rows.append(pred_df)

        roc = safe_roc_auc(y_test, y_score)
        pr = safe_pr_auc(y_test, y_score)
        positive_df = pred_df.loc[pred_df[LABEL_COL].astype(int) == 1].copy()

        if positive_df.empty:
            recall_at_1 = float("nan")
            recall_at_3 = float("nan")
            mrr = float("nan")
        else:
            pos_ranks = positive_df["rank_within_locus"].astype(int).to_numpy()
            recall_at_1 = float(np.mean(pos_ranks <= 1))
            recall_at_3 = float(np.mean(pos_ranks <= 3))
            mrr = float(1.0 / pos_ranks.min())
            positive_rank_rows.append(
                positive_df[
                    [
                        "validation_mode",
                        "embedding_mode",
                        "mode",
                        "cv_mode",
                        "fold_index",
                        "fold_id",
                        "heldout_locus_id",
                        LOCUS_COL,
                        GENE_ID_COL,
                        GENE_SYMBOL_COL,
                        LABEL_COL,
                        "predicted_score",
                        "rank_within_locus",
                    ]
                ].copy()
            )

        fold_rows.append(
            {
                **base_fold_info,
                "roc_auc": float_or_nan(roc),
                "pr_auc": float_or_nan(pr),
                "recall_at_1": float_or_nan(recall_at_1),
                "recall_at_3": float_or_nan(recall_at_3),
                "mrr": float_or_nan(mrr),
                "status": "ok",
            }
        )

    if prediction_rows:
        all_predictions = pd.concat(prediction_rows, axis=0, ignore_index=True)
    else:
        all_predictions = pd.DataFrame(columns=df.columns.tolist() + ["predicted_score", "rank_within_locus"])

    all_predictions = all_predictions.sort_values(
        by=["fold_index", LOCUS_COL, "predicted_linear_score"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)

    fold_metrics_df = pd.DataFrame(fold_rows)
    positive_ranks_df = (
        pd.concat(positive_rank_rows, axis=0, ignore_index=True)
        if positive_rank_rows
        else pd.DataFrame(
            columns=[
                "validation_mode",
                "embedding_mode",
                "mode",
                "cv_mode",
                "fold_index",
                "fold_id",
                "heldout_locus_id",
                LOCUS_COL,
                GENE_ID_COL,
                GENE_SYMBOL_COL,
                LABEL_COL,
                "predicted_score",
                "rank_within_locus",
            ]
        )
    )

    top3_df = all_predictions.loc[all_predictions["rank_within_locus"] <= 3].copy()
    false_positive_df = all_predictions.loc[
        (all_predictions[LABEL_COL].astype(int) == 0)
        & (all_predictions["rank_within_locus"] <= int(args.false_positive_rank_threshold))
    ].copy()

    pooled_roc = safe_roc_auc(
        all_predictions[LABEL_COL].astype(int).to_numpy() if not all_predictions.empty else np.array([], dtype=int),
        all_predictions["predicted_score"].to_numpy() if not all_predictions.empty else np.array([], dtype=float),
    )
    pooled_pr = safe_pr_auc(
        all_predictions[LABEL_COL].astype(int).to_numpy() if not all_predictions.empty else np.array([], dtype=int),
        all_predictions["predicted_score"].to_numpy() if not all_predictions.empty else np.array([], dtype=float),
    )
    linear_all = (
        pd.to_numeric(all_predictions["predicted_linear_score"], errors="coerce").to_numpy(dtype=float)
        if not all_predictions.empty and "predicted_linear_score" in all_predictions.columns
        else np.array([], dtype=float)
    )
    score_all = (
        pd.to_numeric(all_predictions["predicted_score"], errors="coerce").to_numpy(dtype=float)
        if not all_predictions.empty and "predicted_score" in all_predictions.columns
        else np.array([], dtype=float)
    )
    valid_linear_all = np.isfinite(linear_all)
    valid_score_all = np.isfinite(score_all)
    n_validation_rows = int(len(all_predictions))
    n_validation_linear_lt_minus50 = int(np.sum(valid_linear_all & (linear_all < -50.0)))
    frac_validation_linear_lt_minus50 = (
        float(n_validation_linear_lt_minus50 / n_validation_rows) if n_validation_rows > 0 else float("nan")
    )
    n_validation_score_eq_zero = int(np.sum(valid_score_all & np.isclose(score_all, 0.0, rtol=0.0, atol=0.0)))
    frac_validation_score_eq_zero = (
        float(n_validation_score_eq_zero / n_validation_rows) if n_validation_rows > 0 else float("nan")
    )
    n_unique_validation_linear_scores = (
        int(pd.Series(linear_all[valid_linear_all]).nunique(dropna=True)) if np.any(valid_linear_all) else 0
    )
    n_unique_validation_scores = (
        int(pd.Series(score_all[valid_score_all]).nunique(dropna=True)) if np.any(valid_score_all) else 0
    )
    linear_all = (
        pd.to_numeric(all_predictions["predicted_linear_score"], errors="coerce").to_numpy(dtype=float)
        if not all_predictions.empty and "predicted_linear_score" in all_predictions.columns
        else np.array([], dtype=float)
    )
    score_all = (
        pd.to_numeric(all_predictions["predicted_score"], errors="coerce").to_numpy(dtype=float)
        if not all_predictions.empty and "predicted_score" in all_predictions.columns
        else np.array([], dtype=float)
    )
    valid_linear_all = np.isfinite(linear_all)
    valid_score_all = np.isfinite(score_all)
    n_validation_rows = int(len(all_predictions))
    n_validation_linear_lt_minus50 = int(np.sum(valid_linear_all & (linear_all < -50.0)))
    frac_validation_linear_lt_minus50 = (
        float(n_validation_linear_lt_minus50 / n_validation_rows) if n_validation_rows > 0 else float("nan")
    )
    n_validation_score_eq_zero = int(np.sum(valid_score_all & np.isclose(score_all, 0.0, rtol=0.0, atol=0.0)))
    frac_validation_score_eq_zero = (
        float(n_validation_score_eq_zero / n_validation_rows) if n_validation_rows > 0 else float("nan")
    )
    n_unique_validation_linear_scores = (
        int(pd.Series(linear_all[valid_linear_all]).nunique(dropna=True)) if np.any(valid_linear_all) else 0
    )
    n_unique_validation_scores = (
        int(pd.Series(score_all[valid_score_all]).nunique(dropna=True)) if np.any(valid_score_all) else 0
    )

    if mode == "none":
        used_embedding_feature_count = 0
    elif mode == "full":
        used_embedding_feature_count = int(len(embedding_cols))
    else:
        used_embedding_feature_count = int(min(int(args.pca_dim), len(embedding_cols), len(df)))

    if fold_network_stats:
        mean_fold_network_seed_count = mean_ignore_nan([float(s.seed_count) for s in fold_network_stats])
        mean_fold_network_seed_mapped_count = mean_ignore_nan([float(s.seed_mapped_count) for s in fold_network_stats])
        mean_fold_test_rows_nonzero_network_score = mean_ignore_nan(
            [float(s.test_rows_nonzero_network_score) for s in fold_network_stats]
        )
        mean_fold_test_rows_with_network_coverage = mean_ignore_nan(
            [float(s.test_rows_with_network_coverage) for s in fold_network_stats]
        )
    else:
        mean_fold_network_seed_count = float("nan")
        mean_fold_network_seed_mapped_count = float("nan")
        mean_fold_test_rows_nonzero_network_score = float("nan")
        mean_fold_test_rows_with_network_coverage = float("nan")

    full_fit_df = df.copy()
    if (network_model is not None) and (network_score_col in baseline_cols):
        seed_genes_full = sorted(
            {
                normalize_network_gene_symbol(g)
                for g in full_fit_df.loc[pd.to_numeric(full_fit_df[LABEL_COL], errors="coerce").fillna(0).astype(int) == 1, GENE_SYMBOL_COL].tolist()
                if normalize_network_gene_symbol(g) is not None
            }
        )
        ppr_scores_full, _ = network_model.personalized_pagerank(
            seed_genes=seed_genes_full,
            alpha=float(args.network_alpha),
            max_iter=int(args.network_max_iter),
            tol=float(args.network_tol),
        )
        full_norm = full_fit_df[GENE_SYMBOL_COL].map(normalize_network_gene_symbol)
        full_fit_df[network_score_col] = (
            pd.Series(network_model.score_genes(full_norm.tolist(), ppr_scores_full, default_score=0.0), index=full_fit_df.index)
            .astype(float)
            .fillna(0.0)
        )
    full_rows_nonzero_network_score = (
        int((pd.to_numeric(full_fit_df.get(network_score_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0) > 0).sum())
        if network_score_col in full_fit_df.columns
        else 0
    )
    full_rows_with_network_coverage = (
        int(
            full_fit_df[GENE_SYMBOL_COL]
            .map(normalize_network_gene_symbol)
            .isin(set(network_model.gene_to_idx.keys()))
            .sum()
        )
        if (network_model is not None and GENE_SYMBOL_COL in full_fit_df.columns)
        else 0
    )

    summary = {
        "validation_mode": cv_mode,
        "embedding_mode": mode,
        "mode": mode,
        "cv_mode": cv_mode,
        "baseline_profile": baseline_profile,
        "penalty": str(args.penalty),
        "regularization_strength_C": float(args.regularization_strength),
        "l1_ratio": float(args.l1_ratio),
        "n_rows": int(len(df)),
        "n_loci": int(df[LOCUS_COL].nunique()),
        "n_positive_rows": int(df[LABEL_COL].astype(int).sum()),
        "n_positive_genes": int(df.loc[df[LABEL_COL].astype(int) == 1, GENE_SYMBOL_COL].nunique()),
        "available_abundance_feature_count": int(len(abundance_cols)),
        "used_abundance_feature_count": int(len(abundance_cols)),
        "available_network_feature_count": int(len(network_cols)),
        "used_network_feature_count": int(len(network_cols)),
        "available_baseline_genetic_feature_count": int(len(baseline_genetic_cols)),
        "used_baseline_genetic_feature_count": int(len(baseline_genetic_cols)),
        "available_embedding_feature_count": int(len(embedding_cols)),
        "used_embedding_feature_count": used_embedding_feature_count,
        "network_score_feature_enabled": int((network_score_col in baseline_cols) and (network_model is not None)),
        "network_score_column": str(network_score_col),
        "mean_fold_network_seed_count": float_or_nan(mean_fold_network_seed_count),
        "mean_fold_network_seed_mapped_count": float_or_nan(mean_fold_network_seed_mapped_count),
        "mean_fold_test_rows_nonzero_network_score": float_or_nan(mean_fold_test_rows_nonzero_network_score),
        "mean_fold_test_rows_with_network_coverage": float_or_nan(mean_fold_test_rows_with_network_coverage),
        "full_rows_nonzero_network_score": int(full_rows_nonzero_network_score),
        "full_rows_with_network_coverage": int(full_rows_with_network_coverage),
        "folds_total": int(len(fold_metrics_df)),
        "folds_ok": int((fold_metrics_df["status"] == "ok").sum()) if not fold_metrics_df.empty else 0,
        "mean_n_train_rows": mean_ignore_nan(fold_metrics_df["n_train_rows"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_n_test_rows": mean_ignore_nan(fold_metrics_df["n_test_rows"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_n_unique_genes_train": mean_ignore_nan(fold_metrics_df["n_unique_genes_train"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_n_unique_genes_test": mean_ignore_nan(fold_metrics_df["n_unique_genes_test"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_n_gene_overlap_train_test": mean_ignore_nan(fold_metrics_df["n_gene_overlap_train_test"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_gene_overlap_fraction": mean_ignore_nan(fold_metrics_df["gene_overlap_fraction"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "total_rows_removed_due_to_gene_exclusion": int(fold_metrics_df["n_rows_removed_due_to_gene_exclusion"].sum()) if not fold_metrics_df.empty else 0,
        "total_unique_genes_removed_due_to_gene_exclusion": int(fold_metrics_df["n_unique_genes_removed_due_to_gene_exclusion"].sum()) if not fold_metrics_df.empty else 0,
        "mean_fold_roc_auc": mean_ignore_nan(fold_metrics_df["roc_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "std_fold_roc_auc": std_ignore_nan(fold_metrics_df["roc_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_fold_pr_auc": mean_ignore_nan(fold_metrics_df["pr_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "std_fold_pr_auc": std_ignore_nan(fold_metrics_df["pr_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_recall_at_1": mean_ignore_nan(fold_metrics_df["recall_at_1"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_recall_at_3": mean_ignore_nan(fold_metrics_df["recall_at_3"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_mrr": mean_ignore_nan(fold_metrics_df["mrr"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "pooled_roc_auc": float_or_nan(pooled_roc),
        "pooled_pr_auc": float_or_nan(pooled_pr),
        "n_validation_rows": n_validation_rows,
        "n_validation_linear_lt_minus50": n_validation_linear_lt_minus50,
        "frac_validation_linear_lt_minus50": frac_validation_linear_lt_minus50,
        "n_validation_score_eq_zero": n_validation_score_eq_zero,
        "frac_validation_score_eq_zero": frac_validation_score_eq_zero,
        "n_unique_validation_linear_scores": n_unique_validation_linear_scores,
        "n_unique_validation_scores": n_unique_validation_scores,
    }

    final_feature_builder = ModeFeatureBuilder(
        baseline_cols=baseline_cols,
        embedding_cols=embedding_cols,
        mode=mode,
        pca_dim=int(args.pca_dim),
        random_state=int(args.random_state),
    )
    x_full = final_feature_builder.fit_transform(full_fit_df)
    y_full = full_fit_df[LABEL_COL].astype(int).to_numpy()
    final_model = build_model(
        penalty=args.penalty,
        c_value=float(args.regularization_strength),
        l1_ratio=float(args.l1_ratio),
        max_iter=int(args.max_iter),
        random_state=int(args.random_state),
    )
    final_model.fit(x_full, y_full)
    final_coef = final_model.named_steps["logreg"].coef_.ravel()
    final_intercept = float(final_model.named_steps["logreg"].intercept_.ravel()[0])
    final_scaler = final_model.named_steps["scaler"]
    final_logreg = final_model.named_steps["logreg"]
    final_feature_names = final_feature_builder.feature_names_ or []

    coef_df = pd.DataFrame(
        {
            "validation_mode": cv_mode,
            "embedding_mode": mode,
            "mode": mode,
            "cv_mode": cv_mode,
            "feature": final_feature_names,
            "coefficient": final_coef,
            "abs_coefficient": np.abs(final_coef),
            "non_zero": (np.abs(final_coef) > 1e-12).astype(int),
        }
    ).sort_values("abs_coefficient", ascending=False, kind="stable")

    feature_group = []
    for name in coef_df["feature"]:
        if name in baseline_genetic_cols:
            feature_group.append("baseline")
        elif name in abundance_cols:
            feature_group.append("abundance")
        elif name in network_cols:
            feature_group.append("network")
        elif name.startswith("gene_emb_"):
            feature_group.append("embedding_raw")
        elif name.startswith("emb_pca_"):
            feature_group.append("embedding_pca")
        else:
            feature_group.append("other")
    coef_df["feature_group"] = feature_group

    scaler_df = pd.DataFrame(
        {
            "validation_mode": cv_mode,
            "embedding_mode": mode,
            "mode": mode,
            "cv_mode": cv_mode,
            "feature": final_feature_names,
            "scaler_mean": np.asarray(getattr(final_scaler, "mean_", np.zeros(len(final_feature_names))), dtype=float),
            "scaler_scale": np.asarray(getattr(final_scaler, "scale_", np.ones(len(final_feature_names))), dtype=float),
            "scaler_var": np.asarray(getattr(final_scaler, "var_", np.zeros(len(final_feature_names))), dtype=float),
        }
    )

    if mode == "pca" and final_feature_builder.pca_explained_variance_ratio_ is not None:
        evr = final_feature_builder.pca_explained_variance_ratio_
        pca_df = pd.DataFrame(
            {
                "validation_mode": cv_mode,
                "embedding_mode": mode,
                "mode": mode,
                "cv_mode": cv_mode,
                "component": [f"emb_pca_{i:03d}" for i in range(len(evr))],
                "explained_variance_ratio": evr,
                "cumulative_explained_variance_ratio": np.cumsum(evr),
            }
        )
    else:
        pca_df = pd.DataFrame(
            columns=[
                "validation_mode",
                "embedding_mode",
                "mode",
                "cv_mode",
                "component",
                "explained_variance_ratio",
                "cumulative_explained_variance_ratio",
            ]
        )

    pca_transformer_artifacts: Optional[Dict[str, np.ndarray]] = None
    if mode == "pca" and final_feature_builder.emb_scaler is not None and final_feature_builder.pca is not None:
        pca_feature_names = [f for f in final_feature_names if f.startswith("emb_pca_")]
        pca_transformer_artifacts = {
            "embedding_feature_names": np.asarray(list(embedding_cols), dtype=object),
            "pca_feature_names": np.asarray(pca_feature_names, dtype=object),
            "emb_scaler_mean": np.asarray(final_feature_builder.emb_scaler.mean_, dtype=np.float64),
            "emb_scaler_scale": np.asarray(final_feature_builder.emb_scaler.scale_, dtype=np.float64),
            "pca_components": np.asarray(final_feature_builder.pca.components_, dtype=np.float64),
            "pca_mean": np.asarray(final_feature_builder.pca.mean_, dtype=np.float64),
        }

    return {
        "summary": summary,
        "fold_metrics": fold_metrics_df,
        "positive_ranks": positive_ranks_df,
        "all_predictions": all_predictions,
        "top3_predictions": top3_df,
        "false_positives": false_positive_df,
        "coefficients": coef_df,
        "scaler_feature_stats": scaler_df,
        "model_parameters": {
            "validation_mode": cv_mode,
            "embedding_mode": mode,
            "mode": mode,
            "cv_mode": cv_mode,
            "baseline_profile": baseline_profile,
            "penalty": str(args.penalty),
            "regularization_strength_C": float(args.regularization_strength),
            "l1_ratio": float(args.l1_ratio),
            "solver": str(getattr(final_logreg, "solver", "")),
            "class_weight": str(getattr(final_logreg, "class_weight", "")),
            "max_iter": int(getattr(final_logreg, "max_iter", args.max_iter)),
            "model_intercept": final_intercept,
            "network_score_feature_enabled": int((network_score_col in baseline_cols) and (network_model is not None)),
            "network_score_column": str(network_score_col),
            "network_alpha": float(args.network_alpha),
            "network_max_iter": int(args.network_max_iter),
            "network_tol": float(args.network_tol),
            "network_min_combined_score": float(args.network_min_combined_score),
        },
        "feature_lists": {
            "baseline_profile": baseline_profile,
            "baseline_genetic_features": list(baseline_genetic_cols),
            "abundance_features": list(abundance_cols),
            "network_features": list(network_cols),
            "network_score_column": str(network_score_col),
            "baseline_features_used_in_mode": list(baseline_cols),
            "embedding_features_available": list(embedding_cols),
            "embedding_mode": mode,
            "embedding_features_used_count": int(used_embedding_feature_count),
        },
        "pca_transformer_artifacts": pca_transformer_artifacts,
        "pca_explained_variance": pca_df,
    }


def run_cv_for_residual_pca(
    df: pd.DataFrame,
    gene_series: pd.Series,
    baseline_profile: str,
    mode: str,
    cv_mode: str,
    baseline_genetic_cols: List[str],
    baseline_cols: List[str],
    abundance_cols: List[str],
    network_cols: List[str],
    embedding_cols: List[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    folds = build_cv_folds(df=df, gene_series=gene_series, cv_mode=cv_mode)

    fold_rows: List[Dict[str, object]] = []
    prediction_rows: List[pd.DataFrame] = []
    positive_rank_rows: List[pd.DataFrame] = []
    residual_fit_rows: List[Dict[str, object]] = []

    for fold in folds:
        train_df = df.iloc[fold.train_idx].reset_index(drop=True)
        test_df = df.iloc[fold.test_idx].reset_index(drop=True)

        train_genes = set(gene_series.iloc[fold.train_idx].astype(str).tolist())
        test_genes = set(gene_series.iloc[fold.test_idx].astype(str).tolist())
        overlap = train_genes.intersection(test_genes)

        n_unique_genes_train = int(len(train_genes))
        n_unique_genes_test = int(len(test_genes))
        n_gene_overlap = int(len(overlap))
        overlap_fraction = float(n_gene_overlap / n_unique_genes_test) if n_unique_genes_test > 0 else float("nan")

        y_train = train_df[LABEL_COL].astype(int).to_numpy()
        y_test = test_df[LABEL_COL].astype(int).to_numpy()

        base_fold_info = {
            "validation_mode": cv_mode,
            "embedding_mode": mode,
            "mode": mode,
            "cv_mode": cv_mode,
            "fold_index": int(fold.fold_index),
            "fold_id": str(fold.fold_id),
            "heldout_locus_id": str(fold.fold_id),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "n_unique_genes_train": n_unique_genes_train,
            "n_unique_genes_test": n_unique_genes_test,
            "n_gene_overlap_train_test": n_gene_overlap,
            "gene_overlap_fraction": float_or_nan(overlap_fraction),
            "n_rows_removed_due_to_gene_exclusion": int(fold.n_rows_removed_due_to_gene_exclusion),
            "n_unique_genes_removed_due_to_gene_exclusion": int(fold.n_unique_genes_removed_due_to_gene_exclusion),
        }

        if len(train_df) == 0 or len(test_df) == 0:
            fold_rows.append(
                {
                    **base_fold_info,
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                    "recall_at_1": float("nan"),
                    "recall_at_3": float("nan"),
                    "mrr": float("nan"),
                    "status": "skipped_empty_train_or_test",
                }
            )
            continue

        if np.unique(y_train).size < 2:
            fold_rows.append(
                {
                    **base_fold_info,
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                    "recall_at_1": float("nan"),
                    "recall_at_3": float("nan"),
                    "mrr": float("nan"),
                    "status": "skipped_train_single_class",
                }
            )
            continue

        x_train_base = as_numeric_matrix(train_df, baseline_cols, fill_value=0.0)
        x_test_base = as_numeric_matrix(test_df, baseline_cols, fill_value=0.0)
        baseline_model = build_model(
            penalty=args.penalty,
            c_value=float(args.regularization_strength),
            l1_ratio=float(args.l1_ratio),
            max_iter=int(args.max_iter),
            random_state=int(args.random_state),
        )
        baseline_model.fit(x_train_base, y_train)
        baseline_linear_train = baseline_model.decision_function(x_train_base)
        baseline_linear_test = baseline_model.decision_function(x_test_base)
        baseline_score_test = baseline_model.predict_proba(x_test_base)[:, 1]

        emb_builder = ModeFeatureBuilder(
            baseline_cols=[],
            embedding_cols=embedding_cols,
            mode="pca",
            pca_dim=int(args.pca_dim),
            random_state=int(args.random_state),
        )
        x_train_pca = emb_builder.fit_transform(train_df)
        x_test_pca = emb_builder.transform(test_df)
        pca_feature_names = emb_builder.feature_names_ or []

        has_emb_train = infer_has_embedding_mask(train_df, embedding_cols)
        has_emb_test = infer_has_embedding_mask(test_df, embedding_cols)
        if x_train_pca.size > 0:
            x_train_pca = np.asarray(x_train_pca, dtype=np.float64)
            x_test_pca = np.asarray(x_test_pca, dtype=np.float64)
            x_train_pca[~has_emb_train, :] = 0.0
            x_test_pca[~has_emb_test, :] = 0.0

        residual_coef, residual_stats = fit_residual_offset_logistic(
            x_emb_pca=x_train_pca,
            baseline_linear=baseline_linear_train,
            y_true=y_train,
            penalty=str(args.penalty),
            c_value=float(args.regularization_strength),
            l1_ratio=float(args.l1_ratio),
            max_iter=int(args.max_iter),
        )
        residual_linear_test = (
            (x_test_pca @ residual_coef.reshape(-1, 1)).ravel() if residual_coef.size > 0 else np.zeros(len(test_df))
        )
        residual_linear_test[~has_emb_test] = 0.0
        final_linear_test = baseline_linear_test + residual_linear_test
        final_score_test = sigmoid_stable(final_linear_test)

        pred_df = test_df.copy()
        pred_df["validation_mode"] = cv_mode
        pred_df["embedding_mode"] = mode
        pred_df["mode"] = mode
        pred_df["cv_mode"] = cv_mode
        pred_df["fold_index"] = int(fold.fold_index)
        pred_df["fold_id"] = str(fold.fold_id)
        pred_df["heldout_locus_id"] = str(fold.fold_id)
        pred_df["has_embedding_for_residual"] = has_emb_test.astype(int)
        pred_df["baseline_predicted_linear_score"] = baseline_linear_test
        pred_df["baseline_predicted_score"] = baseline_score_test
        pred_df["embedding_residual_linear_score"] = residual_linear_test
        pred_df["final_predicted_linear_score"] = final_linear_test
        pred_df["final_predicted_score"] = final_score_test
        pred_df["predicted_linear_score"] = final_linear_test
        pred_df["predicted_score"] = final_score_test
        pred_df = assign_rank_within_locus(pred_df, ranking_col="predicted_linear_score")

        baseline_scaler = baseline_model.named_steps["scaler"]
        baseline_logreg = baseline_model.named_steps["logreg"]
        baseline_coef = baseline_logreg.coef_.ravel()
        z_test_base = baseline_scaler.transform(x_test_base)
        contrib_base = z_test_base * baseline_coef.reshape(1, -1) if baseline_coef.size > 0 else np.zeros((len(test_df), 0))
        contrib_resid = x_test_pca * residual_coef.reshape(1, -1) if residual_coef.size > 0 else np.zeros((len(test_df), 0))
        contrib_all = np.column_stack([contrib_base, contrib_resid])
        contrib_names = list(baseline_cols) + list(pca_feature_names)
        pred_df["top_feature_contributions"] = [
            format_top_contributions(contrib_all[i], np.ones(contrib_all.shape[1], dtype=float), contrib_names, top_k=int(args.top_k_contrib))
            for i in range(contrib_all.shape[0])
        ]
        prediction_rows.append(pred_df)

        roc = safe_roc_auc(y_test, final_score_test)
        pr = safe_pr_auc(y_test, final_score_test)
        positive_df = pred_df.loc[pred_df[LABEL_COL].astype(int) == 1].copy()
        if positive_df.empty:
            recall_at_1 = float("nan")
            recall_at_3 = float("nan")
            mrr = float("nan")
        else:
            pos_ranks = positive_df["rank_within_locus"].astype(int).to_numpy()
            recall_at_1 = float(np.mean(pos_ranks <= 1))
            recall_at_3 = float(np.mean(pos_ranks <= 3))
            mrr = float(1.0 / pos_ranks.min())
            positive_rank_rows.append(
                positive_df[
                    [
                        "validation_mode",
                        "embedding_mode",
                        "mode",
                        "cv_mode",
                        "fold_index",
                        "fold_id",
                        "heldout_locus_id",
                        LOCUS_COL,
                        GENE_ID_COL,
                        GENE_SYMBOL_COL,
                        LABEL_COL,
                        "predicted_score",
                        "rank_within_locus",
                    ]
                ].copy()
            )

        fold_rows.append(
            {
                **base_fold_info,
                "roc_auc": float_or_nan(roc),
                "pr_auc": float_or_nan(pr),
                "recall_at_1": float_or_nan(recall_at_1),
                "recall_at_3": float_or_nan(recall_at_3),
                "mrr": float_or_nan(mrr),
                "status": "ok",
                "n_nonzero_residual_pca_coef": int(np.sum(np.abs(residual_coef) > 1e-12)),
                "n_test_rows_missing_embedding": int((~has_emb_test).sum()),
                "max_abs_residual_on_missing_embedding": float(
                    np.max(np.abs(residual_linear_test[~has_emb_test])) if np.any(~has_emb_test) else 0.0
                ),
            }
        )
        residual_fit_rows.append(
            {
                "validation_mode": cv_mode,
                "embedding_mode": mode,
                "mode": mode,
                "cv_mode": cv_mode,
                "fold_index": int(fold.fold_index),
                "fold_id": str(fold.fold_id),
                "heldout_locus_id": str(fold.fold_id),
                "n_residual_features": int(len(pca_feature_names)),
                "n_nonzero_residual_coefficients": int(np.sum(np.abs(residual_coef) > 1e-12)),
                "n_train_rows_missing_embedding": int((~has_emb_train).sum()),
                "n_test_rows_missing_embedding": int((~has_emb_test).sum()),
                "max_abs_residual_on_missing_embedding_test": float(
                    np.max(np.abs(residual_linear_test[~has_emb_test])) if np.any(~has_emb_test) else 0.0
                ),
                **residual_stats,
            }
        )

    if prediction_rows:
        all_predictions = pd.concat(prediction_rows, axis=0, ignore_index=True)
    else:
        all_predictions = pd.DataFrame(columns=df.columns.tolist() + ["predicted_score", "rank_within_locus"])
    all_predictions = all_predictions.sort_values(
        by=["fold_index", LOCUS_COL, "predicted_linear_score"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)

    fold_metrics_df = pd.DataFrame(fold_rows)
    residual_fit_df = pd.DataFrame(residual_fit_rows)
    positive_ranks_df = (
        pd.concat(positive_rank_rows, axis=0, ignore_index=True)
        if positive_rank_rows
        else pd.DataFrame(
            columns=[
                "validation_mode",
                "embedding_mode",
                "mode",
                "cv_mode",
                "fold_index",
                "fold_id",
                "heldout_locus_id",
                LOCUS_COL,
                GENE_ID_COL,
                GENE_SYMBOL_COL,
                LABEL_COL,
                "predicted_score",
                "rank_within_locus",
            ]
        )
    )

    top3_df = all_predictions.loc[all_predictions["rank_within_locus"] <= 3].copy()
    false_positive_df = all_predictions.loc[
        (all_predictions[LABEL_COL].astype(int) == 0)
        & (all_predictions["rank_within_locus"] <= int(args.false_positive_rank_threshold))
    ].copy()

    pooled_roc = safe_roc_auc(
        all_predictions[LABEL_COL].astype(int).to_numpy() if not all_predictions.empty else np.array([], dtype=int),
        all_predictions["predicted_score"].to_numpy() if not all_predictions.empty else np.array([], dtype=float),
    )
    pooled_pr = safe_pr_auc(
        all_predictions[LABEL_COL].astype(int).to_numpy() if not all_predictions.empty else np.array([], dtype=int),
        all_predictions["predicted_score"].to_numpy() if not all_predictions.empty else np.array([], dtype=float),
    )
    linear_all = (
        pd.to_numeric(all_predictions["predicted_linear_score"], errors="coerce").to_numpy(dtype=float)
        if not all_predictions.empty and "predicted_linear_score" in all_predictions.columns
        else np.array([], dtype=float)
    )
    score_all = (
        pd.to_numeric(all_predictions["predicted_score"], errors="coerce").to_numpy(dtype=float)
        if not all_predictions.empty and "predicted_score" in all_predictions.columns
        else np.array([], dtype=float)
    )
    valid_linear_all = np.isfinite(linear_all)
    valid_score_all = np.isfinite(score_all)
    n_validation_rows = int(len(all_predictions))
    n_validation_linear_lt_minus50 = int(np.sum(valid_linear_all & (linear_all < -50.0)))
    frac_validation_linear_lt_minus50 = (
        float(n_validation_linear_lt_minus50 / n_validation_rows) if n_validation_rows > 0 else float("nan")
    )
    n_validation_score_eq_zero = int(np.sum(valid_score_all & np.isclose(score_all, 0.0, rtol=0.0, atol=0.0)))
    frac_validation_score_eq_zero = (
        float(n_validation_score_eq_zero / n_validation_rows) if n_validation_rows > 0 else float("nan")
    )
    n_unique_validation_linear_scores = (
        int(pd.Series(linear_all[valid_linear_all]).nunique(dropna=True)) if np.any(valid_linear_all) else 0
    )
    n_unique_validation_scores = (
        int(pd.Series(score_all[valid_score_all]).nunique(dropna=True)) if np.any(valid_score_all) else 0
    )

    used_embedding_feature_count = int(min(int(args.pca_dim), len(embedding_cols), len(df)))
    max_abs_resid_missing = 0.0
    if (
        not all_predictions.empty
        and "has_embedding_for_residual" in all_predictions.columns
        and "embedding_residual_linear_score" in all_predictions.columns
    ):
        missing_mask_all = pd.to_numeric(all_predictions["has_embedding_for_residual"], errors="coerce").fillna(0).to_numpy(dtype=float) < 0.5
        resid_all = pd.to_numeric(all_predictions["embedding_residual_linear_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if np.any(missing_mask_all):
            max_abs_resid_missing = float(np.max(np.abs(resid_all[missing_mask_all])))

    summary = {
        "validation_mode": cv_mode,
        "embedding_mode": mode,
        "mode": mode,
        "cv_mode": cv_mode,
        "baseline_profile": baseline_profile,
        "penalty": str(args.penalty),
        "regularization_strength_C": float(args.regularization_strength),
        "l1_ratio": float(args.l1_ratio),
        "n_rows": int(len(df)),
        "n_loci": int(df[LOCUS_COL].nunique()),
        "n_positive_rows": int(df[LABEL_COL].astype(int).sum()),
        "n_positive_genes": int(df.loc[df[LABEL_COL].astype(int) == 1, GENE_SYMBOL_COL].nunique()),
        "available_abundance_feature_count": int(len(abundance_cols)),
        "used_abundance_feature_count": int(len(abundance_cols)),
        "available_network_feature_count": int(len(network_cols)),
        "used_network_feature_count": int(len(network_cols)),
        "available_baseline_genetic_feature_count": int(len(baseline_genetic_cols)),
        "used_baseline_genetic_feature_count": int(len(baseline_genetic_cols)),
        "available_embedding_feature_count": int(len(embedding_cols)),
        "used_embedding_feature_count": used_embedding_feature_count,
        "folds_total": int(len(fold_metrics_df)),
        "folds_ok": int((fold_metrics_df["status"] == "ok").sum()) if not fold_metrics_df.empty else 0,
        "mean_n_train_rows": mean_ignore_nan(fold_metrics_df["n_train_rows"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_n_test_rows": mean_ignore_nan(fold_metrics_df["n_test_rows"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_n_unique_genes_train": mean_ignore_nan(fold_metrics_df["n_unique_genes_train"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_n_unique_genes_test": mean_ignore_nan(fold_metrics_df["n_unique_genes_test"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_n_gene_overlap_train_test": mean_ignore_nan(fold_metrics_df["n_gene_overlap_train_test"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_gene_overlap_fraction": mean_ignore_nan(fold_metrics_df["gene_overlap_fraction"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "total_rows_removed_due_to_gene_exclusion": int(fold_metrics_df["n_rows_removed_due_to_gene_exclusion"].sum()) if not fold_metrics_df.empty else 0,
        "total_unique_genes_removed_due_to_gene_exclusion": int(fold_metrics_df["n_unique_genes_removed_due_to_gene_exclusion"].sum()) if not fold_metrics_df.empty else 0,
        "mean_fold_roc_auc": mean_ignore_nan(fold_metrics_df["roc_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "std_fold_roc_auc": std_ignore_nan(fold_metrics_df["roc_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_fold_pr_auc": mean_ignore_nan(fold_metrics_df["pr_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "std_fold_pr_auc": std_ignore_nan(fold_metrics_df["pr_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_recall_at_1": mean_ignore_nan(fold_metrics_df["recall_at_1"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_recall_at_3": mean_ignore_nan(fold_metrics_df["recall_at_3"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_mrr": mean_ignore_nan(fold_metrics_df["mrr"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "pooled_roc_auc": float_or_nan(pooled_roc),
        "pooled_pr_auc": float_or_nan(pooled_pr),
        "n_validation_rows": n_validation_rows,
        "n_validation_linear_lt_minus50": n_validation_linear_lt_minus50,
        "frac_validation_linear_lt_minus50": frac_validation_linear_lt_minus50,
        "n_validation_score_eq_zero": n_validation_score_eq_zero,
        "frac_validation_score_eq_zero": frac_validation_score_eq_zero,
        "n_unique_validation_linear_scores": n_unique_validation_linear_scores,
        "n_unique_validation_scores": n_unique_validation_scores,
        "residual_missing_embedding_zero_enforced": 1,
        "max_abs_residual_on_missing_embedding_validation": float(max_abs_resid_missing),
        "mean_nonzero_residual_pca_coef": mean_ignore_nan(
            pd.to_numeric(fold_metrics_df.get("n_nonzero_residual_pca_coef", pd.Series(dtype=float)), errors="coerce").tolist()
        )
        if not fold_metrics_df.empty
        else float("nan"),
    }

    x_full_base = as_numeric_matrix(df, baseline_cols, fill_value=0.0)
    y_full = df[LABEL_COL].astype(int).to_numpy()
    baseline_model_full = build_model(
        penalty=args.penalty,
        c_value=float(args.regularization_strength),
        l1_ratio=float(args.l1_ratio),
        max_iter=int(args.max_iter),
        random_state=int(args.random_state),
    )
    baseline_model_full.fit(x_full_base, y_full)
    baseline_linear_full = baseline_model_full.decision_function(x_full_base)
    baseline_scaler_full = baseline_model_full.named_steps["scaler"]
    baseline_logreg_full = baseline_model_full.named_steps["logreg"]
    baseline_coef_full = baseline_logreg_full.coef_.ravel()
    baseline_intercept_full = float(baseline_logreg_full.intercept_.ravel()[0])

    emb_builder_full = ModeFeatureBuilder(
        baseline_cols=[],
        embedding_cols=embedding_cols,
        mode="pca",
        pca_dim=int(args.pca_dim),
        random_state=int(args.random_state),
    )
    x_full_pca = emb_builder_full.fit_transform(df)
    pca_feature_names_full = emb_builder_full.feature_names_ or []
    has_emb_full = infer_has_embedding_mask(df, embedding_cols)
    if x_full_pca.size > 0:
        x_full_pca = np.asarray(x_full_pca, dtype=np.float64)
        x_full_pca[~has_emb_full, :] = 0.0

    residual_coef_full, residual_fit_full = fit_residual_offset_logistic(
        x_emb_pca=x_full_pca,
        baseline_linear=baseline_linear_full,
        y_true=y_full,
        penalty=str(args.penalty),
        c_value=float(args.regularization_strength),
        l1_ratio=float(args.l1_ratio),
        max_iter=int(args.max_iter),
    )
    residual_linear_full = (
        (x_full_pca @ residual_coef_full.reshape(-1, 1)).ravel() if residual_coef_full.size > 0 else np.zeros(len(df))
    )
    residual_linear_full[~has_emb_full] = 0.0
    final_linear_full = baseline_linear_full + residual_linear_full
    final_score_full = sigmoid_stable(final_linear_full)

    baseline_rows = pd.DataFrame(
        {
            "validation_mode": cv_mode,
            "embedding_mode": mode,
            "mode": mode,
            "cv_mode": cv_mode,
            "component": "baseline_logit",
            "feature": list(baseline_cols),
            "coefficient": baseline_coef_full,
            "abs_coefficient": np.abs(baseline_coef_full),
            "non_zero": (np.abs(baseline_coef_full) > 1e-12).astype(int),
        }
    )
    residual_rows = pd.DataFrame(
        {
            "validation_mode": cv_mode,
            "embedding_mode": mode,
            "mode": mode,
            "cv_mode": cv_mode,
            "component": "embedding_residual",
            "feature": list(pca_feature_names_full),
            "coefficient": residual_coef_full,
            "abs_coefficient": np.abs(residual_coef_full),
            "non_zero": (np.abs(residual_coef_full) > 1e-12).astype(int),
        }
    )
    coef_df = pd.concat([baseline_rows, residual_rows], axis=0, ignore_index=True)
    feature_group: List[str] = []
    for name in coef_df["feature"].astype(str).tolist():
        if name in baseline_genetic_cols:
            feature_group.append("baseline")
        elif name in abundance_cols:
            feature_group.append("abundance")
        elif name in network_cols:
            feature_group.append("network")
        elif name.startswith("emb_pca_"):
            feature_group.append("embedding_pca")
        else:
            feature_group.append("other")
    coef_df["feature_group"] = feature_group
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False, kind="stable").reset_index(drop=True)

    scaler_df = pd.DataFrame(
        {
            "validation_mode": cv_mode,
            "embedding_mode": mode,
            "mode": mode,
            "cv_mode": cv_mode,
            "component": "baseline_logit",
            "feature": list(baseline_cols),
            "scaler_mean": np.asarray(getattr(baseline_scaler_full, "mean_", np.zeros(len(baseline_cols))), dtype=float),
            "scaler_scale": np.asarray(getattr(baseline_scaler_full, "scale_", np.ones(len(baseline_cols))), dtype=float),
            "scaler_var": np.asarray(getattr(baseline_scaler_full, "var_", np.zeros(len(baseline_cols))), dtype=float),
        }
    )

    pca_df = pd.DataFrame(
        columns=[
            "validation_mode",
            "embedding_mode",
            "mode",
            "cv_mode",
            "component",
            "explained_variance_ratio",
            "cumulative_explained_variance_ratio",
        ]
    )
    pca_transformer_artifacts: Optional[Dict[str, np.ndarray]] = None
    if emb_builder_full.pca_explained_variance_ratio_ is not None and emb_builder_full.emb_scaler is not None and emb_builder_full.pca is not None:
        evr = emb_builder_full.pca_explained_variance_ratio_
        pca_df = pd.DataFrame(
            {
                "validation_mode": cv_mode,
                "embedding_mode": mode,
                "mode": mode,
                "cv_mode": cv_mode,
                "component": [f"emb_pca_{i:03d}" for i in range(len(evr))],
                "explained_variance_ratio": evr,
                "cumulative_explained_variance_ratio": np.cumsum(evr),
            }
        )
        pca_transformer_artifacts = {
            "embedding_feature_names": np.asarray(list(embedding_cols), dtype=object),
            "pca_feature_names": np.asarray(list(pca_feature_names_full), dtype=object),
            "emb_scaler_mean": np.asarray(emb_builder_full.emb_scaler.mean_, dtype=np.float64),
            "emb_scaler_scale": np.asarray(emb_builder_full.emb_scaler.scale_, dtype=np.float64),
            "pca_components": np.asarray(emb_builder_full.pca.components_, dtype=np.float64),
            "pca_mean": np.asarray(emb_builder_full.pca.mean_, dtype=np.float64),
        }

    residual_pca_artifacts: Optional[Dict[str, np.ndarray]] = None
    if pca_transformer_artifacts is not None:
        residual_pca_artifacts = {
            "baseline_feature_names": np.asarray(list(baseline_cols), dtype=object),
            "baseline_scaler_mean": np.asarray(getattr(baseline_scaler_full, "mean_", np.zeros(len(baseline_cols))), dtype=np.float64),
            "baseline_scaler_scale": np.asarray(getattr(baseline_scaler_full, "scale_", np.ones(len(baseline_cols))), dtype=np.float64),
            "baseline_coefficients": np.asarray(baseline_coef_full, dtype=np.float64),
            "baseline_intercept": np.asarray([baseline_intercept_full], dtype=np.float64),
            "residual_pca_feature_names": np.asarray(list(pca_feature_names_full), dtype=object),
            "residual_coefficients": np.asarray(residual_coef_full, dtype=np.float64),
            "embedding_feature_names": np.asarray(list(embedding_cols), dtype=object),
            "emb_scaler_mean": np.asarray(pca_transformer_artifacts["emb_scaler_mean"], dtype=np.float64),
            "emb_scaler_scale": np.asarray(pca_transformer_artifacts["emb_scaler_scale"], dtype=np.float64),
            "pca_components": np.asarray(pca_transformer_artifacts["pca_components"], dtype=np.float64),
            "pca_mean": np.asarray(pca_transformer_artifacts["pca_mean"], dtype=np.float64),
            "has_embedding_feature_name": np.asarray([EMBEDDING_INDICATOR_FEATURE], dtype=object),
            "missing_embedding_residual_zero": np.asarray([1], dtype=np.int8),
        }

    model_parameters = {
        "validation_mode": cv_mode,
        "embedding_mode": mode,
        "mode": mode,
        "cv_mode": cv_mode,
        "baseline_profile": baseline_profile,
        "penalty": str(args.penalty),
        "regularization_strength_C": float(args.regularization_strength),
        "l1_ratio": float(args.l1_ratio),
        "baseline_solver": str(getattr(baseline_logreg_full, "solver", "")),
        "baseline_class_weight": str(getattr(baseline_logreg_full, "class_weight", "")),
        "baseline_model_intercept": float(baseline_intercept_full),
        "embedding_residual_intercept": 0.0,
        "final_score_formula": "sigmoid(baseline_linear + embedding_residual_linear)",
        "missing_embedding_behavior": "embedding_residual_linear_score is forced to 0 when has_embedding=0",
        "residual_fit_converged_full_data": bool(residual_fit_full.get("converged", False)),
        "residual_fit_n_iter_full_data": int(residual_fit_full.get("n_iter", 0)),
    }

    residual_contrib_df = pd.DataFrame(
        {
            "baseline_predicted_linear_score": baseline_linear_full,
            "embedding_residual_linear_score": residual_linear_full,
            "final_predicted_linear_score": final_linear_full,
            "final_predicted_score": final_score_full,
            "has_embedding_for_residual": has_emb_full.astype(int),
        }
    )

    return {
        "summary": summary,
        "fold_metrics": fold_metrics_df,
        "positive_ranks": positive_ranks_df,
        "all_predictions": all_predictions,
        "top3_predictions": top3_df,
        "false_positives": false_positive_df,
        "coefficients": coef_df,
        "scaler_feature_stats": scaler_df,
        "model_parameters": model_parameters,
        "feature_lists": {
            "baseline_profile": baseline_profile,
            "baseline_genetic_features": list(baseline_genetic_cols),
            "abundance_features": list(abundance_cols),
            "network_features": list(network_cols),
            "baseline_features_used_in_mode": list(baseline_cols),
            "embedding_features_available": list(embedding_cols),
            "embedding_mode": mode,
            "embedding_features_used_count": int(used_embedding_feature_count),
            "residual_pca_feature_names": list(pca_feature_names_full),
        },
        "pca_transformer_artifacts": pca_transformer_artifacts,
        "pca_explained_variance": pca_df,
        "residual_pca_artifacts": residual_pca_artifacts,
        "residual_fit_diagnostics": residual_fit_df,
        "residual_contribution_table": residual_contrib_df,
    }


def run_cv_for_baseline_then_pca(
    df: pd.DataFrame,
    gene_series: pd.Series,
    baseline_profile: str,
    mode: str,
    cv_mode: str,
    baseline_genetic_cols: List[str],
    baseline_cols: List[str],
    abundance_cols: List[str],
    network_cols: List[str],
    embedding_cols: List[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    folds = build_cv_folds(df=df, gene_series=gene_series, cv_mode=cv_mode)

    fold_rows: List[Dict[str, object]] = []
    prediction_rows: List[pd.DataFrame] = []
    positive_rank_rows: List[pd.DataFrame] = []

    for fold in folds:
        train_df = df.iloc[fold.train_idx].reset_index(drop=True)
        test_df = df.iloc[fold.test_idx].reset_index(drop=True)

        train_genes = set(gene_series.iloc[fold.train_idx].astype(str).tolist())
        test_genes = set(gene_series.iloc[fold.test_idx].astype(str).tolist())
        overlap = train_genes.intersection(test_genes)

        n_unique_genes_train = int(len(train_genes))
        n_unique_genes_test = int(len(test_genes))
        n_gene_overlap = int(len(overlap))
        overlap_fraction = float(n_gene_overlap / n_unique_genes_test) if n_unique_genes_test > 0 else float("nan")

        y_train = train_df[LABEL_COL].astype(int).to_numpy()
        y_test = test_df[LABEL_COL].astype(int).to_numpy()

        base_fold_info = {
            "validation_mode": cv_mode,
            "embedding_mode": mode,
            "mode": mode,
            "cv_mode": cv_mode,
            "fold_index": int(fold.fold_index),
            "fold_id": str(fold.fold_id),
            "heldout_locus_id": str(fold.fold_id),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "n_unique_genes_train": n_unique_genes_train,
            "n_unique_genes_test": n_unique_genes_test,
            "n_gene_overlap_train_test": n_gene_overlap,
            "gene_overlap_fraction": float_or_nan(overlap_fraction),
            "n_rows_removed_due_to_gene_exclusion": int(fold.n_rows_removed_due_to_gene_exclusion),
            "n_unique_genes_removed_due_to_gene_exclusion": int(fold.n_unique_genes_removed_due_to_gene_exclusion),
        }

        if len(train_df) == 0 or len(test_df) == 0:
            fold_rows.append(
                {
                    **base_fold_info,
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                    "recall_at_1": float("nan"),
                    "recall_at_3": float("nan"),
                    "mrr": float("nan"),
                    "status": "skipped_empty_train_or_test",
                }
            )
            continue

        if np.unique(y_train).size < 2:
            fold_rows.append(
                {
                    **base_fold_info,
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                    "recall_at_1": float("nan"),
                    "recall_at_3": float("nan"),
                    "mrr": float("nan"),
                    "status": "skipped_train_single_class",
                }
            )
            continue

        # Stage 1: baseline-only model that generates score_baseline.
        x_train_base = as_numeric_matrix(train_df, baseline_cols, fill_value=0.0)
        x_test_base = as_numeric_matrix(test_df, baseline_cols, fill_value=0.0)
        baseline_model = build_model(
            penalty=args.penalty,
            c_value=float(args.regularization_strength),
            l1_ratio=float(args.l1_ratio),
            max_iter=int(args.max_iter),
            random_state=int(args.random_state),
        )
        baseline_model.fit(x_train_base, y_train)
        baseline_linear_train = baseline_model.decision_function(x_train_base)
        baseline_linear_test = baseline_model.decision_function(x_test_base)
        baseline_score_train = baseline_model.predict_proba(x_train_base)[:, 1]
        baseline_score_test = baseline_model.predict_proba(x_test_base)[:, 1]

        # Stage 2: learn embedding-driven modifier from score_baseline + PCA embeddings.
        emb_builder = ModeFeatureBuilder(
            baseline_cols=[],
            embedding_cols=embedding_cols,
            mode="pca",
            pca_dim=int(args.pca_dim),
            random_state=int(args.random_state),
        )
        x_train_pca = emb_builder.fit_transform(train_df)
        x_test_pca = emb_builder.transform(test_df)
        pca_feature_names = emb_builder.feature_names_ or []
        stage2_feature_names = ["score_baseline"] + list(pca_feature_names)

        x_train_stage2 = np.column_stack([baseline_score_train.reshape(-1, 1), x_train_pca])
        x_test_stage2 = np.column_stack([baseline_score_test.reshape(-1, 1), x_test_pca])
        stage2_model = build_model(
            penalty=args.penalty,
            c_value=float(args.regularization_strength),
            l1_ratio=float(args.l1_ratio),
            max_iter=int(args.max_iter),
            random_state=int(args.random_state),
        )
        stage2_model.fit(x_train_stage2, y_train)
        y_score = stage2_model.predict_proba(x_test_stage2)[:, 1]
        y_linear = stage2_model.decision_function(x_test_stage2)

        pred_df = test_df.copy()
        pred_df["validation_mode"] = cv_mode
        pred_df["embedding_mode"] = mode
        pred_df["mode"] = mode
        pred_df["cv_mode"] = cv_mode
        pred_df["fold_index"] = int(fold.fold_index)
        pred_df["fold_id"] = str(fold.fold_id)
        pred_df["heldout_locus_id"] = str(fold.fold_id)
        pred_df["baseline_predicted_linear_score"] = baseline_linear_test
        pred_df["baseline_predicted_score"] = baseline_score_test
        pred_df["predicted_linear_score"] = y_linear
        pred_df["predicted_score"] = y_score
        pred_df = assign_rank_within_locus(pred_df, ranking_col="predicted_linear_score")

        stage2_scaler = stage2_model.named_steps["scaler"]
        stage2_logreg = stage2_model.named_steps["logreg"]
        stage2_coef = stage2_logreg.coef_.ravel()
        z_test_stage2 = stage2_scaler.transform(x_test_stage2)
        contrib = z_test_stage2 * stage2_coef.reshape(1, -1)
        pred_df["top_feature_contributions"] = [
            format_top_contributions(contrib[i], stage2_coef, stage2_feature_names, top_k=int(args.top_k_contrib))
            for i in range(contrib.shape[0])
        ]
        prediction_rows.append(pred_df)

        roc = safe_roc_auc(y_test, y_score)
        pr = safe_pr_auc(y_test, y_score)
        positive_df = pred_df.loc[pred_df[LABEL_COL].astype(int) == 1].copy()
        if positive_df.empty:
            recall_at_1 = float("nan")
            recall_at_3 = float("nan")
            mrr = float("nan")
        else:
            pos_ranks = positive_df["rank_within_locus"].astype(int).to_numpy()
            recall_at_1 = float(np.mean(pos_ranks <= 1))
            recall_at_3 = float(np.mean(pos_ranks <= 3))
            mrr = float(1.0 / pos_ranks.min())
            positive_rank_rows.append(
                positive_df[
                    [
                        "validation_mode",
                        "embedding_mode",
                        "mode",
                        "cv_mode",
                        "fold_index",
                        "fold_id",
                        "heldout_locus_id",
                        LOCUS_COL,
                        GENE_ID_COL,
                        GENE_SYMBOL_COL,
                        LABEL_COL,
                        "predicted_score",
                        "rank_within_locus",
                    ]
                ].copy()
            )

        fold_rows.append(
            {
                **base_fold_info,
                "roc_auc": float_or_nan(roc),
                "pr_auc": float_or_nan(pr),
                "recall_at_1": float_or_nan(recall_at_1),
                "recall_at_3": float_or_nan(recall_at_3),
                "mrr": float_or_nan(mrr),
                "status": "ok",
            }
        )

    if prediction_rows:
        all_predictions = pd.concat(prediction_rows, axis=0, ignore_index=True)
    else:
        all_predictions = pd.DataFrame(columns=df.columns.tolist() + ["predicted_score", "rank_within_locus"])
    all_predictions = all_predictions.sort_values(
        by=["fold_index", LOCUS_COL, "predicted_linear_score"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)

    fold_metrics_df = pd.DataFrame(fold_rows)
    positive_ranks_df = (
        pd.concat(positive_rank_rows, axis=0, ignore_index=True)
        if positive_rank_rows
        else pd.DataFrame(
            columns=[
                "validation_mode",
                "embedding_mode",
                "mode",
                "cv_mode",
                "fold_index",
                "fold_id",
                "heldout_locus_id",
                LOCUS_COL,
                GENE_ID_COL,
                GENE_SYMBOL_COL,
                LABEL_COL,
                "predicted_score",
                "rank_within_locus",
            ]
        )
    )

    top3_df = all_predictions.loc[all_predictions["rank_within_locus"] <= 3].copy()
    false_positive_df = all_predictions.loc[
        (all_predictions[LABEL_COL].astype(int) == 0)
        & (all_predictions["rank_within_locus"] <= int(args.false_positive_rank_threshold))
    ].copy()

    pooled_roc = safe_roc_auc(
        all_predictions[LABEL_COL].astype(int).to_numpy() if not all_predictions.empty else np.array([], dtype=int),
        all_predictions["predicted_score"].to_numpy() if not all_predictions.empty else np.array([], dtype=float),
    )
    pooled_pr = safe_pr_auc(
        all_predictions[LABEL_COL].astype(int).to_numpy() if not all_predictions.empty else np.array([], dtype=int),
        all_predictions["predicted_score"].to_numpy() if not all_predictions.empty else np.array([], dtype=float),
    )
    linear_all = (
        pd.to_numeric(all_predictions["predicted_linear_score"], errors="coerce").to_numpy(dtype=float)
        if not all_predictions.empty and "predicted_linear_score" in all_predictions.columns
        else np.array([], dtype=float)
    )
    score_all = (
        pd.to_numeric(all_predictions["predicted_score"], errors="coerce").to_numpy(dtype=float)
        if not all_predictions.empty and "predicted_score" in all_predictions.columns
        else np.array([], dtype=float)
    )
    valid_linear_all = np.isfinite(linear_all)
    valid_score_all = np.isfinite(score_all)
    n_validation_rows = int(len(all_predictions))
    n_validation_linear_lt_minus50 = int(np.sum(valid_linear_all & (linear_all < -50.0)))
    frac_validation_linear_lt_minus50 = (
        float(n_validation_linear_lt_minus50 / n_validation_rows) if n_validation_rows > 0 else float("nan")
    )
    n_validation_score_eq_zero = int(np.sum(valid_score_all & np.isclose(score_all, 0.0, rtol=0.0, atol=0.0)))
    frac_validation_score_eq_zero = (
        float(n_validation_score_eq_zero / n_validation_rows) if n_validation_rows > 0 else float("nan")
    )
    n_unique_validation_linear_scores = (
        int(pd.Series(linear_all[valid_linear_all]).nunique(dropna=True)) if np.any(valid_linear_all) else 0
    )
    n_unique_validation_scores = (
        int(pd.Series(score_all[valid_score_all]).nunique(dropna=True)) if np.any(valid_score_all) else 0
    )

    used_embedding_feature_count = int(min(int(args.pca_dim), len(embedding_cols), len(df)))
    summary = {
        "validation_mode": cv_mode,
        "embedding_mode": mode,
        "mode": mode,
        "cv_mode": cv_mode,
        "baseline_profile": baseline_profile,
        "penalty": str(args.penalty),
        "regularization_strength_C": float(args.regularization_strength),
        "l1_ratio": float(args.l1_ratio),
        "n_rows": int(len(df)),
        "n_loci": int(df[LOCUS_COL].nunique()),
        "n_positive_rows": int(df[LABEL_COL].astype(int).sum()),
        "n_positive_genes": int(df.loc[df[LABEL_COL].astype(int) == 1, GENE_SYMBOL_COL].nunique()),
        "available_abundance_feature_count": int(len(abundance_cols)),
        "used_abundance_feature_count": int(len(abundance_cols)),
        "available_network_feature_count": int(len(network_cols)),
        "used_network_feature_count": int(len(network_cols)),
        "available_baseline_genetic_feature_count": int(len(baseline_genetic_cols)),
        "used_baseline_genetic_feature_count": int(len(baseline_genetic_cols)),
        "available_embedding_feature_count": int(len(embedding_cols)),
        "used_embedding_feature_count": used_embedding_feature_count,
        "stage2_uses_score_baseline": 1,
        "folds_total": int(len(fold_metrics_df)),
        "folds_ok": int((fold_metrics_df["status"] == "ok").sum()) if not fold_metrics_df.empty else 0,
        "mean_n_train_rows": mean_ignore_nan(fold_metrics_df["n_train_rows"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_n_test_rows": mean_ignore_nan(fold_metrics_df["n_test_rows"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_n_unique_genes_train": mean_ignore_nan(fold_metrics_df["n_unique_genes_train"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_n_unique_genes_test": mean_ignore_nan(fold_metrics_df["n_unique_genes_test"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_n_gene_overlap_train_test": mean_ignore_nan(fold_metrics_df["n_gene_overlap_train_test"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_gene_overlap_fraction": mean_ignore_nan(fold_metrics_df["gene_overlap_fraction"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "total_rows_removed_due_to_gene_exclusion": int(fold_metrics_df["n_rows_removed_due_to_gene_exclusion"].sum()) if not fold_metrics_df.empty else 0,
        "total_unique_genes_removed_due_to_gene_exclusion": int(fold_metrics_df["n_unique_genes_removed_due_to_gene_exclusion"].sum()) if not fold_metrics_df.empty else 0,
        "mean_fold_roc_auc": mean_ignore_nan(fold_metrics_df["roc_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "std_fold_roc_auc": std_ignore_nan(fold_metrics_df["roc_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_fold_pr_auc": mean_ignore_nan(fold_metrics_df["pr_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "std_fold_pr_auc": std_ignore_nan(fold_metrics_df["pr_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_recall_at_1": mean_ignore_nan(fold_metrics_df["recall_at_1"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_recall_at_3": mean_ignore_nan(fold_metrics_df["recall_at_3"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_mrr": mean_ignore_nan(fold_metrics_df["mrr"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "pooled_roc_auc": float_or_nan(pooled_roc),
        "pooled_pr_auc": float_or_nan(pooled_pr),
        "n_validation_rows": n_validation_rows,
        "n_validation_linear_lt_minus50": n_validation_linear_lt_minus50,
        "frac_validation_linear_lt_minus50": frac_validation_linear_lt_minus50,
        "n_validation_score_eq_zero": n_validation_score_eq_zero,
        "frac_validation_score_eq_zero": frac_validation_score_eq_zero,
        "n_unique_validation_linear_scores": n_unique_validation_linear_scores,
        "n_unique_validation_scores": n_unique_validation_scores,
    }

    x_full_base = as_numeric_matrix(df, baseline_cols, fill_value=0.0)
    y_full = df[LABEL_COL].astype(int).to_numpy()
    baseline_model_full = build_model(
        penalty=args.penalty,
        c_value=float(args.regularization_strength),
        l1_ratio=float(args.l1_ratio),
        max_iter=int(args.max_iter),
        random_state=int(args.random_state),
    )
    baseline_model_full.fit(x_full_base, y_full)
    baseline_linear_full = baseline_model_full.decision_function(x_full_base)
    baseline_score_full = baseline_model_full.predict_proba(x_full_base)[:, 1]
    baseline_scaler_full = baseline_model_full.named_steps["scaler"]
    baseline_logreg_full = baseline_model_full.named_steps["logreg"]
    baseline_coef_full = baseline_logreg_full.coef_.ravel()
    baseline_intercept_full = float(baseline_logreg_full.intercept_.ravel()[0])

    emb_builder_full = ModeFeatureBuilder(
        baseline_cols=[],
        embedding_cols=embedding_cols,
        mode="pca",
        pca_dim=int(args.pca_dim),
        random_state=int(args.random_state),
    )
    x_full_pca = emb_builder_full.fit_transform(df)
    pca_feature_names_full = emb_builder_full.feature_names_ or []
    stage2_feature_names_full = ["score_baseline"] + list(pca_feature_names_full)
    x_full_stage2 = np.column_stack([baseline_score_full.reshape(-1, 1), x_full_pca])

    stage2_model_full = build_model(
        penalty=args.penalty,
        c_value=float(args.regularization_strength),
        l1_ratio=float(args.l1_ratio),
        max_iter=int(args.max_iter),
        random_state=int(args.random_state),
    )
    stage2_model_full.fit(x_full_stage2, y_full)
    stage2_scaler_full = stage2_model_full.named_steps["scaler"]
    stage2_logreg_full = stage2_model_full.named_steps["logreg"]
    stage2_coef_full = stage2_logreg_full.coef_.ravel()
    stage2_intercept_full = float(stage2_logreg_full.intercept_.ravel()[0])
    final_linear_full = stage2_model_full.decision_function(x_full_stage2)
    final_score_full = stage2_model_full.predict_proba(x_full_stage2)[:, 1]

    coef_df = pd.DataFrame(
        {
            "validation_mode": cv_mode,
            "embedding_mode": mode,
            "mode": mode,
            "cv_mode": cv_mode,
            "feature": stage2_feature_names_full,
            "coefficient": stage2_coef_full,
            "abs_coefficient": np.abs(stage2_coef_full),
            "non_zero": (np.abs(stage2_coef_full) > 1e-12).astype(int),
        }
    )
    feature_group: List[str] = []
    for name in coef_df["feature"].astype(str).tolist():
        if name == "score_baseline":
            feature_group.append("baseline_score")
        elif name.startswith("emb_pca_"):
            feature_group.append("embedding_pca")
        else:
            feature_group.append("other")
    coef_df["feature_group"] = feature_group
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False, kind="stable").reset_index(drop=True)

    scaler_df = pd.DataFrame(
        {
            "validation_mode": cv_mode,
            "embedding_mode": mode,
            "mode": mode,
            "cv_mode": cv_mode,
            "feature": stage2_feature_names_full,
            "scaler_mean": np.asarray(getattr(stage2_scaler_full, "mean_", np.zeros(len(stage2_feature_names_full))), dtype=float),
            "scaler_scale": np.asarray(getattr(stage2_scaler_full, "scale_", np.ones(len(stage2_feature_names_full))), dtype=float),
            "scaler_var": np.asarray(getattr(stage2_scaler_full, "var_", np.zeros(len(stage2_feature_names_full))), dtype=float),
        }
    )

    pca_df = pd.DataFrame(
        columns=[
            "validation_mode",
            "embedding_mode",
            "mode",
            "cv_mode",
            "component",
            "explained_variance_ratio",
            "cumulative_explained_variance_ratio",
        ]
    )
    pca_transformer_artifacts: Optional[Dict[str, np.ndarray]] = None
    if (
        emb_builder_full.pca_explained_variance_ratio_ is not None
        and emb_builder_full.emb_scaler is not None
        and emb_builder_full.pca is not None
    ):
        evr = emb_builder_full.pca_explained_variance_ratio_
        pca_df = pd.DataFrame(
            {
                "validation_mode": cv_mode,
                "embedding_mode": mode,
                "mode": mode,
                "cv_mode": cv_mode,
                "component": [f"emb_pca_{i:03d}" for i in range(len(evr))],
                "explained_variance_ratio": evr,
                "cumulative_explained_variance_ratio": np.cumsum(evr),
            }
        )
        pca_transformer_artifacts = {
            "embedding_feature_names": np.asarray(list(embedding_cols), dtype=object),
            "pca_feature_names": np.asarray(list(pca_feature_names_full), dtype=object),
            "emb_scaler_mean": np.asarray(emb_builder_full.emb_scaler.mean_, dtype=np.float64),
            "emb_scaler_scale": np.asarray(emb_builder_full.emb_scaler.scale_, dtype=np.float64),
            "pca_components": np.asarray(emb_builder_full.pca.components_, dtype=np.float64),
            "pca_mean": np.asarray(emb_builder_full.pca.mean_, dtype=np.float64),
        }

    model_parameters = {
        "validation_mode": cv_mode,
        "embedding_mode": mode,
        "mode": mode,
        "cv_mode": cv_mode,
        "baseline_profile": baseline_profile,
        "penalty": str(args.penalty),
        "regularization_strength_C": float(args.regularization_strength),
        "l1_ratio": float(args.l1_ratio),
        "stage1_solver": str(getattr(baseline_logreg_full, "solver", "")),
        "stage2_solver": str(getattr(stage2_logreg_full, "solver", "")),
        "stage1_class_weight": str(getattr(baseline_logreg_full, "class_weight", "")),
        "stage2_class_weight": str(getattr(stage2_logreg_full, "class_weight", "")),
        "stage1_intercept": float(baseline_intercept_full),
        "stage2_intercept": float(stage2_intercept_full),
        "final_score_formula": "stage2_logit(score_baseline, emb_pca_*) -> sigmoid",
        "stage2_uses_score_baseline": 1,
    }

    two_stage_contrib_df = pd.DataFrame(
        {
            "baseline_predicted_linear_score": baseline_linear_full,
            "baseline_predicted_score": baseline_score_full,
            "final_predicted_linear_score": final_linear_full,
            "final_predicted_score": final_score_full,
        }
    )

    return {
        "summary": summary,
        "fold_metrics": fold_metrics_df,
        "positive_ranks": positive_ranks_df,
        "all_predictions": all_predictions,
        "top3_predictions": top3_df,
        "false_positives": false_positive_df,
        "coefficients": coef_df,
        "scaler_feature_stats": scaler_df,
        "model_parameters": model_parameters,
        "feature_lists": {
            "baseline_profile": baseline_profile,
            "baseline_genetic_features": list(baseline_genetic_cols),
            "abundance_features": list(abundance_cols),
            "network_features": list(network_cols),
            "baseline_features_used_in_stage1": list(baseline_cols),
            "stage2_features_used": list(stage2_feature_names_full),
            "embedding_features_available": list(embedding_cols),
            "embedding_mode": mode,
            "embedding_features_used_count": int(used_embedding_feature_count),
        },
        "pca_transformer_artifacts": pca_transformer_artifacts,
        "pca_explained_variance": pca_df,
        "two_stage_contribution_table": two_stage_contrib_df,
    }


def write_outputs_for_mode(mode_dir: Path, mode_result: Dict[str, object]) -> None:
    mode_dir.mkdir(parents=True, exist_ok=True)

    summary_path = mode_dir / "summary_metrics.json"
    fold_path = mode_dir / "fold_metrics.csv"
    pos_rank_path = mode_dir / "positive_gene_ranks.csv"
    all_pred_path = mode_dir / "all_ranked_predictions.csv"
    top3_path = mode_dir / "top3_per_locus.csv"
    fp_path = mode_dir / "high_rank_false_positives.csv"
    coef_path = mode_dir / "coefficient_table.csv"
    scaler_path = mode_dir / "scaler_feature_stats.csv"
    model_params_path = mode_dir / "model_parameters.json"
    feature_lists_path = mode_dir / "feature_lists.json"
    feature_table_path = mode_dir / "feature_lists.csv"
    pca_artifacts_path = mode_dir / "pca_transformer_artifacts.npz"
    residual_artifacts_path = mode_dir / "residual_pca_artifacts.npz"
    residual_fit_path = mode_dir / "residual_fit_diagnostics.csv"
    residual_contrib_path = mode_dir / "residual_contribution_table.csv"
    two_stage_contrib_path = mode_dir / "two_stage_contribution_table.csv"
    pca_path = mode_dir / "pca_explained_variance.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(mode_result["summary"], f, indent=2, ensure_ascii=False)

    mode_result["fold_metrics"].to_csv(fold_path, index=False)
    mode_result["positive_ranks"].to_csv(pos_rank_path, index=False)
    mode_result["all_predictions"].to_csv(all_pred_path, index=False)
    mode_result["top3_predictions"].to_csv(top3_path, index=False)
    mode_result["false_positives"].to_csv(fp_path, index=False)
    mode_result["coefficients"].to_csv(coef_path, index=False)
    mode_result["scaler_feature_stats"].to_csv(scaler_path, index=False)
    with open(model_params_path, "w", encoding="utf-8") as f:
        json.dump(mode_result["model_parameters"], f, indent=2, ensure_ascii=False)
    with open(feature_lists_path, "w", encoding="utf-8") as f:
        json.dump(mode_result.get("feature_lists", {}), f, indent=2, ensure_ascii=False)
    feature_rows: List[Dict[str, object]] = []
    feature_lists = mode_result.get("feature_lists", {})
    if isinstance(feature_lists, dict):
        for family_key, family_name in [
            ("baseline_genetic_features", "baseline_genetic"),
            ("abundance_features", "abundance"),
            ("network_features", "network"),
            ("baseline_features_used_in_mode", "baseline_features_used_in_mode"),
        ]:
            vals = feature_lists.get(family_key, [])
            if isinstance(vals, list):
                for idx, feat in enumerate(vals):
                    feature_rows.append(
                        {
                            "family": family_name,
                            "order": int(idx),
                            "feature": str(feat),
                        }
                    )
    pd.DataFrame(feature_rows).to_csv(feature_table_path, index=False)
    pca_artifacts = mode_result.get("pca_transformer_artifacts")
    if isinstance(pca_artifacts, dict) and pca_artifacts:
        np.savez(pca_artifacts_path, **pca_artifacts)
    residual_artifacts = mode_result.get("residual_pca_artifacts")
    if isinstance(residual_artifacts, dict) and residual_artifacts:
        np.savez(residual_artifacts_path, **residual_artifacts)
    residual_fit_df = mode_result.get("residual_fit_diagnostics")
    if isinstance(residual_fit_df, pd.DataFrame):
        residual_fit_df.to_csv(residual_fit_path, index=False)
    residual_contrib_df = mode_result.get("residual_contribution_table")
    if isinstance(residual_contrib_df, pd.DataFrame):
        residual_contrib_df.to_csv(residual_contrib_path, index=False)
    two_stage_contrib_df = mode_result.get("two_stage_contribution_table")
    if isinstance(two_stage_contrib_df, pd.DataFrame):
        two_stage_contrib_df.to_csv(two_stage_contrib_path, index=False)
    mode_result["pca_explained_variance"].to_csv(pca_path, index=False)


def _fmt_metric(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def write_gene_leakage_validation_report(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Gene Leakage Validation Report")
    lines.append("")
    lines.append("This report summarizes embedding impact under leakage-aware validation modes.")
    lines.append("")

    for cv_mode in ["lolo_gene_exclusion"]:
        sub = summary_df[summary_df["validation_mode"] == cv_mode].copy()
        if sub.empty:
            continue
        order = {"none": 0, "full": 1, "pca": 2, BASELINE_THEN_PCA_MODE: 3, RESIDUAL_PCA_MODE: 4}
        sub = sub.sort_values("embedding_mode", key=lambda s: s.map(order).fillna(999))

        lines.append(f"## Validation Mode: `{cv_mode}`")
        lines.append("")
        lines.append("| embedding_mode | PR-AUC | ROC-AUC | Recall@1 | Recall@3 | MRR | overlap_fraction |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in sub.itertuples(index=False):
            lines.append(
                "| "
                f"{row.embedding_mode} | {_fmt_metric(row.mean_fold_pr_auc)} | {_fmt_metric(row.mean_fold_roc_auc)} | "
                f"{_fmt_metric(row.mean_recall_at_1)} | {_fmt_metric(row.mean_recall_at_3)} | {_fmt_metric(row.mean_mrr)} | "
                f"{_fmt_metric(row.mean_gene_overlap_fraction)} |"
            )
        lines.append("")

        m = sub.set_index("embedding_mode")
        if "none" in m.index and "full" in m.index:
            d_pr = float(m.loc["full", "mean_fold_pr_auc"] - m.loc["none", "mean_fold_pr_auc"])
            d_r3 = float(m.loc["full", "mean_recall_at_3"] - m.loc["none", "mean_recall_at_3"])
            d_mrr = float(m.loc["full", "mean_mrr"] - m.loc["none", "mean_mrr"])
            lines.append(
                f"- Embedding effect (`full - none`): PR-AUC={_fmt_metric(d_pr)}, "
                f"Recall@3={_fmt_metric(d_r3)}, MRR={_fmt_metric(d_mrr)}."
            )
        if "none" in m.index and "pca" in m.index:
            d_pr = float(m.loc["pca", "mean_fold_pr_auc"] - m.loc["none", "mean_fold_pr_auc"])
            d_r3 = float(m.loc["pca", "mean_recall_at_3"] - m.loc["none", "mean_recall_at_3"])
            d_mrr = float(m.loc["pca", "mean_mrr"] - m.loc["none", "mean_mrr"])
            lines.append(
                f"- PCA effect (`pca - none`): PR-AUC={_fmt_metric(d_pr)}, "
                f"Recall@3={_fmt_metric(d_r3)}, MRR={_fmt_metric(d_mrr)}."
            )
        if "none" in m.index and RESIDUAL_PCA_MODE in m.index:
            d_pr = float(m.loc[RESIDUAL_PCA_MODE, "mean_fold_pr_auc"] - m.loc["none", "mean_fold_pr_auc"])
            d_r3 = float(m.loc[RESIDUAL_PCA_MODE, "mean_recall_at_3"] - m.loc["none", "mean_recall_at_3"])
            d_mrr = float(m.loc[RESIDUAL_PCA_MODE, "mean_mrr"] - m.loc["none", "mean_mrr"])
            lines.append(
                f"- Residual PCA effect (`{RESIDUAL_PCA_MODE} - none`): PR-AUC={_fmt_metric(d_pr)}, "
                f"Recall@3={_fmt_metric(d_r3)}, MRR={_fmt_metric(d_mrr)}."
            )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Validation mode is fixed to `lolo_gene_exclusion` (zero train/test gene overlap by construction).")
    lines.append(
        "- If embedding gains disappear under zero-overlap modes, previous gains likely relied on gene identity leakage."
    )
    lines.append(
        "- If gains persist under zero-overlap modes, embeddings likely capture transferable biological signal."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_l1_benchmark_for_pca(
    *,
    df: pd.DataFrame,
    gene_series: pd.Series,
    cv_mode: str,
    baseline_profile: str,
    baseline_genetic_cols: List[str],
    abundance_cols: List[str],
    network_cols: List[str],
    embedding_cols: List[str],
    args: argparse.Namespace,
    network_model: Optional[GeneNetworkModel] = None,
    network_score_col: str = "network_score",
) -> Tuple[pd.DataFrame, float, Dict[str, object]]:
    c_values = parse_l1_c_grid(str(args.l1_benchmark_c_grid))
    rows: List[Dict[str, object]] = []
    mode_result_by_c: Dict[float, Dict[str, object]] = {}

    for c in c_values:
        bench_args = argparse.Namespace(**vars(args))
        bench_args.penalty = "l1"
        bench_args.regularization_strength = float(c)
        mode_result = run_cv_for_mode(
            df=df,
            gene_series=gene_series,
            baseline_profile=baseline_profile,
            mode="pca",
            cv_mode=cv_mode,
            baseline_genetic_cols=baseline_genetic_cols,
            baseline_cols=baseline_columns_for_mode(
                "pca",
                baseline_genetic_cols=baseline_genetic_cols,
                abundance_cols=abundance_cols,
                network_cols=network_cols,
            ),
            abundance_cols=abundance_cols,
            network_cols=network_cols,
            embedding_cols=embedding_cols,
            args=bench_args,
            network_model=network_model,
            network_score_col=network_score_col,
        )
        mode_result_by_c[float(c)] = mode_result
        summary = mode_result["summary"]
        coef_df = mode_result["coefficients"].copy()
        coef_df["non_zero"] = pd.to_numeric(coef_df["non_zero"], errors="coerce").fillna(0).astype(int)
        pca_mask = coef_df["feature"].astype(str).str.startswith("emb_pca_")
        non_zero_total = int(coef_df["non_zero"].sum())
        non_zero_pca = int(coef_df.loc[pca_mask, "non_zero"].sum())
        non_zero_non_emb = int(coef_df.loc[~pca_mask, "non_zero"].sum())
        total_pca = int(pca_mask.sum())
        frac_pca_non_zero = float(non_zero_pca / total_pca) if total_pca > 0 else float("nan")
        pos_counts = positive_topk_counts(mode_result["positive_ranks"])
        fold_df = mode_result["fold_metrics"]
        std_recall_at_1 = std_ignore_nan(fold_df["recall_at_1"].tolist()) if not fold_df.empty else float("nan")
        std_mrr = std_ignore_nan(fold_df["mrr"].tolist()) if not fold_df.empty else float("nan")

        rows.append(
            {
                "validation_mode": cv_mode,
                "baseline_profile": baseline_profile,
                "embedding_mode": "pca",
                "penalty": "l1",
                "C": float(c),
                "mean_fold_roc_auc": float(summary.get("mean_fold_roc_auc", float("nan"))),
                "mean_fold_pr_auc": float(summary.get("mean_fold_pr_auc", float("nan"))),
                "mean_recall_at_1": float(summary.get("mean_recall_at_1", float("nan"))),
                "mean_recall_at_3": float(summary.get("mean_recall_at_3", float("nan"))),
                "mean_mrr": float(summary.get("mean_mrr", float("nan"))),
                "std_fold_roc_auc": float(summary.get("std_fold_roc_auc", float("nan"))),
                "std_fold_pr_auc": float(summary.get("std_fold_pr_auc", float("nan"))),
                "std_recall_at_1": float(std_recall_at_1),
                "std_mrr": float(std_mrr),
                "pooled_roc_auc": float(summary.get("pooled_roc_auc", float("nan"))),
                "pooled_pr_auc": float(summary.get("pooled_pr_auc", float("nan"))),
                "model_intercept": float(mode_result["model_parameters"].get("model_intercept", float("nan"))),
                "n_validation_rows": int(summary.get("n_validation_rows", 0)),
                "n_validation_linear_lt_minus50": int(summary.get("n_validation_linear_lt_minus50", 0)),
                "frac_validation_linear_lt_minus50": float(
                    summary.get("frac_validation_linear_lt_minus50", float("nan"))
                ),
                "n_validation_score_eq_zero": int(summary.get("n_validation_score_eq_zero", 0)),
                "frac_validation_score_eq_zero": float(summary.get("frac_validation_score_eq_zero", float("nan"))),
                "n_unique_validation_linear_scores": int(summary.get("n_unique_validation_linear_scores", 0)),
                "n_unique_validation_scores": int(summary.get("n_unique_validation_scores", 0)),
                "non_zero_total_coefficients": int(non_zero_total),
                "non_zero_pca_coefficients": int(non_zero_pca),
                "non_zero_non_embedding_coefficients": int(non_zero_non_emb),
                "fraction_non_zero_pca": float(frac_pca_non_zero),
                "selected": 0,
                **pos_counts,
            }
        )

    bench_df = pd.DataFrame(rows).sort_values("C", ascending=True, kind="stable").reset_index(drop=True)
    selected_c = select_l1_penalty_from_benchmark(
        bench_df=bench_df,
        pr_auc_tol=float(args.l1_selection_pr_auc_tol),
    )
    bench_df["selected"] = (np.isclose(bench_df["C"].astype(float), float(selected_c), rtol=0.0, atol=1e-12)).astype(int)
    selected_mode_result = mode_result_by_c[float(selected_c)]
    selected_mode_result["summary"]["selected_from_l1_benchmark"] = 1
    selected_mode_result["summary"]["selected_l1_C"] = float(selected_c)
    selected_mode_result["summary"]["l1_selection_pr_auc_tol"] = float(args.l1_selection_pr_auc_tol)
    selected_mode_result["summary"]["l1_benchmark_grid_size"] = int(len(bench_df))
    selected_mode_result["summary"]["l1_benchmark_c_values"] = [float(v) for v in bench_df["C"].astype(float).tolist()]
    selected_mode_result["summary"]["selection_rule"] = (
        "Choose sparsest model (fewest non-zero coefficients) among runs with "
        "mean_fold_pr_auc >= best_pr_auc - tol; tie-break by higher Recall@1, "
        "then higher MRR, then smaller C."
    )
    return bench_df, float(selected_c), selected_mode_result


def write_l1_benchmark_outputs(
    *,
    cv_dir: Path,
    bench_df: pd.DataFrame,
    selected_c: float,
    selected_summary: Dict[str, object],
    reference_summary: Optional[Dict[str, object]],
    pr_auc_tol: float,
) -> None:
    csv_path = cv_dir / "l1_benchmark_summary.csv"
    json_path = cv_dir / "l1_benchmark_summary.json"
    report_path = cv_dir / "l1_benchmark_report.md"

    bench_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bench_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    best_pr = float(pd.to_numeric(bench_df["mean_fold_pr_auc"], errors="coerce").max()) if not bench_df.empty else float("nan")
    selected_row = bench_df.loc[np.isclose(bench_df["C"].astype(float), float(selected_c), rtol=0.0, atol=1e-12)]
    selected_row_dict = selected_row.iloc[0].to_dict() if not selected_row.empty else {}

    lines: List[str] = []
    lines.append("# L1 Benchmark Report")
    lines.append("")
    lines.append("## Why `has_gene_embedding` Was Removed")
    lines.append("")
    lines.append("- `has_gene_embedding` is a metadata/coverage indicator, not a biological effect-size signal.")
    lines.append("- Keeping it as a model input can leak representational availability rather than biology.")
    lines.append("- The column is kept in tables for diagnostics, but excluded from all training modes.")
    lines.append("")
    lines.append("## Benchmark Setup")
    lines.append("")
    lines.append("- Validation protocol: `lolo_gene_exclusion`.")
    lines.append("- Model family benchmarked: PCA embedding model (`mode_pca`) with `penalty=l1`.")
    lines.append(f"- Grid over `C`: {', '.join([f'{float(v):g}' for v in bench_df['C'].astype(float).tolist()])}")
    lines.append(f"- PR-AUC tolerance for selection: {float(pr_auc_tol):.4f}")
    lines.append("")
    lines.append("## Selection Rule")
    lines.append("")
    lines.append(
        "- Select the sparsest model (minimum total non-zero coefficients) among runs with "
        "`mean_fold_pr_auc >= best_pr_auc - tol`."
    )
    lines.append("- Tie-break: higher Recall@1, then higher MRR, then smaller C.")
    lines.append("")
    lines.append("## Selected Penalty")
    lines.append("")
    lines.append(f"- Selected `C`: {float(selected_c):g}")
    if selected_row_dict:
        lines.append(f"- Selected mean PR-AUC: {float(selected_row_dict.get('mean_fold_pr_auc', float('nan'))):.4f}")
        lines.append(
            f"- Selected non-zero coefficients: total={int(selected_row_dict.get('non_zero_total_coefficients', 0))}, "
            f"pca={int(selected_row_dict.get('non_zero_pca_coefficients', 0))}, "
            f"non-embedding={int(selected_row_dict.get('non_zero_non_embedding_coefficients', 0))}"
        )
        lines.append(f"- Selected model intercept: {float(selected_row_dict.get('model_intercept', float('nan'))):.4f}")
        lines.append(
            f"- Selected saturation proxy (`validation logits < -50`): "
            f"{int(selected_row_dict.get('n_validation_linear_lt_minus50', 0))}/"
            f"{int(selected_row_dict.get('n_validation_rows', 0))} "
            f"({float(selected_row_dict.get('frac_validation_linear_lt_minus50', float('nan'))):.3f})"
        )
        lines.append(
            f"- Selected hard-zero predicted probabilities on validation: "
            f"{int(selected_row_dict.get('n_validation_score_eq_zero', 0))}/"
            f"{int(selected_row_dict.get('n_validation_rows', 0))} "
            f"({float(selected_row_dict.get('frac_validation_score_eq_zero', float('nan'))):.3f})"
        )
    lines.append(f"- Best mean PR-AUC in grid: {best_pr:.4f}")
    lines.append("")
    lines.append("## Comparison vs Previous PCA Reference")
    lines.append("")
    if reference_summary is None:
        lines.append("- No separate no-penalty PCA reference was available in this run.")
    else:
        lines.append(
            f"- Reference (no-penalty) mean PR-AUC: {float(reference_summary.get('mean_fold_pr_auc', float('nan'))):.4f}"
        )
        lines.append(
            f"- Selected L1 mean PR-AUC: {float(selected_summary.get('mean_fold_pr_auc', float('nan'))):.4f}"
        )
        lines.append(
            f"- Reference Recall@1: {float(reference_summary.get('mean_recall_at_1', float('nan'))):.4f}; "
            f"Selected Recall@1: {float(selected_summary.get('mean_recall_at_1', float('nan'))):.4f}"
        )
        lines.append(
            f"- Reference MRR: {float(reference_summary.get('mean_mrr', float('nan'))):.4f}; "
            f"Selected MRR: {float(selected_summary.get('mean_mrr', float('nan'))):.4f}"
        )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.input_table).copy()
    df = impute_distance_features_worst_case(df)
    modes = resolve_modes(args.embedding_mode)
    if bool(args.run_l1_benchmark) and "pca" not in modes:
        raise ValueError("--run-l1-benchmark requires embedding mode to include 'pca' (use --embedding-mode pca/all).")
    cv_modes = resolve_cv_modes(args.cv_mode)
    embedding_cols = sorted([c for c in df.columns if c.startswith("gene_emb_")])
    profile_cols = resolve_baseline_profile_columns(
        df,
        baseline_profile=str(args.baseline_profile),
        include_network_score=bool(args.include_network_score),
        network_score_column=str(args.network_score_column),
    )
    baseline_genetic_cols = profile_cols["baseline_genetic_cols"]
    abundance_cols = profile_cols["abundance_cols"]
    network_cols = profile_cols["network_cols"]
    if bool(args.include_network_score) and str(args.network_score_column) not in df.columns:
        raise ValueError(
            f"--include-network-score requested but column '{args.network_score_column}' is missing in input table."
        )
    validate_input(
        df,
        modes=modes,
        embedding_cols=embedding_cols,
        required_baseline_cols=baseline_genetic_cols,
    )

    network_model: Optional[GeneNetworkModel] = None
    network_build_stats: Dict[str, float] = {}
    if bool(args.include_network_score):
        network_model, network_build_stats = load_or_build_string_gene_network_model(
            aliases_path=Path(args.network_aliases_path),
            links_path=Path(args.network_links_path),
            cache_path=(Path(args.network_cache_path) if args.network_cache_path else None),
            min_combined_score=float(args.network_min_combined_score),
        )
        print(
            "[info] Network model loaded: "
            f"nodes={network_model.transition_matrix.shape[0]} "
            f"from_cache={int(network_build_stats.get('from_cache', 0.0))}"
        )
        if network_build_stats:
            print(f"[info] Network build stats: {network_build_stats}")

    for col in [LOCUS_COL, GENE_ID_COL, GENE_SYMBOL_COL, "gwas_study_id"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)
    gene_series = make_gene_group_series(df)

    all_summary_rows: List[Dict[str, object]] = []

    for cv_mode in cv_modes:
        cv_dir = args.out_dir / f"cv_{cv_mode}"
        cv_dir.mkdir(parents=True, exist_ok=True)

        cv_summary_rows: List[Dict[str, object]] = []
        for mode in modes:
            if mode == "pca" and bool(args.run_l1_benchmark):
                reference_summary: Optional[Dict[str, object]] = None
                if str(args.penalty).lower() != "l1":
                    reference_mode_result = run_cv_for_mode(
                        df=df,
                        gene_series=gene_series,
                        baseline_profile=str(args.baseline_profile),
                        mode="pca",
                        cv_mode=cv_mode,
                        baseline_genetic_cols=baseline_genetic_cols,
                        baseline_cols=baseline_columns_for_mode(
                            "pca",
                            baseline_genetic_cols=baseline_genetic_cols,
                            abundance_cols=abundance_cols,
                            network_cols=network_cols,
                        ),
                        abundance_cols=abundance_cols,
                        network_cols=network_cols,
                        embedding_cols=embedding_cols,
                        args=args,
                        network_model=network_model,
                        network_score_col=str(args.network_score_column),
                    )
                    write_outputs_for_mode(cv_dir / "mode_pca_reference_requested_penalty", reference_mode_result)
                    reference_summary = dict(reference_mode_result["summary"])
                    reference_summary["mode"] = "pca_reference_requested_penalty"
                    reference_summary["embedding_mode"] = "pca_reference_requested_penalty"
                    cv_summary_rows.append(reference_summary)
                    all_summary_rows.append(reference_summary)

                bench_df, selected_c, selected_mode_result = run_l1_benchmark_for_pca(
                    df=df,
                    gene_series=gene_series,
                    cv_mode=cv_mode,
                    baseline_profile=str(args.baseline_profile),
                    baseline_genetic_cols=baseline_genetic_cols,
                    abundance_cols=abundance_cols,
                    network_cols=network_cols,
                    embedding_cols=embedding_cols,
                    args=args,
                    network_model=network_model,
                    network_score_col=str(args.network_score_column),
                )
                selected_mode_result["summary"]["mode"] = "pca"
                selected_mode_result["summary"]["embedding_mode"] = "pca"
                selected_mode_result["summary"]["selected_l1_C"] = float(selected_c)
                selected_mode_result["summary"]["selected_from_l1_benchmark"] = 1
                selected_mode_result["summary"]["penalty"] = "l1"
                selected_mode_result["summary"]["regularization_strength_C"] = float(selected_c)
                selected_mode_result["model_parameters"]["penalty"] = "l1"
                selected_mode_result["model_parameters"]["regularization_strength_C"] = float(selected_c)
                write_outputs_for_mode(cv_dir / "mode_pca", selected_mode_result)
                write_l1_benchmark_outputs(
                    cv_dir=cv_dir,
                    bench_df=bench_df,
                    selected_c=float(selected_c),
                    selected_summary=selected_mode_result["summary"],
                    reference_summary=reference_summary,
                    pr_auc_tol=float(args.l1_selection_pr_auc_tol),
                )
                cv_summary_rows.append(selected_mode_result["summary"])
                all_summary_rows.append(selected_mode_result["summary"])
                continue

            if mode == RESIDUAL_PCA_MODE:
                mode_result = run_cv_for_residual_pca(
                    df=df,
                    gene_series=gene_series,
                    baseline_profile=str(args.baseline_profile),
                    mode=mode,
                    cv_mode=cv_mode,
                    baseline_genetic_cols=baseline_genetic_cols,
                    baseline_cols=baseline_columns_for_mode(
                        mode,
                        baseline_genetic_cols=baseline_genetic_cols,
                        abundance_cols=abundance_cols,
                        network_cols=network_cols,
                    ),
                    abundance_cols=abundance_cols,
                    network_cols=network_cols,
                    embedding_cols=embedding_cols,
                    args=args,
                )
            elif mode == BASELINE_THEN_PCA_MODE:
                mode_result = run_cv_for_baseline_then_pca(
                    df=df,
                    gene_series=gene_series,
                    baseline_profile=str(args.baseline_profile),
                    mode=mode,
                    cv_mode=cv_mode,
                    baseline_genetic_cols=baseline_genetic_cols,
                    baseline_cols=baseline_columns_for_mode(
                        mode,
                        baseline_genetic_cols=baseline_genetic_cols,
                        abundance_cols=abundance_cols,
                        network_cols=network_cols,
                    ),
                    abundance_cols=abundance_cols,
                    network_cols=network_cols,
                    embedding_cols=embedding_cols,
                    args=args,
                )
            else:
                mode_result = run_cv_for_mode(
                    df=df,
                    gene_series=gene_series,
                    baseline_profile=str(args.baseline_profile),
                    mode=mode,
                    cv_mode=cv_mode,
                    baseline_genetic_cols=baseline_genetic_cols,
                    baseline_cols=baseline_columns_for_mode(
                        mode,
                        baseline_genetic_cols=baseline_genetic_cols,
                        abundance_cols=abundance_cols,
                        network_cols=network_cols,
                    ),
                    abundance_cols=abundance_cols,
                    network_cols=network_cols,
                    embedding_cols=embedding_cols,
                    args=args,
                    network_model=network_model,
                    network_score_col=str(args.network_score_column),
                )
            write_outputs_for_mode(cv_dir / f"mode_{mode}", mode_result)
            cv_summary_rows.append(mode_result["summary"])
            all_summary_rows.append(mode_result["summary"])

        pd.DataFrame(cv_summary_rows).to_csv(cv_dir / "ablation_summary.csv", index=False)
        with open(cv_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
            json.dump(cv_summary_rows, f, indent=2, ensure_ascii=False)

    comparison_df = pd.DataFrame(all_summary_rows)
    comparison_df.to_csv(args.out_dir / "validation_comparison_summary.csv", index=False)
    with open(args.out_dir / "validation_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary_rows, f, indent=2, ensure_ascii=False)

    write_gene_leakage_validation_report(
        summary_df=comparison_df,
        out_path=args.out_dir / "gene_leakage_validation_report.md",
    )


if __name__ == "__main__":
    main()

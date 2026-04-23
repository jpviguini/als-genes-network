#!/home/viguinijpv/python310/bin/python3.10
"""Train a global functional-prior logistic model using only HPA + PCA text embeddings.

This experiment intentionally removes GWAS-derived features from training.
Feature set:
- hpa_brain_expression_value
- hpa_muscle_expression_value
- PCA(32) on gene text embeddings

Labels are based on fixed reference-paper ALS genes from config.VALIDATION_GENES.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import VALIDATION_GENES
from build_cs_gene_candidate_feature_table import (
    load_gene_embeddings,
    load_hpa_expression_features,
    normalize_gene_id,
    normalize_gene_symbol,
)


DEFAULT_EMBEDDING_PATH = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/features/"
    "featuresUPPER_pubmedbert_neurodegenerative_disease/features_ALS_pubmedbert.pkl"
)
DEFAULT_HPA_PATH = Path("/home/viguinijpv/200.18.99.75:8000/IC/src/data/reference/rna_tissue_consensus.tsv")
DEFAULT_HGNC_PATH = Path("/home/viguinijpv/200.18.99.75:8000/IC/src/data/reference/hgnc_complete_set.txt")
DEFAULT_OUT_DIR = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/"
    f"global_functional_model_hpa_pca_{date.today().strftime('%Y%m%d')}"
)

HPA_FEATURES = ["hpa_brain_expression_value", "hpa_muscle_expression_value"]


@dataclass
class PCAArtifacts:
    emb_scaler: StandardScaler
    pca: PCA
    n_components: int
    explained_variance_ratio: np.ndarray


@dataclass
class CVResult:
    fold_metrics_df: pd.DataFrame
    oof_scores: np.ndarray
    summary: Dict[str, float]
    n_splits: int


@dataclass
class HighConfidenceThreshold:
    threshold: float
    method: str
    gap_index: Optional[int]
    gap_high_score: Optional[float]
    gap_low_score: Optional[float]
    largest_gap: Optional[float]


def _normalize_gene_id_strict(gene_id: object) -> Optional[str]:
    gid = normalize_gene_id(gene_id)
    if gid is None:
        return None
    s = str(gid).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Global functional-prior model using only HPA brain/muscle + PCA text embeddings."
        )
    )
    parser.add_argument("--embedding-path", type=Path, default=DEFAULT_EMBEDDING_PATH)
    parser.add_argument("--hpa-path", type=Path, default=DEFAULT_HPA_PATH)
    parser.add_argument("--hgnc-path", type=Path, default=DEFAULT_HGNC_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument(
        "--penalty",
        choices=["none", "l1", "l2", "elasticnet"],
        default="l2",
        help="Primary model penalty (baseline default is L2).",
    )
    parser.add_argument("--regularization-strength", type=float, default=0.1, help="Inverse regularization C.")
    parser.add_argument("--l1-ratio", type=float, default=0.5, help="Only used when penalty=elasticnet.")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--train-cv", action="store_true", help="Run gene-level stratified CV diagnostics.")
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=None,
        help="Optional manual threshold over predicted_score for high-confidence seeds.",
    )
    parser.add_argument("--high-confidence-top-window", type=int, default=50)
    parser.add_argument("--high-confidence-min-threshold", type=float, default=0.60)
    parser.add_argument("--high-confidence-max-threshold", type=float, default=0.95)
    parser.add_argument(
        "--l1-survival-c",
        type=float,
        default=0.1,
        help="C used for auxiliary L1 model to report surviving features.",
    )
    parser.add_argument(
        "--hpa-expression-threshold",
        type=float,
        default=1.0,
        help="Threshold passed to HPA extractor for binary evidence fields (not used as model features).",
    )
    return parser.parse_args()



def _to_numeric_array(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    mat = np.zeros((len(df), len(cols)), dtype=np.float64)
    for i, col in enumerate(cols):
        if col in df.columns:
            mat[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return mat



def _build_logistic_model(
    *,
    penalty: str,
    c_value: float,
    l1_ratio: float,
    max_iter: int,
    random_state: int,
) -> Pipeline:
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
            solver="saga",
            C=float(c_value),
            l1_ratio=float(l1_ratio),
            class_weight="balanced",
            max_iter=int(max_iter),
            random_state=int(random_state),
        )
    return Pipeline([("scaler", StandardScaler()), ("logreg", clf)])



def _load_hgnc_symbol_to_ensembl(hgnc_path: Path) -> Dict[str, str]:
    if not hgnc_path.exists():
        return {}
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    if "symbol" not in hgnc.columns or "ensembl_gene_id" not in hgnc.columns:
        return {}
    hgnc["symbol_norm"] = hgnc["symbol"].map(normalize_gene_symbol)
    hgnc["gene_id_norm"] = hgnc["ensembl_gene_id"].map(_normalize_gene_id_strict)
    hgnc = hgnc.dropna(subset=["symbol_norm", "gene_id_norm"]).copy()

    out: Dict[str, str] = {}
    for symbol, sub in hgnc.groupby("symbol_norm", sort=False):
        ids = sorted({str(x).strip() for x in sub["gene_id_norm"].tolist() if str(x).strip()})
        if ids:
            out[str(symbol)] = ids[0]
    return out



def _load_embedding_table(embedding_path: Path) -> Tuple[pd.DataFrame, Dict[str, object], List[str]]:
    embeddings, emb_dim, emb_stats = load_gene_embeddings(embedding_path)
    symbols = sorted(embeddings.keys())
    emb_cols = [f"gene_emb_{i:04d}" for i in range(int(emb_dim))]
    matrix = np.vstack([embeddings[g] for g in symbols]).astype(np.float64)
    emb_df = pd.DataFrame(matrix, columns=emb_cols)
    emb_df.insert(0, "gene_symbol", symbols)
    return emb_df, emb_stats, emb_cols



def _load_hpa_gene_table(hpa_path: Path, expression_threshold: float) -> Tuple[pd.DataFrame, Dict[str, object]]:
    hpa_df, hpa_stats = load_hpa_expression_features(
        hpa_path=hpa_path,
        expression_threshold=float(expression_threshold),
    )

    if hpa_df.empty:
        out = pd.DataFrame(columns=["gene_symbol", "gene_id"] + HPA_FEATURES + ["has_hpa_expression_evidence"])
        return out, hpa_stats

    cur = hpa_df.copy()
    cur["gene_symbol"] = cur["gene_symbol"].map(normalize_gene_symbol)
    cur["gene_id"] = cur["gene_id"].map(_normalize_gene_id_strict)
    cur = cur.dropna(subset=["gene_symbol"]).copy()

    for col in HPA_FEATURES + ["has_hpa_expression_evidence"]:
        if col not in cur.columns:
            cur[col] = 0.0
        cur[col] = pd.to_numeric(cur[col], errors="coerce").fillna(0.0)

    rows: List[Dict[str, object]] = []
    for symbol, sub in cur.groupby("gene_symbol", sort=False):
        gene_ids = [g for g in sub["gene_id"].tolist() if isinstance(g, str) and g.strip()]
        gene_id = sorted(set(gene_ids))[0] if gene_ids else None
        rows.append(
            {
                "gene_symbol": str(symbol),
                "gene_id": gene_id,
                "hpa_brain_expression_value": float(sub["hpa_brain_expression_value"].max()),
                "hpa_muscle_expression_value": float(sub["hpa_muscle_expression_value"].max()),
                "has_hpa_expression_evidence": int((sub["has_hpa_expression_evidence"] > 0).any()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["gene_symbol", "gene_id"] + HPA_FEATURES + ["has_hpa_expression_evidence"])
    return out, hpa_stats



def _fit_embedding_pca(
    emb_matrix: np.ndarray,
    pca_dim: int,
    random_state: int,
) -> Tuple[np.ndarray, PCAArtifacts]:
    emb_scaler = StandardScaler()
    emb_scaled = emb_scaler.fit_transform(emb_matrix)
    n_components = min(int(pca_dim), emb_scaled.shape[0], emb_scaled.shape[1])
    if n_components < 1:
        raise ValueError("Cannot fit PCA: not enough rows/features.")

    pca = PCA(n_components=n_components, random_state=int(random_state))
    emb_pca = pca.fit_transform(emb_scaled)
    artifacts = PCAArtifacts(
        emb_scaler=emb_scaler,
        pca=pca,
        n_components=int(n_components),
        explained_variance_ratio=pca.explained_variance_ratio_.copy(),
    )
    return emb_pca, artifacts



def _transform_embedding_pca(emb_matrix: np.ndarray, artifacts: PCAArtifacts) -> np.ndarray:
    emb_scaled = artifacts.emb_scaler.transform(emb_matrix)
    return artifacts.pca.transform(emb_scaled)



def _choose_high_confidence_threshold(
    gene_scores_desc: Sequence[float],
    top_window: int,
    min_threshold: float,
    max_threshold: float,
    override: Optional[float],
) -> HighConfidenceThreshold:
    if override is not None:
        thr = float(override)
        return HighConfidenceThreshold(
            threshold=thr,
            method="manual_override",
            gap_index=None,
            gap_high_score=None,
            gap_low_score=None,
            largest_gap=None,
        )

    scores = [float(s) for s in gene_scores_desc if pd.notna(s)]
    if len(scores) < 2:
        return HighConfidenceThreshold(
            threshold=0.90,
            method="fallback_single_score",
            gap_index=None,
            gap_high_score=None,
            gap_low_score=None,
            largest_gap=None,
        )

    n = min(int(top_window), len(scores) - 1)
    diffs = [scores[i] - scores[i + 1] for i in range(n)]
    gap_idx = int(np.argmax(diffs))
    gap_high = float(scores[gap_idx])
    gap_low = float(scores[gap_idx + 1])
    midpoint = (gap_high + gap_low) / 2.0
    threshold = float(np.clip(midpoint, float(min_threshold), float(max_threshold)))
    threshold = round(threshold, 2)
    return HighConfidenceThreshold(
        threshold=threshold,
        method="largest_gap_midpoint_top_window",
        gap_index=int(gap_idx),
        gap_high_score=gap_high,
        gap_low_score=gap_low,
        largest_gap=float(diffs[gap_idx]),
    )



def _run_cv(
    df: pd.DataFrame,
    *,
    emb_cols: Sequence[str],
    hpa_cols: Sequence[str],
    pca_dim: int,
    penalty: str,
    c_value: float,
    l1_ratio: float,
    max_iter: int,
    random_state: int,
    cv_folds: int,
) -> CVResult:
    y = df["label_positive"].astype(int).to_numpy()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    max_splits = min(int(cv_folds), n_pos, n_neg)
    if max_splits < 2:
        empty = pd.DataFrame(
            [
                {
                    "fold": 1,
                    "n_train": int(len(df)),
                    "n_val": 0,
                    "n_pos_train": int(n_pos),
                    "n_pos_val": 0,
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                }
            ]
        )
        return CVResult(
            fold_metrics_df=empty,
            oof_scores=np.full(len(df), np.nan, dtype=np.float64),
            summary={
                "roc_auc_mean": float("nan"),
                "roc_auc_std": float("nan"),
                "pr_auc_mean": float("nan"),
                "pr_auc_std": float("nan"),
                "oof_roc_auc": float("nan"),
                "oof_pr_auc": float("nan"),
            },
            n_splits=1,
        )

    skf = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=int(random_state))
    oof_scores = np.full(len(df), np.nan, dtype=np.float64)
    fold_rows: List[Dict[str, float]] = []

    emb_all = _to_numeric_array(df, emb_cols)
    hpa_all = _to_numeric_array(df, hpa_cols)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df)), y), start=1):
        y_train = y[train_idx]
        y_val = y[val_idx]

        emb_train = emb_all[train_idx, :]
        emb_val = emb_all[val_idx, :]
        hpa_train = hpa_all[train_idx, :]
        hpa_val = hpa_all[val_idx, :]

        emb_pca_train, pca_artifacts = _fit_embedding_pca(
            emb_train,
            pca_dim=int(pca_dim),
            random_state=int(random_state) + int(fold_idx),
        )
        emb_pca_val = _transform_embedding_pca(emb_val, pca_artifacts)

        x_train = np.column_stack([hpa_train, emb_pca_train])
        x_val = np.column_stack([hpa_val, emb_pca_val])

        model = _build_logistic_model(
            penalty=penalty,
            c_value=float(c_value),
            l1_ratio=float(l1_ratio),
            max_iter=int(max_iter),
            random_state=int(random_state) + int(fold_idx),
        )
        model.fit(x_train, y_train)
        y_score = model.predict_proba(x_val)[:, 1]
        oof_scores[val_idx] = y_score

        if len(np.unique(y_val)) == 2:
            roc = float(roc_auc_score(y_val, y_score))
            pr = float(average_precision_score(y_val, y_score))
        else:
            roc = float("nan")
            pr = float("nan")

        fold_rows.append(
            {
                "fold": float(fold_idx),
                "n_train": float(len(train_idx)),
                "n_val": float(len(val_idx)),
                "n_pos_train": float(int(y_train.sum())),
                "n_pos_val": float(int(y_val.sum())),
                "roc_auc": roc,
                "pr_auc": pr,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    roc_vals = fold_df["roc_auc"].dropna().to_numpy(dtype=float)
    pr_vals = fold_df["pr_auc"].dropna().to_numpy(dtype=float)

    valid_mask = np.isfinite(oof_scores)
    if valid_mask.sum() > 0 and len(np.unique(y[valid_mask])) == 2:
        oof_roc = float(roc_auc_score(y[valid_mask], oof_scores[valid_mask]))
        oof_pr = float(average_precision_score(y[valid_mask], oof_scores[valid_mask]))
    else:
        oof_roc = float("nan")
        oof_pr = float("nan")

    summary = {
        "roc_auc_mean": float(np.mean(roc_vals)) if roc_vals.size else float("nan"),
        "roc_auc_std": float(np.std(roc_vals, ddof=1)) if roc_vals.size > 1 else 0.0,
        "pr_auc_mean": float(np.mean(pr_vals)) if pr_vals.size else float("nan"),
        "pr_auc_std": float(np.std(pr_vals, ddof=1)) if pr_vals.size > 1 else 0.0,
        "oof_roc_auc": float(oof_roc),
        "oof_pr_auc": float(oof_pr),
    }

    return CVResult(
        fold_metrics_df=fold_df,
        oof_scores=oof_scores,
        summary=summary,
        n_splits=int(max_splits),
    )



def _plot_score_distributions(df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}

    score_path = out_dir / "score_distribution_all_genes.png"
    plt.figure(figsize=(7, 4))
    plt.hist(df["predicted_score"].astype(float), bins=40, color="#4C78A8", alpha=0.85)
    plt.xlabel("Predicted score")
    plt.ylabel("Gene count")
    plt.title("Global score distribution (all genes)")
    plt.tight_layout()
    plt.savefig(score_path, dpi=180)
    plt.close()
    paths["score_distribution_all_genes"] = str(score_path)

    by_label_path = out_dir / "score_distribution_by_label.png"
    plt.figure(figsize=(7, 4))
    neg = df.loc[df["label_positive"] == 0, "predicted_score"].astype(float)
    pos = df.loc[df["label_positive"] == 1, "predicted_score"].astype(float)
    if len(neg) > 0:
        plt.hist(neg, bins=40, alpha=0.7, label="Negatives", color="#9FB1C5")
    if len(pos) > 0:
        plt.hist(pos, bins=20, alpha=0.8, label="Positives", color="#E45756")
    plt.xlabel("Predicted score")
    plt.ylabel("Gene count")
    plt.title("Score distribution by label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(by_label_path, dpi=180)
    plt.close()
    paths["score_distribution_by_label"] = str(by_label_path)

    return paths



def _plot_coefficients(coef_df: pd.DataFrame, out_path: Path, title: str, top_n: int = 25) -> None:
    cur = coef_df.copy()
    cur["abs_coefficient"] = pd.to_numeric(cur["abs_coefficient"], errors="coerce")
    cur = cur.sort_values("abs_coefficient", ascending=False, kind="stable").head(int(top_n)).copy()
    cur = cur.iloc[::-1]

    plt.figure(figsize=(8, max(4, 0.22 * len(cur))))
    plt.barh(cur["feature"].astype(str), cur["coefficient"].astype(float), color="#4C78A8")
    plt.xlabel("Coefficient")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {args.embedding_path}")
    if not args.hpa_path.exists():
        raise FileNotFoundError(f"HPA file not found: {args.hpa_path}")

    emb_df, emb_stats, emb_cols = _load_embedding_table(args.embedding_path)
    hpa_df, hpa_stats = _load_hpa_gene_table(args.hpa_path, expression_threshold=float(args.hpa_expression_threshold))
    hgnc_map = _load_hgnc_symbol_to_ensembl(args.hgnc_path)

    universe = emb_df.merge(hpa_df, on="gene_symbol", how="left", suffixes=("", "_hpa"))
    if "gene_id" not in universe.columns:
        universe["gene_id"] = None
    universe["gene_id"] = universe["gene_id"].map(_normalize_gene_id_strict)

    for col in HPA_FEATURES + ["has_hpa_expression_evidence"]:
        if col not in universe.columns:
            universe[col] = 0.0
        universe[col] = pd.to_numeric(universe[col], errors="coerce").fillna(0.0)

    missing_gene_id = universe["gene_id"].isna() | (universe["gene_id"].astype(str).str.strip() == "")
    universe.loc[missing_gene_id, "gene_id"] = universe.loc[missing_gene_id, "gene_symbol"].map(hgnc_map)

    universe["gene_id"] = universe["gene_id"].map(_normalize_gene_id_strict)
    universe["gene_id_mapping_source"] = np.where(
        universe["gene_id"].isna(),
        "unmapped",
        np.where(missing_gene_id, "hgnc_symbol", "hpa_or_embedding"),
    )

    positive_set = {normalize_gene_symbol(g) for g in VALIDATION_GENES if normalize_gene_symbol(g)}
    universe["label_positive"] = universe["gene_symbol"].isin(positive_set).astype(int)

    total_embeddings = int(len(emb_df))
    total_hpa_unique_genes = int(hpa_df["gene_symbol"].nunique()) if not hpa_df.empty else 0
    final_genes = int(len(universe))
    positive_genes = int(universe["label_positive"].sum())
    positive_genes_symbols = sorted(universe.loc[universe["label_positive"] == 1, "gene_symbol"].astype(str).tolist())
    missing_positive_symbols = sorted(list(positive_set.difference(set(universe["gene_symbol"].astype(str).tolist()))))

    emb_matrix = _to_numeric_array(universe, emb_cols)
    hpa_matrix = _to_numeric_array(universe, HPA_FEATURES)

    emb_pca, pca_artifacts = _fit_embedding_pca(
        emb_matrix,
        pca_dim=int(args.pca_dim),
        random_state=int(args.random_state),
    )
    pca_feature_names = [f"emb_pca_{i:03d}" for i in range(int(pca_artifacts.n_components))]

    x_final = np.column_stack([hpa_matrix, emb_pca])
    y = universe["label_positive"].astype(int).to_numpy()
    feature_names = list(HPA_FEATURES) + pca_feature_names

    model = _build_logistic_model(
        penalty=str(args.penalty),
        c_value=float(args.regularization_strength),
        l1_ratio=float(args.l1_ratio),
        max_iter=int(args.max_iter),
        random_state=int(args.random_state),
    )
    model.fit(x_final, y)
    score = model.predict_proba(x_final)[:, 1]
    linear = model.decision_function(x_final)

    universe_out = universe[["gene_id", "gene_symbol", "label_positive"] + HPA_FEATURES].copy()
    for i, col in enumerate(pca_feature_names):
        universe_out[col] = emb_pca[:, i]
    universe_out["predicted_linear_score"] = linear
    universe_out["predicted_score"] = score
    universe_out = universe_out.sort_values("predicted_score", ascending=False, kind="stable").reset_index(drop=True)

    coef = model.named_steps["logreg"].coef_.ravel()
    intercept = float(model.named_steps["logreg"].intercept_.ravel()[0])
    coef_rows: List[Dict[str, object]] = []
    for feat, w in zip(feature_names, coef):
        coef_rows.append(
            {
                "feature": feat,
                "coefficient": float(w),
                "abs_coefficient": float(abs(w)),
                "feature_group": "hpa" if feat in HPA_FEATURES else "embedding_pca",
                "is_nonzero": int(abs(float(w)) > 1e-12),
            }
        )
    coef_df = pd.DataFrame(coef_rows).sort_values("abs_coefficient", ascending=False, kind="stable")

    # Auxiliary L1 model to report sparse survivors.
    l1_model = _build_logistic_model(
        penalty="l1",
        c_value=float(args.l1_survival_c),
        l1_ratio=float(args.l1_ratio),
        max_iter=int(args.max_iter),
        random_state=int(args.random_state),
    )
    l1_model.fit(x_final, y)
    l1_coef = l1_model.named_steps["logreg"].coef_.ravel()
    l1_intercept = float(l1_model.named_steps["logreg"].intercept_.ravel()[0])

    l1_rows: List[Dict[str, object]] = []
    for feat, w in zip(feature_names, l1_coef):
        l1_rows.append(
            {
                "feature": feat,
                "coefficient": float(w),
                "abs_coefficient": float(abs(w)),
                "feature_group": "hpa" if feat in HPA_FEATURES else "embedding_pca",
                "is_nonzero": int(abs(float(w)) > 1e-12),
            }
        )
    l1_coef_df = pd.DataFrame(l1_rows).sort_values("abs_coefficient", ascending=False, kind="stable")
    l1_nonzero_df = l1_coef_df.loc[l1_coef_df["is_nonzero"] == 1].copy()

    # Optional CV diagnostics.
    cv_result: Optional[CVResult] = None
    if bool(args.train_cv):
        cv_result = _run_cv(
            universe,
            emb_cols=emb_cols,
            hpa_cols=HPA_FEATURES,
            pca_dim=int(args.pca_dim),
            penalty=str(args.penalty),
            c_value=float(args.regularization_strength),
            l1_ratio=float(args.l1_ratio),
            max_iter=int(args.max_iter),
            random_state=int(args.random_state),
            cv_folds=int(args.cv_folds),
        )

    threshold_info = _choose_high_confidence_threshold(
        gene_scores_desc=universe_out["predicted_score"].astype(float).tolist(),
        top_window=int(args.high_confidence_top_window),
        min_threshold=float(args.high_confidence_min_threshold),
        max_threshold=float(args.high_confidence_max_threshold),
        override=args.high_confidence_threshold,
    )
    high_conf_df = universe_out.loc[universe_out["predicted_score"] >= float(threshold_info.threshold)].copy()
    high_conf_df = high_conf_df.sort_values("predicted_score", ascending=False, kind="stable").reset_index(drop=True)

    top_200 = universe_out.head(200).copy()

    # Persist tables.
    universe_out.to_csv(args.out_dir / "all_gene_predictions.csv", index=False)
    top_200.to_csv(args.out_dir / "top_200_genes.csv", index=False)
    high_conf_df.to_csv(args.out_dir / "high_confidence_genes.csv", index=False)
    coef_df.to_csv(args.out_dir / "coefficient_table.csv", index=False)
    l1_coef_df.to_csv(args.out_dir / "l1_coefficient_table.csv", index=False)
    l1_nonzero_df.to_csv(args.out_dir / "l1_nonzero_features.csv", index=False)

    # Keep full universe with raw embedding features for reproducibility.
    universe_raw = universe[["gene_id", "gene_symbol", "label_positive", "gene_id_mapping_source"] + HPA_FEATURES + emb_cols].copy()
    universe_raw.to_csv(args.out_dir / "candidate_universe_embeddings_hpa.csv", index=False)

    # PCA artifacts for reproducibility.
    np.savez_compressed(
        args.out_dir / "pca_artifacts.npz",
        pca_components=pca_artifacts.pca.components_.astype(np.float32),
        pca_mean=pca_artifacts.pca.mean_.astype(np.float32),
        pca_explained_variance_ratio=pca_artifacts.explained_variance_ratio.astype(np.float32),
        emb_scaler_mean=pca_artifacts.emb_scaler.mean_.astype(np.float32),
        emb_scaler_scale=pca_artifacts.emb_scaler.scale_.astype(np.float32),
        embedding_feature_names=np.asarray(emb_cols, dtype=object),
        pca_feature_names=np.asarray(pca_feature_names, dtype=object),
    )

    plots = _plot_score_distributions(universe_out, args.out_dir)
    _plot_coefficients(
        coef_df,
        args.out_dir / "coefficient_magnitude_top25.png",
        title=f"Top coefficient magnitudes ({args.penalty.upper()})",
    )
    if not l1_nonzero_df.empty:
        _plot_coefficients(
            l1_nonzero_df,
            args.out_dir / "l1_surviving_coefficients.png",
            title=f"L1-surviving coefficients (C={float(args.l1_survival_c):.3g})",
            top_n=min(30, len(l1_nonzero_df)),
        )

    if cv_result is not None:
        cv_result.fold_metrics_df.to_csv(args.out_dir / "cv_fold_metrics.csv", index=False)

    universe_summary = {
        "total_genes_with_embeddings": total_embeddings,
        "total_genes_with_hpa": total_hpa_unique_genes,
        "total_genes_final_model": final_genes,
        "total_positive_genes_in_final_model": positive_genes,
        "reference_positive_genes_total": int(len(positive_set)),
        "reference_positive_genes_present": positive_genes_symbols,
        "reference_positive_genes_missing": missing_positive_symbols,
        "genes_with_hpa_evidence_in_final_model": int((universe["has_hpa_expression_evidence"] > 0).sum()),
        "genes_without_hpa_evidence_in_final_model": int((universe["has_hpa_expression_evidence"] <= 0).sum()),
        "gene_id_mapping_source_counts": {
            str(k): int(v)
            for k, v in universe["gene_id_mapping_source"].value_counts(dropna=False).to_dict().items()
        },
    }

    model_summary = {
        "experiment_name": args.out_dir.name,
        "training_type": "global_gene_classification",
        "feature_set": ["hpa_brain_expression_value", "hpa_muscle_expression_value"] + pca_feature_names,
        "embedding_pca_components_requested": int(args.pca_dim),
        "embedding_pca_components_used": int(pca_artifacts.n_components),
        "embedding_pca_explained_variance_ratio_sum": float(np.sum(pca_artifacts.explained_variance_ratio)),
        "penalty": str(args.penalty),
        "regularization_strength_C": float(args.regularization_strength),
        "l1_ratio": float(args.l1_ratio),
        "max_iter": int(args.max_iter),
        "random_state": int(args.random_state),
        "intercept": float(intercept),
        "l1_survival_c": float(args.l1_survival_c),
        "l1_intercept": float(l1_intercept),
        "l1_surviving_feature_count": int(len(l1_nonzero_df)),
        "high_confidence_threshold": {
            "threshold": float(threshold_info.threshold),
            "method": threshold_info.method,
            "gap_index": threshold_info.gap_index,
            "gap_high_score": threshold_info.gap_high_score,
            "gap_low_score": threshold_info.gap_low_score,
            "largest_gap": threshold_info.largest_gap,
        },
        "high_confidence_gene_count": int(len(high_conf_df)),
        "score_quantiles": {
            str(k): float(v)
            for k, v in universe_out["predicted_score"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict().items()
        },
        "n_genes_score_ge_0_9": int((universe_out["predicted_score"] >= 0.9).sum()),
        "n_genes_score_ge_0_8": int((universe_out["predicted_score"] >= 0.8).sum()),
        "n_genes_score_ge_0_7": int((universe_out["predicted_score"] >= 0.7).sum()),
        "n_genes_score_ge_0_6": int((universe_out["predicted_score"] >= 0.6).sum()),
    }

    cv_summary = None
    if cv_result is not None:
        cv_summary = {
            "enabled": True,
            "n_splits": int(cv_result.n_splits),
            **cv_result.summary,
        }
    else:
        cv_summary = {"enabled": False}

    run_summary = {
        "universe_summary": universe_summary,
        "model_summary": model_summary,
        "cv_summary": cv_summary,
        "embedding_source": {
            "path": str(args.embedding_path),
            "stats": emb_stats,
        },
        "hpa_source": {
            "path": str(args.hpa_path),
            "stats": hpa_stats,
        },
        "outputs": {
            "out_dir": str(args.out_dir),
            "all_gene_predictions_csv": str(args.out_dir / "all_gene_predictions.csv"),
            "high_confidence_genes_csv": str(args.out_dir / "high_confidence_genes.csv"),
            "coefficient_table_csv": str(args.out_dir / "coefficient_table.csv"),
            "l1_nonzero_features_csv": str(args.out_dir / "l1_nonzero_features.csv"),
            "top_200_genes_csv": str(args.out_dir / "top_200_genes.csv"),
            "score_distribution_all_genes_png": plots["score_distribution_all_genes"],
            "score_distribution_by_label_png": plots["score_distribution_by_label"],
        },
    }

    (args.out_dir / "universe_summary.json").write_text(json.dumps(universe_summary, indent=2), encoding="utf-8")
    (args.out_dir / "model_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
    (args.out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    feature_manifest = {
        "model_feature_names": ["hpa_brain_expression_value", "hpa_muscle_expression_value"] + pca_feature_names,
        "hpa_features": HPA_FEATURES,
        "embedding_pca_features": pca_feature_names,
        "excluded_feature_families": [
            "distance_to_gene",
            "distance_to_tss",
            "qtl_features",
            "colocalisation_features",
            "all_other_gwas_derived_features",
        ],
    }
    (args.out_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

    report_lines = [
        "# Global Functional Model Report (HPA + PCA)",
        "",
        "## Rationale",
        "- This experiment trains a functional-prior model using orthogonal evidence: text embeddings + HPA expression.",
        "- GWAS-derived features were excluded from model training.",
        "",
        "## Candidate Universe",
        f"- Genes with embeddings: `{total_embeddings}`",
        f"- Genes with HPA context available in source table: `{total_hpa_unique_genes}`",
        f"- Final genes used in model (embedding-defined universe): `{final_genes}`",
        f"- Positive genes in final universe (from `config.VALIDATION_GENES`): `{positive_genes}`",
        "",
        "## Features Used",
        "- `hpa_brain_expression_value`",
        "- `hpa_muscle_expression_value`",
        f"- `emb_pca_000..emb_pca_{int(pca_artifacts.n_components)-1:03d}` ({int(pca_artifacts.n_components)} components)",
        "",
        "## Training Setup",
        "- Model: logistic regression",
        f"- Penalty: `{args.penalty}`",
        f"- C: `{float(args.regularization_strength):.4g}`",
        f"- Class weight: `balanced`",
        f"- PCA explained variance sum: `{float(np.sum(pca_artifacts.explained_variance_ratio)):.4f}`",
        "",
        "## High-confidence Genes",
        f"- Threshold method: `{threshold_info.method}`",
        f"- Threshold used: `{float(threshold_info.threshold):.2f}`",
        f"- Number of high-confidence genes: `{len(high_conf_df)}`",
        "",
        "## Key Outputs",
        "- `all_gene_predictions.csv`",
        "- `high_confidence_genes.csv`",
        "- `coefficient_table.csv`",
        "- `l1_nonzero_features.csv`",
        "- `score_distribution_all_genes.png`",
        "- `score_distribution_by_label.png`",
    ]

    if cv_result is not None:
        report_lines.extend(
            [
                "",
                "## Cross-validation (Auxiliary)",
                f"- Splits: `{int(cv_result.n_splits)}`",
                f"- Mean ROC-AUC: `{cv_result.summary['roc_auc_mean']:.4f}`",
                f"- Mean PR-AUC: `{cv_result.summary['pr_auc_mean']:.4f}`",
                f"- OOF ROC-AUC: `{cv_result.summary['oof_roc_auc']:.4f}`",
                f"- OOF PR-AUC: `{cv_result.summary['oof_pr_auc']:.4f}`",
            ]
        )

    (args.out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Global functional model run completed.")
    print(f"Output directory: {args.out_dir}")
    print(f"Final genes: {final_genes}")
    print(f"Positive genes in universe: {positive_genes}")
    print(f"High-confidence genes: {len(high_conf_df)} (threshold={float(threshold_info.threshold):.2f})")


if __name__ == "__main__":
    main()

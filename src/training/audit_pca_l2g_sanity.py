#!/usr/bin/env python3
"""
Audit PCA-based locus-to-gene ranking for leakage/sanity issues.

This script performs:
1) Fold-by-fold PCA audit tables per validation mode.
2) Explicit train/test overlap checks (gene/locus/identifier leakage).
3) Single-feature separability analysis (baseline + has_gene_embedding + PCA comps).
4) Label-permutation sanity checks (>=20 seeds).
5) Random ranking baseline comparisons.
6) Per-locus inspection exports.
7) Small-data warning diagnostics.
8) Final markdown report with quantitative conclusions.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import sys

TRAINING_DIR = Path("/home/viguinijpv/200.18.99.75:8000/IC/src/training")
if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

import train_locus_gene_ranker as ranker  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)


DEFAULT_EVAL_DIR = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/"
    "locus_gene_ranker_leakage_eval_20260324"
)
DEFAULT_INPUT_TABLE = ranker.DEFAULT_INPUT_TABLE
CV_MODES = ["lolo", "gene_grouped", "lolo_gene_exclusion"]
MODE = "pca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PCA-based locus-to-gene ranking sanity/leakage.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="Directory with cv_* outputs.")
    parser.add_argument("--input-table", type=Path, default=DEFAULT_INPUT_TABLE, help="Original feature table.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for audit artifacts. Defaults to eval-dir.",
    )
    parser.add_argument("--n-permutations", type=int, default=20, help="Number of shuffled-label runs per CV mode.")
    parser.add_argument(
        "--n-random-ranking-seeds",
        type=int,
        default=200,
        help="Number of random-ranking seeds per CV mode.",
    )
    parser.add_argument(
        "--gene-grouped-n-splits",
        type=int,
        default=5,
        help="Fold target for gene_grouped mode (must match training setup).",
    )
    parser.add_argument(
        "--penalty",
        choices=["none", "l1", "elasticnet"],
        default="l1",
        help="Penalty used for permutation reruns (match training setting).",
    )
    parser.add_argument("--regularization-strength", type=float, default=0.1, help="C for LogisticRegression.")
    parser.add_argument("--l1-ratio", type=float, default=0.5, help="l1_ratio for elasticnet.")
    parser.add_argument("--max-iter", type=int, default=10000, help="Max iterations for logistic regression.")
    parser.add_argument("--model-random-state", type=int, default=42, help="Random state for model fitting/PCA.")
    return parser.parse_args()


def float_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def nanstd(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1))


def qtile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def to_numeric_series(series: pd.Series) -> pd.Series:
    return ranker._to_numeric_feature(series).astype(float)  # pylint: disable=protected-access


def make_row_key(df: pd.DataFrame) -> pd.Series:
    gid = df.get(ranker.GENE_ID_COL, pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.strip()
    gsym = df.get(ranker.GENE_SYMBOL_COL, pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.strip()
    gene = gid.where(gid != "", gsym)
    locus = df[ranker.LOCUS_COL].fillna("").astype(str)
    return locus + "||" + gene


def load_mode_tables(eval_dir: Path, cv_mode: str, mode: str = MODE) -> Dict[str, pd.DataFrame]:
    mode_dir = eval_dir / f"cv_{cv_mode}" / f"mode_{mode}"
    if not mode_dir.exists():
        raise FileNotFoundError(f"Missing mode directory: {mode_dir}")
    summary_path = mode_dir / "summary_metrics.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = pd.DataFrame([json.load(f)])
    return {
        "summary": summary,
        "fold_metrics": pd.read_csv(mode_dir / "fold_metrics.csv"),
        "all_predictions": pd.read_csv(mode_dir / "all_ranked_predictions.csv"),
        "coefficients": pd.read_csv(mode_dir / "coefficient_table.csv"),
        "pca_evr": pd.read_csv(mode_dir / "pca_explained_variance.csv"),
    }


def detect_pca_dim(eval_dir: Path) -> int:
    sample = eval_dir / "cv_lolo" / "mode_pca" / "pca_explained_variance.csv"
    if not sample.exists():
        return 16
    p = pd.read_csv(sample)
    if "component" not in p.columns:
        return int(len(p))
    return int(p["component"].astype(str).nunique())


def compute_rank_metrics(pred_df: pd.DataFrame, label_col: str = ranker.LABEL_COL) -> Dict[str, float]:
    if pred_df.empty:
        return {
            "mean_fold_roc_auc": float("nan"),
            "mean_fold_pr_auc": float("nan"),
            "mean_recall_at_1": float("nan"),
            "mean_recall_at_3": float("nan"),
            "mean_mrr": float("nan"),
            "pooled_roc_auc": float("nan"),
            "pooled_pr_auc": float("nan"),
            "folds_total": 0,
            "folds_ok": 0,
        }

    fold_rows: List[Dict[str, float]] = []
    for fold_index, d_fold in pred_df.groupby("fold_index", sort=True):
        y = d_fold[label_col].astype(int).to_numpy()
        s = d_fold["predicted_score"].astype(float).to_numpy()
        roc = safe_roc_auc(y, s)
        pr = safe_pr_auc(y, s)
        pos = d_fold[d_fold[label_col].astype(int) == 1]
        if pos.empty:
            r1 = float("nan")
            r3 = float("nan")
            mrr = float("nan")
        else:
            ranks = pos["rank_within_locus"].astype(int).to_numpy()
            r1 = float(np.mean(ranks <= 1))
            r3 = float(np.mean(ranks <= 3))
            mrr = float(1.0 / ranks.min())
        fold_rows.append(
            {
                "fold_index": float(fold_index),
                "roc_auc": roc,
                "pr_auc": pr,
                "recall_at_1": r1,
                "recall_at_3": r3,
                "mrr": mrr,
                "status_ok": 1.0,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    pooled_roc = safe_roc_auc(
        pred_df[label_col].astype(int).to_numpy(),
        pred_df["predicted_score"].astype(float).to_numpy(),
    )
    pooled_pr = safe_pr_auc(
        pred_df[label_col].astype(int).to_numpy(),
        pred_df["predicted_score"].astype(float).to_numpy(),
    )
    return {
        "mean_fold_roc_auc": nanmean(fold_df["roc_auc"].tolist()),
        "mean_fold_pr_auc": nanmean(fold_df["pr_auc"].tolist()),
        "mean_recall_at_1": nanmean(fold_df["recall_at_1"].tolist()),
        "mean_recall_at_3": nanmean(fold_df["recall_at_3"].tolist()),
        "mean_mrr": nanmean(fold_df["mrr"].tolist()),
        "pooled_roc_auc": pooled_roc,
        "pooled_pr_auc": pooled_pr,
        "folds_total": int(len(fold_df)),
        "folds_ok": int(fold_df["status_ok"].sum()),
    }


def format_positive_rows(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    chunks: List[str] = []
    for row in df.itertuples(index=False):
        chunks.append(
            f"{getattr(row, ranker.GENE_SYMBOL_COL)}"
            f"(locus={getattr(row, ranker.LOCUS_COL)},rank={int(getattr(row, 'rank_within_locus'))},"
            f"score={float(getattr(row, 'predicted_score')):.6f})"
        )
    return "; ".join(chunks)


def build_fold_audit_csv(cv_mode: str, pred_df: pd.DataFrame, fold_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    pred_df = pred_df.copy()
    pred_df[ranker.LABEL_COL] = pred_df[ranker.LABEL_COL].astype(int)
    pred_df["predicted_score"] = pred_df["predicted_score"].astype(float)
    pred_df["rank_within_locus"] = pred_df["rank_within_locus"].astype(int)

    fold_info = fold_df.set_index("fold_index").to_dict(orient="index")
    rows: List[Dict[str, object]] = []

    for fold_index, d_fold in pred_df.groupby("fold_index", sort=True):
        d_fold = d_fold.copy()
        d_pos = d_fold[d_fold[ranker.LABEL_COL] == 1]
        n_loci_test = int(d_fold[ranker.LOCUS_COL].astype(str).nunique())
        fold_pos_details = format_positive_rows(
            d_pos[
                [
                    ranker.GENE_SYMBOL_COL,
                    ranker.LOCUS_COL,
                    "rank_within_locus",
                    "predicted_score",
                ]
            ]
        )
        meta = fold_info.get(int(fold_index), {})

        for locus_id, d_locus in d_fold.groupby(ranker.LOCUS_COL, sort=True):
            d_locus = d_locus.sort_values("predicted_score", ascending=False, kind="stable").reset_index(drop=True)
            top1 = d_locus.iloc[0]
            top2_score = float(d_locus.iloc[1]["predicted_score"]) if len(d_locus) > 1 else float("nan")
            top_gap = float(top1["predicted_score"] - top2_score) if np.isfinite(top2_score) else float("nan")
            d_pos_locus = d_locus[d_locus[ranker.LABEL_COL] == 1]
            pos_locus_details = format_positive_rows(
                d_pos_locus[
                    [
                        ranker.GENE_SYMBOL_COL,
                        ranker.LOCUS_COL,
                        "rank_within_locus",
                        "predicted_score",
                    ]
                ]
            )
            rows.append(
                {
                    "cv_mode": cv_mode,
                    "fold_index": int(fold_index),
                    "fold_id": str(meta.get("fold_id", "")),
                    "n_train_rows": int(meta.get("n_train_rows", np.nan)),
                    "n_test_rows": int(meta.get("n_test_rows", np.nan)),
                    "n_positive_rows_test_fold": int(d_pos.shape[0]),
                    "n_unique_positive_genes_test_fold": int(d_pos[ranker.GENE_ID_COL].astype(str).nunique()),
                    "n_loci_in_test_fold": n_loci_test,
                    "gwas_study_locus_id": str(locus_id),
                    "n_candidate_genes_in_test_locus": int(len(d_locus)),
                    "positive_gene_details_test_fold": fold_pos_details,
                    "positive_gene_details_test_locus": pos_locus_details,
                    "top1_gene_id": str(top1[ranker.GENE_ID_COL]),
                    "top1_gene_symbol": str(top1[ranker.GENE_SYMBOL_COL]),
                    "top1_score": float(top1["predicted_score"]),
                    "top1_is_positive": int(top1[ranker.LABEL_COL]),
                    "top2_score": float_or_nan(top2_score),
                    "top1_minus_top2_score_gap": float_or_nan(top_gap),
                }
            )

    out = pd.DataFrame(rows).sort_values(
        ["fold_index", "gwas_study_locus_id"], ascending=[True, True], kind="stable"
    )
    out_path = out_dir / f"pca_fold_audit_{cv_mode}.csv"
    out.to_csv(out_path, index=False)
    return out


def leakage_overlap_for_cv(
    cv_mode: str,
    full_df: pd.DataFrame,
    gene_group: pd.Series,
    pred_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    full = full_df.copy()
    full["_row_key"] = make_row_key(full)
    full["_gene_group"] = gene_group.astype(str).values

    pred = pred_df.copy()
    pred["_row_key"] = make_row_key(pred)

    other_identifier_cols = [
        c
        for c in [
            "gwas_lead_variant_id",
            "gwas_study_id",
            "gwas_lead_variant_rsids",
            "gwas_cs_region",
            "gwas_cs_index",
        ]
        if c in full.columns
    ]

    rows: List[Dict[str, object]] = []
    for fold in fold_df.itertuples(index=False):
        fold_idx = int(fold.fold_index)
        d_test_pred = pred[pred["fold_index"] == fold_idx]
        test_keys = set(d_test_pred["_row_key"].astype(str).tolist())

        test_df = full[full["_row_key"].isin(test_keys)].copy()
        train_df = full[~full["_row_key"].isin(test_keys)].copy()
        if cv_mode == "lolo_gene_exclusion":
            test_genes = set(test_df["_gene_group"].astype(str).tolist())
            train_df = train_df[~train_df["_gene_group"].astype(str).isin(test_genes)].copy()

        overlap_gene_id = set(train_df[ranker.GENE_ID_COL].astype(str)).intersection(set(test_df[ranker.GENE_ID_COL].astype(str)))
        overlap_gene_symbol = set(train_df[ranker.GENE_SYMBOL_COL].astype(str)).intersection(
            set(test_df[ranker.GENE_SYMBOL_COL].astype(str))
        )
        overlap_locus = set(train_df[ranker.LOCUS_COL].astype(str)).intersection(set(test_df[ranker.LOCUS_COL].astype(str)))

        row = {
            "cv_mode": cv_mode,
            "fold_index": fold_idx,
            "fold_id": str(fold.fold_id),
            "n_train_rows_reconstructed": int(len(train_df)),
            "n_test_rows_reconstructed": int(len(test_df)),
            "overlap_gene_id_count": int(len(overlap_gene_id)),
            "overlap_gene_symbol_count": int(len(overlap_gene_symbol)),
            "overlap_locus_id_count": int(len(overlap_locus)),
            "overlap_gene_id_examples": "; ".join(sorted(list(overlap_gene_id))[:10]),
            "overlap_gene_symbol_examples": "; ".join(sorted(list(overlap_gene_symbol))[:10]),
            "overlap_locus_id_examples": "; ".join(sorted(list(overlap_locus))[:10]),
        }

        other_any = 0
        for col in other_identifier_cols:
            inter = set(train_df[col].astype(str)).intersection(set(test_df[col].astype(str)))
            row[f"overlap_{col}_count"] = int(len(inter))
            row[f"overlap_{col}_examples"] = "; ".join(sorted(list(inter))[:10])
            if len(inter) > 0:
                other_any = 1
        row["overlap_any_other_identifier"] = int(other_any)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("fold_index", kind="stable")
    out_path = out_dir / f"pca_train_test_overlap_{cv_mode}.csv"
    out.to_csv(out_path, index=False)
    return out


def single_feature_separability(feature_name: str, values: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    pos = values[y == 1]
    neg = values[y == 0]
    if pos.size == 0 or neg.size == 0:
        return {
            "feature": feature_name,
            "n_pos": int(pos.size),
            "n_neg": int(neg.size),
            "pos_mean": float("nan"),
            "neg_mean": float("nan"),
            "pos_std": float("nan"),
            "neg_std": float("nan"),
            "pos_min": float("nan"),
            "pos_max": float("nan"),
            "neg_min": float("nan"),
            "neg_max": float("nan"),
            "single_feature_auc": float("nan"),
            "interval_overlap": float("nan"),
            "almost_perfect_separator": 0,
        }

    auc_raw = safe_roc_auc(y, values)
    auc_abs = float(max(auc_raw, 1.0 - auc_raw)) if np.isfinite(auc_raw) else float("nan")
    pos_min, pos_max = float(pos.min()), float(pos.max())
    neg_min, neg_max = float(neg.min()), float(neg.max())
    overlap = max(0.0, min(pos_max, neg_max) - max(pos_min, neg_min))
    almost_perfect = int((np.isfinite(auc_abs) and auc_abs >= 0.98) or overlap == 0.0)
    return {
        "feature": feature_name,
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
        "pos_mean": float(pos.mean()),
        "neg_mean": float(neg.mean()),
        "pos_std": float(pos.std(ddof=1)) if pos.size > 1 else float("nan"),
        "neg_std": float(neg.std(ddof=1)) if neg.size > 1 else float("nan"),
        "pos_min": pos_min,
        "pos_max": pos_max,
        "neg_min": neg_min,
        "neg_max": neg_max,
        "single_feature_auc": float_or_nan(auc_abs),
        "interval_overlap": float_or_nan(overlap),
        "almost_perfect_separator": almost_perfect,
    }


def feature_separability_analysis(
    full_df: pd.DataFrame,
    pca_dim: int,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = full_df[ranker.LABEL_COL].astype(int).to_numpy()
    baseline_features = list(ranker.BASELINE_GENETIC_FEATURES)
    if ranker.EMBEDDING_INDICATOR_FEATURE in full_df.columns:
        baseline_features.append(ranker.EMBEDDING_INDICATOR_FEATURE)

    rows: List[Dict[str, object]] = []
    for feat in baseline_features:
        vals = to_numeric_series(full_df[feat]).fillna(0.0).to_numpy(dtype=np.float64)
        rows.append(single_feature_separability(feat, vals, y))

    embedding_cols = sorted([c for c in full_df.columns if c.startswith("gene_emb_")])
    if embedding_cols:
        x_emb = ranker.as_numeric_matrix(full_df, embedding_cols, fill_value=0.0)
        n_components = int(min(pca_dim, x_emb.shape[1], x_emb.shape[0]))
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_emb)
        pca = PCA(n_components=n_components, random_state=42)
        x_pca = pca.fit_transform(x_scaled)
        for i in range(n_components):
            rows.append(single_feature_separability(f"emb_pca_{i:03d}", x_pca[:, i], y))

    summary_df = pd.DataFrame(rows).sort_values(
        ["almost_perfect_separator", "single_feature_auc"],
        ascending=[False, False],
        kind="stable",
    )
    suspicious_df = summary_df[summary_df["almost_perfect_separator"].astype(int) == 1].copy()

    summary_df.to_csv(out_dir / "pca_feature_separability_summary.csv", index=False)
    suspicious_df.to_csv(out_dir / "pca_feature_separability_suspicious.csv", index=False)
    return summary_df, suspicious_df


def build_cv_args(gene_grouped_n_splits: int) -> SimpleNamespace:
    return SimpleNamespace(gene_grouped_n_splits=int(gene_grouped_n_splits))


def evaluate_pca_with_labels(
    full_df: pd.DataFrame,
    gene_series: pd.Series,
    labels: np.ndarray,
    cv_mode: str,
    pca_dim: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    cv_args = build_cv_args(args.gene_grouped_n_splits)
    folds = ranker.build_cv_folds(df=full_df, gene_series=gene_series, args=cv_args, cv_mode=cv_mode)
    embedding_cols = sorted([c for c in full_df.columns if c.startswith("gene_emb_")])
    baseline_cols = ranker.baseline_columns_for_mode("pca")

    pred_rows: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, float]] = []

    for fold in folds:
        train_df = full_df.iloc[fold.train_idx].reset_index(drop=True)
        test_df = full_df.iloc[fold.test_idx].reset_index(drop=True)
        y_train = labels[fold.train_idx]
        y_test = labels[fold.test_idx]

        if len(train_df) == 0 or len(test_df) == 0:
            continue
        if np.unique(y_train).size < 2:
            continue

        fb = ranker.ModeFeatureBuilder(
            baseline_cols=baseline_cols,
            embedding_cols=embedding_cols,
            mode="pca",
            pca_dim=int(pca_dim),
            random_state=int(args.model_random_state),
        )
        x_train = fb.fit_transform(train_df)
        x_test = fb.transform(test_df)

        model = ranker.build_model(
            penalty=args.penalty,
            c_value=float(args.regularization_strength),
            l1_ratio=float(args.l1_ratio),
            max_iter=int(args.max_iter),
            random_state=int(args.model_random_state),
        )
        model.fit(x_train, y_train)
        score = model.predict_proba(x_test)[:, 1]

        pred = test_df[[ranker.LOCUS_COL, ranker.GENE_ID_COL, ranker.GENE_SYMBOL_COL]].copy()
        pred["fold_index"] = int(fold.fold_index)
        pred[ranker.LABEL_COL] = y_test.astype(int)
        pred["predicted_score"] = score
        pred = pred.sort_values([ranker.LOCUS_COL, "predicted_score"], ascending=[True, False], kind="stable")
        pred["rank_within_locus"] = (
            pred.groupby([ranker.LOCUS_COL])["predicted_score"].rank(method="first", ascending=False).astype(int)
        )
        pred_rows.append(pred)

        roc = safe_roc_auc(y_test, score)
        pr = safe_pr_auc(y_test, score)
        pos = pred[pred[ranker.LABEL_COL].astype(int) == 1]
        if pos.empty:
            r1, r3, mrr = float("nan"), float("nan"), float("nan")
        else:
            ranks = pos["rank_within_locus"].astype(int).to_numpy()
            r1 = float(np.mean(ranks <= 1))
            r3 = float(np.mean(ranks <= 3))
            mrr = float(1.0 / ranks.min())
        fold_metrics.append({"roc_auc": roc, "pr_auc": pr, "recall_at_1": r1, "recall_at_3": r3, "mrr": mrr})

    if pred_rows:
        all_pred = pd.concat(pred_rows, axis=0, ignore_index=True)
    else:
        all_pred = pd.DataFrame(columns=[ranker.LABEL_COL, "predicted_score", "rank_within_locus"])
    fm = pd.DataFrame(fold_metrics)
    if fm.empty:
        return {
            "mean_fold_pr_auc": float("nan"),
            "mean_fold_roc_auc": float("nan"),
            "mean_recall_at_1": float("nan"),
            "mean_recall_at_3": float("nan"),
            "mean_mrr": float("nan"),
            "pooled_pr_auc": float("nan"),
            "pooled_roc_auc": float("nan"),
            "folds_ok": 0,
        }

    pooled_roc = safe_roc_auc(
        all_pred[ranker.LABEL_COL].astype(int).to_numpy(),
        all_pred["predicted_score"].astype(float).to_numpy(),
    )
    pooled_pr = safe_pr_auc(
        all_pred[ranker.LABEL_COL].astype(int).to_numpy(),
        all_pred["predicted_score"].astype(float).to_numpy(),
    )
    return {
        "mean_fold_pr_auc": nanmean(fm["pr_auc"].tolist()),
        "mean_fold_roc_auc": nanmean(fm["roc_auc"].tolist()),
        "mean_recall_at_1": nanmean(fm["recall_at_1"].tolist()),
        "mean_recall_at_3": nanmean(fm["recall_at_3"].tolist()),
        "mean_mrr": nanmean(fm["mrr"].tolist()),
        "pooled_pr_auc": pooled_pr,
        "pooled_roc_auc": pooled_roc,
        "folds_ok": int(len(fm)),
    }


def permutation_sanity_check(
    full_df: pd.DataFrame,
    out_dir: Path,
    pca_dim: int,
    args: argparse.Namespace,
    real_summary_by_cv: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_true = full_df[ranker.LABEL_COL].astype(int).to_numpy()
    gene_series = ranker.make_gene_group_series(full_df)

    rows: List[Dict[str, object]] = []
    for cv_mode in CV_MODES:
        for seed in range(int(args.n_permutations)):
            rng = np.random.default_rng(seed)
            y_perm = rng.permutation(y_true)
            m = evaluate_pca_with_labels(
                full_df=full_df,
                gene_series=gene_series,
                labels=y_perm,
                cv_mode=cv_mode,
                pca_dim=pca_dim,
                args=args,
            )
            rows.append({"cv_mode": cv_mode, "perm_seed": seed, **m})

    perm_df = pd.DataFrame(rows)
    perm_df.to_csv(out_dir / "pca_label_permutation_metrics.csv", index=False)

    summary_rows: List[Dict[str, object]] = []
    for cv_mode, d in perm_df.groupby("cv_mode", sort=True):
        real = real_summary_by_cv.get(cv_mode, {})
        row: Dict[str, object] = {"cv_mode": cv_mode}
        for metric in [
            "mean_fold_pr_auc",
            "mean_fold_roc_auc",
            "mean_recall_at_1",
            "mean_recall_at_3",
            "mean_mrr",
        ]:
            vals = d[metric].astype(float).to_numpy()
            row[f"perm_mean_{metric}"] = nanmean(vals)
            row[f"perm_std_{metric}"] = nanstd(vals)
            row[f"perm_q95_{metric}"] = qtile(vals, 0.95)
            row[f"real_{metric}"] = float_or_nan(float(real.get(metric, float("nan"))))
            row[f"real_minus_perm_mean_{metric}"] = float_or_nan(
                float(real.get(metric, float("nan"))) - nanmean(vals)
            )
            row[f"real_gt_perm_q95_{metric}"] = int(
                np.isfinite(float(real.get(metric, float("nan"))))
                and np.isfinite(qtile(vals, 0.95))
                and float(real.get(metric, float("nan"))) > qtile(vals, 0.95)
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("cv_mode", kind="stable")
    summary_df.to_csv(out_dir / "pca_label_permutation_summary.csv", index=False)
    return perm_df, summary_df


def random_ranking_metrics(pred_df: pd.DataFrame, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    d = pred_df.copy()
    d["predicted_score"] = rng.random(len(d))
    d["rank_within_locus"] = (
        d.groupby(["fold_index", ranker.LOCUS_COL])["predicted_score"].rank(method="first", ascending=False).astype(int)
    )
    return compute_rank_metrics(d)


def random_baseline_check(
    preds_by_cv: Dict[str, pd.DataFrame],
    out_dir: Path,
    n_seeds: int,
    baseline_summary_by_cv: Dict[str, Dict[str, float]],
    pca_summary_by_cv: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dist_rows: List[Dict[str, object]] = []
    for cv_mode in CV_MODES:
        pred_df = preds_by_cv[cv_mode]
        for seed in range(int(n_seeds)):
            m = random_ranking_metrics(pred_df, seed=seed)
            dist_rows.append({"cv_mode": cv_mode, "seed": seed, **m})
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(out_dir / "pca_random_ranking_distribution.csv", index=False)

    comp_rows: List[Dict[str, object]] = []
    for cv_mode, d in dist_df.groupby("cv_mode", sort=True):
        base = baseline_summary_by_cv.get(cv_mode, {})
        pca = pca_summary_by_cv.get(cv_mode, {})
        row: Dict[str, object] = {"cv_mode": cv_mode}
        for metric in ["mean_fold_pr_auc", "mean_fold_roc_auc", "mean_recall_at_1", "mean_recall_at_3", "mean_mrr"]:
            vals = d[metric].astype(float).to_numpy()
            row[f"random_mean_{metric}"] = nanmean(vals)
            row[f"random_std_{metric}"] = nanstd(vals)
            row[f"random_q95_{metric}"] = qtile(vals, 0.95)
            row[f"baseline_{metric}"] = float_or_nan(float(base.get(metric, float("nan"))))
            row[f"pca_{metric}"] = float_or_nan(float(pca.get(metric, float("nan"))))
            row[f"pca_minus_random_mean_{metric}"] = float_or_nan(
                float(pca.get(metric, float("nan"))) - nanmean(vals)
            )
            row[f"pca_minus_baseline_{metric}"] = float_or_nan(
                float(pca.get(metric, float("nan"))) - float(base.get(metric, float("nan")))
            )
        comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows).sort_values("cv_mode", kind="stable")
    comp_df.to_csv(out_dir / "pca_random_baseline_comparison.csv", index=False)
    return dist_df, comp_df


def export_per_locus_tables(cv_mode: str, pred_df: pd.DataFrame, out_dir: Path) -> Path:
    cols = [
        "fold_index",
        "fold_id",
        ranker.LOCUS_COL,
        ranker.GENE_ID_COL,
        ranker.GENE_SYMBOL_COL,
        ranker.LABEL_COL,
        "predicted_score",
        "rank_within_locus",
        "top_feature_contributions",
    ]
    keep = [c for c in cols if c in pred_df.columns]
    all_out = pred_df[keep].copy()
    all_out = all_out.sort_values(["fold_index", ranker.LOCUS_COL, "rank_within_locus"], kind="stable")
    all_out.to_csv(out_dir / f"pca_per_locus_inspection_{cv_mode}.csv", index=False)

    locus_dir = out_dir / f"pca_per_locus_tables_{cv_mode}"
    locus_dir.mkdir(parents=True, exist_ok=True)
    for (fold_idx, locus), d in all_out.groupby(["fold_index", ranker.LOCUS_COL], sort=True):
        safe_locus = str(locus).replace("/", "_")
        out = locus_dir / f"fold_{int(fold_idx):02d}_locus_{safe_locus}.csv"
        d.sort_values("rank_within_locus", kind="stable").to_csv(out, index=False)
    return locus_dir


def small_data_warning_analysis(
    fold_audit_by_cv: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cv_mode, fa in fold_audit_by_cv.items():
        if fa.empty:
            rows.append({"cv_mode": cv_mode})
            continue
        fold_level = (
            fa.groupby("fold_index", as_index=False)
            .agg(
                n_test_rows=("n_test_rows", "first"),
                n_train_rows=("n_train_rows", "first"),
                n_positive_rows_test_fold=("n_positive_rows_test_fold", "first"),
                n_unique_positive_genes_test_fold=("n_unique_positive_genes_test_fold", "first"),
                n_loci_in_test_fold=("n_loci_in_test_fold", "first"),
            )
            .copy()
        )
        pos_locus = fa[fa["positive_gene_details_test_locus"].astype(str).str.len() > 0].copy()

        rows.append(
            {
                "cv_mode": cv_mode,
                "n_folds": int(fold_level.shape[0]),
                "n_folds_with_positive_test": int((fold_level["n_positive_rows_test_fold"] > 0).sum()),
                "n_folds_with_single_positive": int((fold_level["n_positive_rows_test_fold"] == 1).sum()),
                "median_positive_rows_per_fold": float_or_nan(float(fold_level["n_positive_rows_test_fold"].median())),
                "max_positive_rows_per_fold": int(fold_level["n_positive_rows_test_fold"].max()),
                "median_test_rows_per_fold": float_or_nan(float(fold_level["n_test_rows"].median())),
                "min_test_rows_per_fold": int(fold_level["n_test_rows"].min()),
                "max_test_rows_per_fold": int(fold_level["n_test_rows"].max()),
                "n_positive_loci_total": int(pos_locus.shape[0]),
                "median_candidates_in_positive_locus": float_or_nan(
                    float(pos_locus["n_candidate_genes_in_test_locus"].median())
                )
                if not pos_locus.empty
                else float("nan"),
                "min_candidates_in_positive_locus": int(pos_locus["n_candidate_genes_in_test_locus"].min())
                if not pos_locus.empty
                else 0,
                "max_candidates_in_positive_locus": int(pos_locus["n_candidate_genes_in_test_locus"].max())
                if not pos_locus.empty
                else 0,
                "fraction_positive_loci_with_leq3_candidates": float_or_nan(
                    float((pos_locus["n_candidate_genes_in_test_locus"] <= 3).mean())
                )
                if not pos_locus.empty
                else float("nan"),
            }
        )

    out = pd.DataFrame(rows).sort_values("cv_mode", kind="stable")
    out.to_csv(out_dir / "pca_small_data_warning_analysis.csv", index=False)
    return out


def write_final_report(
    out_dir: Path,
    pca_summary_df: pd.DataFrame,
    overlap_by_cv: Dict[str, pd.DataFrame],
    suspicious_features: pd.DataFrame,
    perm_summary: pd.DataFrame,
    random_comp: pd.DataFrame,
    small_data_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# PCA Locus-to-Gene Sanity Audit")
    lines.append("")
    lines.append("This report audits whether high PCA metrics are trustworthy or likely inflated.")
    lines.append("")

    lines.append("## 1) Observed PCA Performance")
    lines.append("")
    if pca_summary_df.empty:
        lines.append("- No PCA summary rows found.")
    else:
        for row in pca_summary_df.itertuples(index=False):
            lines.append(
                f"- `{row.cv_mode}`: PR-AUC={row.mean_fold_pr_auc:.4f}, ROC-AUC={row.mean_fold_roc_auc:.4f}, "
                f"Recall@1={row.mean_recall_at_1:.4f}, Recall@3={row.mean_recall_at_3:.4f}, MRR={row.mean_mrr:.4f}."
            )
    lines.append("")

    lines.append("## 2) Train-Test Overlap (Leakage Check)")
    lines.append("")
    for cv_mode in CV_MODES:
        ov = overlap_by_cv.get(cv_mode)
        if ov is None or ov.empty:
            lines.append(f"- `{cv_mode}`: no overlap table generated.")
            continue
        lines.append(
            f"- `{cv_mode}`: max overlap_gene_id_count={int(ov['overlap_gene_id_count'].max())}, "
            f"max overlap_gene_symbol_count={int(ov['overlap_gene_symbol_count'].max())}, "
            f"max overlap_locus_id_count={int(ov['overlap_locus_id_count'].max())}."
        )
    lines.append("")

    lines.append("## 3) Single-Feature Separability")
    lines.append("")
    if suspicious_features.empty:
        lines.append("- No feature/component met the 'almost perfect separator' rule (AUC>=0.98 or non-overlapping ranges).")
    else:
        lines.append(
            f"- {len(suspicious_features)} feature(s) flagged as almost perfect separators. "
            "Inspect `pca_feature_separability_suspicious.csv`."
        )
        top = suspicious_features.head(10)
        for row in top.itertuples(index=False):
            auc = row.single_feature_auc
            auc_txt = "nan" if not np.isfinite(auc) else f"{auc:.4f}"
            lines.append(f"  - {row.feature}: single_feature_auc={auc_txt}, interval_overlap={row.interval_overlap:.6f}.")
    lines.append("")

    lines.append("## 4) Label Permutation Sanity Check")
    lines.append("")
    if perm_summary.empty:
        lines.append("- Permutation summary missing.")
    else:
        for row in perm_summary.itertuples(index=False):
            lines.append(
                f"- `{row.cv_mode}`: real PR-AUC={row.real_mean_fold_pr_auc:.4f} vs perm mean={row.perm_mean_mean_fold_pr_auc:.4f} "
                f"(q95={row.perm_q95_mean_fold_pr_auc:.4f}); real ROC-AUC={row.real_mean_fold_roc_auc:.4f} "
                f"vs perm mean={row.perm_mean_mean_fold_roc_auc:.4f} (q95={row.perm_q95_mean_fold_roc_auc:.4f})."
            )
    lines.append("")

    lines.append("## 5) Random Ranking Baseline")
    lines.append("")
    if random_comp.empty:
        lines.append("- Random baseline comparison missing.")
    else:
        for row in random_comp.itertuples(index=False):
            lines.append(
                f"- `{row.cv_mode}`: PCA PR-AUC={row.pca_mean_fold_pr_auc:.4f}, baseline PR-AUC={row.baseline_mean_fold_pr_auc:.4f}, "
                f"random mean PR-AUC={row.random_mean_mean_fold_pr_auc:.4f}. "
                f"PCA ROC-AUC={row.pca_mean_fold_roc_auc:.4f}, random mean ROC-AUC={row.random_mean_mean_fold_roc_auc:.4f}."
            )
    lines.append("")

    lines.append("## 6) Small-Data Instability Diagnostics")
    lines.append("")
    if small_data_df.empty:
        lines.append("- Small-data warning table missing.")
    else:
        for row in small_data_df.itertuples(index=False):
            lines.append(
                f"- `{row.cv_mode}`: folds={int(row.n_folds)}, folds_with_positive_test={int(row.n_folds_with_positive_test)}, "
                f"median positives/fold={row.median_positive_rows_per_fold:.2f}, "
                f"median candidates in positive loci={row.median_candidates_in_positive_locus:.2f}."
            )
    lines.append("")

    lines.append("## Final Verdict")
    lines.append("")
    verdict = []
    if not perm_summary.empty:
        gt_flags = []
        for cv_mode in CV_MODES:
            sub = perm_summary[perm_summary["cv_mode"] == cv_mode]
            if sub.empty:
                continue
            gt = int(sub.iloc[0]["real_gt_perm_q95_mean_fold_pr_auc"]) and int(
                sub.iloc[0]["real_gt_perm_q95_mean_fold_roc_auc"]
            )
            gt_flags.append(gt)
        if len(gt_flags) > 0 and all(gt_flags):
            verdict.append(
                "PCA performance is above shuffled-label null in all CV modes, which argues against a trivial implementation bug."
            )
        else:
            verdict.append(
                "At least one CV mode does not exceed permutation q95 clearly; this suggests possible instability or bug and warrants deeper debugging."
            )
    if any(not overlap_by_cv[m].empty and int(overlap_by_cv[m]["overlap_gene_id_count"].max()) > 0 for m in CV_MODES):
        verdict.append("There is train-test gene overlap in at least one mode (expected for standard LOLO), which can inflate embedding gains.")
    else:
        verdict.append("Gene overlap is zero in all audited modes, reducing direct gene-identity leakage risk.")
    if not small_data_df.empty and np.nanmedian(small_data_df["median_positive_rows_per_fold"].to_numpy(dtype=float)) <= 1.0:
        verdict.append(
            "Dataset is very small with few positives per fold; near-perfect metrics can still occur by small-sample instability."
        )
    if suspicious_features.empty:
        verdict.append("No single baseline/PCA feature was an obvious near-perfect separator by itself.")
    else:
        verdict.append("Some single features/components show near-perfect class separation and should be investigated as potential leakage proxies.")

    for item in verdict:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `pca_fold_audit_<cv_mode>.csv`")
    lines.append("- `pca_train_test_overlap_<cv_mode>.csv`")
    lines.append("- `pca_per_locus_inspection_<cv_mode>.csv` and `pca_per_locus_tables_<cv_mode>/`")
    lines.append("- `pca_feature_separability_summary.csv`, `pca_feature_separability_suspicious.csv`")
    lines.append("- `pca_label_permutation_metrics.csv`, `pca_label_permutation_summary.csv`")
    lines.append("- `pca_random_ranking_distribution.csv`, `pca_random_baseline_comparison.csv`")
    lines.append("- `pca_small_data_warning_analysis.csv`")

    (out_dir / "pca_sanity_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = ranker.load_table(args.input_table).copy()
    full_df[ranker.LABEL_COL] = pd.to_numeric(full_df[ranker.LABEL_COL], errors="coerce").fillna(0).astype(int)
    for col in [ranker.LOCUS_COL, ranker.GENE_ID_COL, ranker.GENE_SYMBOL_COL]:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype(str)

    pca_dim = detect_pca_dim(args.eval_dir)

    preds_by_cv: Dict[str, pd.DataFrame] = {}
    pca_summary_by_cv: Dict[str, Dict[str, float]] = {}
    baseline_summary_by_cv: Dict[str, Dict[str, float]] = {}
    fold_audit_by_cv: Dict[str, pd.DataFrame] = {}
    overlap_by_cv: Dict[str, pd.DataFrame] = {}

    for cv_mode in CV_MODES:
        pca_tables = load_mode_tables(args.eval_dir, cv_mode=cv_mode, mode="pca")
        none_tables = load_mode_tables(args.eval_dir, cv_mode=cv_mode, mode="none")

        pred_df = pca_tables["all_predictions"].copy()
        fold_df = pca_tables["fold_metrics"].copy()
        preds_by_cv[cv_mode] = pred_df

        pca_summary_by_cv[cv_mode] = {
            k: float(v)
            for k, v in pca_tables["summary"].iloc[0].to_dict().items()
            if isinstance(v, (int, float, np.floating, np.integer))
        }
        baseline_summary_by_cv[cv_mode] = {
            k: float(v)
            for k, v in none_tables["summary"].iloc[0].to_dict().items()
            if isinstance(v, (int, float, np.floating, np.integer))
        }

        fold_audit_by_cv[cv_mode] = build_fold_audit_csv(cv_mode, pred_df, fold_df, out_dir)
        overlap_by_cv[cv_mode] = leakage_overlap_for_cv(
            cv_mode=cv_mode,
            full_df=full_df,
            gene_group=ranker.make_gene_group_series(full_df),
            pred_df=pred_df,
            fold_df=fold_df,
            out_dir=out_dir,
        )
        export_per_locus_tables(cv_mode=cv_mode, pred_df=pred_df, out_dir=out_dir)

    pca_summary_rows = []
    for cv_mode in CV_MODES:
        m = pca_summary_by_cv[cv_mode]
        pca_summary_rows.append(
            {
                "cv_mode": cv_mode,
                "mean_fold_pr_auc": float_or_nan(float(m.get("mean_fold_pr_auc", float("nan")))),
                "mean_fold_roc_auc": float_or_nan(float(m.get("mean_fold_roc_auc", float("nan")))),
                "mean_recall_at_1": float_or_nan(float(m.get("mean_recall_at_1", float("nan")))),
                "mean_recall_at_3": float_or_nan(float(m.get("mean_recall_at_3", float("nan")))),
                "mean_mrr": float_or_nan(float(m.get("mean_mrr", float("nan")))),
            }
        )
    pca_summary_df = pd.DataFrame(pca_summary_rows).sort_values("cv_mode", kind="stable")
    pca_summary_df.to_csv(out_dir / "pca_observed_metrics_summary.csv", index=False)

    _, suspicious_features = feature_separability_analysis(full_df=full_df, pca_dim=pca_dim, out_dir=out_dir)
    _, perm_summary = permutation_sanity_check(
        full_df=full_df,
        out_dir=out_dir,
        pca_dim=pca_dim,
        args=args,
        real_summary_by_cv=pca_summary_by_cv,
    )
    _, random_comp = random_baseline_check(
        preds_by_cv=preds_by_cv,
        out_dir=out_dir,
        n_seeds=int(args.n_random_ranking_seeds),
        baseline_summary_by_cv=baseline_summary_by_cv,
        pca_summary_by_cv=pca_summary_by_cv,
    )
    small_data_df = small_data_warning_analysis(fold_audit_by_cv=fold_audit_by_cv, out_dir=out_dir)

    write_final_report(
        out_dir=out_dir,
        pca_summary_df=pca_summary_df,
        overlap_by_cv=overlap_by_cv,
        suspicious_features=suspicious_features,
        perm_summary=perm_summary,
        random_comp=random_comp,
        small_data_df=small_data_df,
    )

    print(f"[done] PCA sanity audit artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()

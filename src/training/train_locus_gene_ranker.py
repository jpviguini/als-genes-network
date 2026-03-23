#!/usr/bin/env python3
"""Proof-of-concept global locus-to-gene ranking with LOLO CV."""

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


DEFAULT_INPUT_TABLE = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/"
    "GCST90027164_cs_gene_candidate_feature_table.csv"
)
DEFAULT_OUT_DIR = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/locus_gene_ranker"
)

LOCUS_COL = "gwas_study_locus_id"
LABEL_COL = "label_positive"

# Small interpretable genetic baseline requested by user.
BASELINE_GENETIC_FEATURES = [
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
EMBEDDING_INDICATOR_FEATURE = "has_gene_embedding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a global locus-to-gene ranker with Leave-One-Locus-Out CV "
            "and export locus-level ranking outputs."
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
        choices=["none", "full", "pca", "all"],
        default="all",
        help="Embedding mode to run. Use 'all' for ablation (none/full/pca).",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=16,
        help="PCA components for embedding_mode='pca'.",
    )
    parser.add_argument(
        "--penalty",
        choices=["l1", "elasticnet"],
        default="l1",
        help="Logistic regression penalty.",
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
        help="l1_ratio when penalty='elasticnet'.",
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
    if penalty == "l1":
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
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

        # PCA mode
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


def rank_within_locus(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank_within_locus"] = (
        out.groupby(LOCUS_COL)["predicted_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return out


def resolve_modes(embedding_mode: str) -> List[str]:
    if embedding_mode == "all":
        return ["none", "full", "pca"]
    return [embedding_mode]


def baseline_columns_for_mode(mode: str) -> List[str]:
    cols = list(BASELINE_GENETIC_FEATURES)
    if mode in {"full", "pca"}:
        cols.append(EMBEDDING_INDICATOR_FEATURE)
    return cols


def validate_input(df: pd.DataFrame, modes: Sequence[str], embedding_cols: Sequence[str]) -> None:
    required_cols = [LOCUS_COL, LABEL_COL, "gene_symbol"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input table missing required columns: {missing}")

    missing_baseline = [c for c in BASELINE_GENETIC_FEATURES if c not in df.columns]
    if missing_baseline:
        raise ValueError(
            "Input table missing required baseline features: "
            f"{missing_baseline}"
        )

    if any(mode in {"full", "pca"} for mode in modes):
        if EMBEDDING_INDICATOR_FEATURE not in df.columns:
            raise ValueError(
                "Input table missing required embedding indicator feature: "
                f"'{EMBEDDING_INDICATOR_FEATURE}'"
            )
        if len(embedding_cols) == 0:
            raise ValueError(
                "Embedding mode requested but no 'gene_emb_*' columns were found."
            )

    y = pd.to_numeric(df[LABEL_COL], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Column '{LABEL_COL}' contains non-numeric values.")
    if not y.isin([0, 1]).all():
        raise ValueError(f"Column '{LABEL_COL}' must be binary 0/1.")


def run_lolo_for_mode(
    df: pd.DataFrame,
    mode: str,
    baseline_cols: List[str],
    embedding_cols: List[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    loci = sorted(df[LOCUS_COL].astype(str).unique().tolist())
    fold_rows: List[Dict[str, object]] = []
    prediction_rows: List[pd.DataFrame] = []
    positive_rank_rows: List[pd.DataFrame] = []

    for fold_idx, heldout_locus in enumerate(loci, start=1):
        test_mask = df[LOCUS_COL].astype(str) == heldout_locus
        train_df = df.loc[~test_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)
        y_train = train_df[LABEL_COL].astype(int).to_numpy()
        y_test = test_df[LABEL_COL].astype(int).to_numpy()

        if np.unique(y_train).size < 2:
            fold_rows.append(
                {
                    "mode": mode,
                    "fold_index": fold_idx,
                    "heldout_locus_id": heldout_locus,
                    "n_genes_in_locus": int(len(test_df)),
                    "n_positive_in_locus": int(y_test.sum()),
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

        pred_df = test_df.copy()
        pred_df["mode"] = mode
        pred_df["fold_index"] = fold_idx
        pred_df["heldout_locus_id"] = heldout_locus
        pred_df["predicted_score"] = y_score
        pred_df = rank_within_locus(pred_df)

        # Row-level top feature contributions in the held-out locus.
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
                        "mode",
                        "fold_index",
                        "heldout_locus_id",
                        LOCUS_COL,
                        "gene_id",
                        "gene_symbol",
                        LABEL_COL,
                        "predicted_score",
                        "rank_within_locus",
                    ]
                ].copy()
            )

        fold_rows.append(
            {
                "mode": mode,
                "fold_index": fold_idx,
                "heldout_locus_id": heldout_locus,
                "n_genes_in_locus": int(len(test_df)),
                "n_positive_in_locus": int(y_test.sum()),
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
        by=[LOCUS_COL, "predicted_score"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)
    all_predictions = rank_within_locus(all_predictions)

    fold_metrics_df = pd.DataFrame(fold_rows)
    positive_ranks_df = (
        pd.concat(positive_rank_rows, axis=0, ignore_index=True)
        if positive_rank_rows
        else pd.DataFrame(
            columns=[
                "mode",
                "fold_index",
                "heldout_locus_id",
                LOCUS_COL,
                "gene_id",
                "gene_symbol",
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
        all_predictions[LABEL_COL].astype(int).to_numpy(),
        all_predictions["predicted_score"].to_numpy(),
    )
    pooled_pr = safe_pr_auc(
        all_predictions[LABEL_COL].astype(int).to_numpy(),
        all_predictions["predicted_score"].to_numpy(),
    )

    summary = {
        "mode": mode,
        "n_rows": int(len(df)),
        "n_loci": int(df[LOCUS_COL].nunique()),
        "n_positive_rows": int(df[LABEL_COL].astype(int).sum()),
        "n_positive_genes": int(df.loc[df[LABEL_COL].astype(int) == 1, "gene_symbol"].nunique()),
        "embedding_feature_count": int(len(embedding_cols)),
        "folds_total": int(len(fold_metrics_df)),
        "folds_ok": int((fold_metrics_df["status"] == "ok").sum()) if not fold_metrics_df.empty else 0,
        "mean_fold_roc_auc": mean_ignore_nan(fold_metrics_df["roc_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "std_fold_roc_auc": std_ignore_nan(fold_metrics_df["roc_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_fold_pr_auc": mean_ignore_nan(fold_metrics_df["pr_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "std_fold_pr_auc": std_ignore_nan(fold_metrics_df["pr_auc"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_recall_at_1": mean_ignore_nan(fold_metrics_df["recall_at_1"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_recall_at_3": mean_ignore_nan(fold_metrics_df["recall_at_3"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "mean_mrr": mean_ignore_nan(fold_metrics_df["mrr"].tolist()) if not fold_metrics_df.empty else float("nan"),
        "pooled_roc_auc": float_or_nan(pooled_roc),
        "pooled_pr_auc": float_or_nan(pooled_pr),
    }

    # Fit final model on full data for coefficient export.
    final_feature_builder = ModeFeatureBuilder(
        baseline_cols=baseline_cols,
        embedding_cols=embedding_cols,
        mode=mode,
        pca_dim=int(args.pca_dim),
        random_state=int(args.random_state),
    )
    x_full = final_feature_builder.fit_transform(df)
    y_full = df[LABEL_COL].astype(int).to_numpy()
    final_model = build_model(
        penalty=args.penalty,
        c_value=float(args.regularization_strength),
        l1_ratio=float(args.l1_ratio),
        max_iter=int(args.max_iter),
        random_state=int(args.random_state),
    )
    final_model.fit(x_full, y_full)
    final_coef = final_model.named_steps["logreg"].coef_.ravel()
    final_feature_names = final_feature_builder.feature_names_ or []

    coef_df = pd.DataFrame(
        {
            "mode": mode,
            "feature": final_feature_names,
            "coefficient": final_coef,
            "abs_coefficient": np.abs(final_coef),
            "non_zero": (np.abs(final_coef) > 1e-12).astype(int),
        }
    ).sort_values("abs_coefficient", ascending=False, kind="stable")

    feature_group = []
    for name in coef_df["feature"]:
        if name in baseline_cols:
            feature_group.append("baseline")
        elif name.startswith("gene_emb_"):
            feature_group.append("embedding_raw")
        elif name.startswith("emb_pca_"):
            feature_group.append("embedding_pca")
        else:
            feature_group.append("other")
    coef_df["feature_group"] = feature_group

    if mode == "pca" and final_feature_builder.pca_explained_variance_ratio_ is not None:
        evr = final_feature_builder.pca_explained_variance_ratio_
        pca_df = pd.DataFrame(
            {
                "mode": mode,
                "component": [f"emb_pca_{i:03d}" for i in range(len(evr))],
                "explained_variance_ratio": evr,
                "cumulative_explained_variance_ratio": np.cumsum(evr),
            }
        )
    else:
        pca_df = pd.DataFrame(
            columns=["mode", "component", "explained_variance_ratio", "cumulative_explained_variance_ratio"]
        )

    return {
        "summary": summary,
        "fold_metrics": fold_metrics_df,
        "positive_ranks": positive_ranks_df,
        "all_predictions": all_predictions,
        "top3_predictions": top3_df,
        "false_positives": false_positive_df,
        "coefficients": coef_df,
        "pca_explained_variance": pca_df,
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
    pca_path = mode_dir / "pca_explained_variance.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(mode_result["summary"], f, indent=2, ensure_ascii=False)

    mode_result["fold_metrics"].to_csv(fold_path, index=False)
    mode_result["positive_ranks"].to_csv(pos_rank_path, index=False)
    mode_result["all_predictions"].to_csv(all_pred_path, index=False)
    mode_result["top3_predictions"].to_csv(top3_path, index=False)
    mode_result["false_positives"].to_csv(fp_path, index=False)
    mode_result["coefficients"].to_csv(coef_path, index=False)
    mode_result["pca_explained_variance"].to_csv(pca_path, index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.input_table).copy()
    modes = resolve_modes(args.embedding_mode)
    embedding_cols = sorted([c for c in df.columns if c.startswith("gene_emb_")])
    validate_input(df, modes=modes, embedding_cols=embedding_cols)

    # Preserve identifiers.
    for col in [LOCUS_COL, "gene_id", "gene_symbol", "gwas_study_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)

    all_mode_summaries: List[Dict[str, object]] = []

    for mode in modes:
        mode_result = run_lolo_for_mode(
            df=df,
            mode=mode,
            baseline_cols=baseline_columns_for_mode(mode),
            embedding_cols=embedding_cols,
            args=args,
        )
        write_outputs_for_mode(args.out_dir / f"mode_{mode}", mode_result)
        all_mode_summaries.append(mode_result["summary"])

    pd.DataFrame(all_mode_summaries).to_csv(args.out_dir / "ablation_summary.csv", index=False)
    with open(args.out_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_mode_summaries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

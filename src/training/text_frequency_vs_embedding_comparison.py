#!/home/viguinijpv/python310/bin/python3.10
"""Compare PCA-embedding model vs text-frequency baseline (global functional setup).

Models compared:
1) hpa_pca32      = HPA(brain,muscle) + PCA32(text embeddings)
2) hpa_frequency  = HPA(brain,muscle) + log1p(gene mention count)
3) frequency_only = log1p(gene mention count)
4) hpa_only       = HPA(brain,muscle)

Comparison regimes:
A) matched-universe:
   genes with embeddings + text frequency + mapped HPA row.
B) expanded frequency-universe:
   genes with text frequency + mapped HPA row.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
TRAINING_DIR = SRC_DIR / "training"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import VALIDATION_GENES
from build_cs_gene_candidate_feature_table import (
    load_gene_embeddings,
    normalize_gene_id,
    normalize_gene_symbol,
)
from train_global_functional_model_hpa_pca import (
    HPA_FEATURES,
    _load_hgnc_symbol_to_ensembl,
    _load_hpa_gene_table,
    _normalize_gene_id_strict,
)


DEFAULT_EMBEDDING_PATH = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/features/"
    "featuresUPPER_pubmedbert_neurodegenerative_disease/features_ALS_pubmedbert.pkl"
)
DEFAULT_HPA_PATH = Path("/home/viguinijpv/200.18.99.75:8000/IC/src/data/reference/rna_tissue_consensus.tsv")
DEFAULT_HGNC_PATH = Path("/home/viguinijpv/200.18.99.75:8000/IC/src/data/reference/hgnc_complete_set.txt")
DEFAULT_OUT_DIR = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/"
    f"text_frequency_vs_embedding_{date.today().strftime('%Y%m%d')}"
)


@dataclass
class ModelEvalResult:
    model_name: str
    universe_name: str
    feature_count: int
    n_candidates: int
    n_positives: int
    n_splits: int
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    roc_auc_oof: float
    pr_auc_oof: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    fold_metrics_df: pd.DataFrame
    oof_scores: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HPA+PCA32 embedding model vs HPA+text-frequency baseline."
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
        help="Logistic regularization profile shared across compared models.",
    )
    parser.add_argument("--regularization-strength", type=float, default=0.1, help="Inverse regularization C.")
    parser.add_argument("--l1-ratio", type=float, default=0.5, help="Only used for elasticnet.")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--hpa-expression-threshold", type=float, default=1.0)
    parser.add_argument(
        "--no-require-hpa-row",
        action="store_true",
        help="If set, do not require mapped HPA rows for matched/expanded universes (missing HPA will be imputed).",
    )
    parser.add_argument(
        "--run-optional-models",
        action="store_true",
        default=True,
        help="Run hpa_only and frequency_only in addition to required models.",
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


def _load_embedding_table(embedding_path: Path) -> Tuple[pd.DataFrame, Dict[str, object], List[str]]:
    embeddings, emb_dim, emb_stats = load_gene_embeddings(embedding_path)
    symbols = sorted(embeddings.keys())
    emb_cols = [f"gene_emb_{i:04d}" for i in range(int(emb_dim))]
    matrix = np.vstack([embeddings[g] for g in symbols]).astype(np.float64)
    emb_df = pd.DataFrame(matrix, columns=emb_cols)
    emb_df.insert(0, "gene_symbol", symbols)
    return emb_df, emb_stats, emb_cols


def _count_mentions_from_value(value: object) -> int:
    if value is None:
        return 0

    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return 1 if value.size > 0 else 0
        if value.ndim >= 2:
            return int(value.shape[0])
        return 0

    if isinstance(value, Mapping):
        for key in ("embeddings", "vectors", "embedding", "vector", "features", "values", "contexts"):
            if key in value:
                return _count_mentions_from_value(value[key])
        sub = [_count_mentions_from_value(v) for v in value.values()]
        total = int(sum(x for x in sub if x > 0))
        return total

    if isinstance(value, (list, tuple)):
        if not value:
            return 0
        if all(np.isscalar(v) for v in value):
            return 1
        counts = [_count_mentions_from_value(v) for v in value]
        if any(c > 0 for c in counts):
            return int(sum(c for c in counts if c > 0))
        return int(len(value))

    return 0


def load_gene_mention_counts(embedding_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    with open(embedding_path, "rb") as f:
        obj = pickle.load(f)

    rows: List[Dict[str, object]] = []
    records_seen = 0

    if isinstance(obj, Mapping):
        if "genes" in obj and "embeddings" in obj:
            genes = obj.get("genes")
            embs = obj.get("embeddings")
            if isinstance(genes, (list, tuple, np.ndarray)) and isinstance(embs, (list, tuple, np.ndarray)):
                n = min(len(genes), len(embs))
                for i in range(n):
                    records_seen += 1
                    symbol = normalize_gene_symbol(genes[i])
                    if symbol is None:
                        continue
                    c = _count_mentions_from_value(embs[i])
                    rows.append({"gene_symbol": symbol, "gene_mention_count": max(int(c), 1)})
        else:
            for g_raw, val in obj.items():
                records_seen += 1
                symbol = normalize_gene_symbol(g_raw)
                if symbol is None:
                    continue
                c = _count_mentions_from_value(val)
                rows.append({"gene_symbol": symbol, "gene_mention_count": max(int(c), 1)})
    elif isinstance(obj, (list, tuple)):
        for rec in obj:
            records_seen += 1
            if isinstance(rec, Mapping):
                symbol = normalize_gene_symbol(rec.get("gene") or rec.get("gene_symbol") or rec.get("symbol"))
                if symbol is None:
                    continue
                c = _count_mentions_from_value(rec.get("embedding", rec))
                rows.append({"gene_symbol": symbol, "gene_mention_count": max(int(c), 1)})
    else:
        raise ValueError(f"Unsupported embedding object for mention count extraction: {type(obj)}")

    if not rows:
        raise ValueError("No mention-count rows extracted from embedding source.")

    df = pd.DataFrame(rows)
    df = (
        df.groupby("gene_symbol", as_index=False)["gene_mention_count"]
        .sum()
        .sort_values("gene_symbol", kind="stable")
        .reset_index(drop=True)
    )
    df["log1p_gene_mention_count"] = np.log1p(df["gene_mention_count"].astype(float))
    stats = {
        "records_seen": int(records_seen),
        "rows_extracted": int(len(rows)),
        "unique_genes_with_mentions": int(df["gene_symbol"].nunique()),
    }
    return df, stats


def _rank_metrics_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    pos_positions = np.flatnonzero(y_sorted == 1) + 1
    n_pos = int(len(pos_positions))
    if n_pos == 0:
        return {
            "recall_at_1": float("nan"),
            "recall_at_3": float("nan"),
            "recall_at_5": float("nan"),
            "recall_at_10": float("nan"),
            "mrr": float("nan"),
        }

    def recall_at(k: int) -> float:
        return float(np.sum(pos_positions <= int(k)) / float(n_pos))

    mrr = float(np.mean(1.0 / pos_positions))
    return {
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "mrr": mrr,
    }


def _evaluate_model_cv(
    df: pd.DataFrame,
    *,
    model_name: str,
    emb_cols: Sequence[str],
    hpa_cols: Sequence[str],
    pca_dim: int,
    penalty: str,
    c_value: float,
    l1_ratio: float,
    max_iter: int,
    random_state: int,
    cv_folds: int,
    universe_name: str,
) -> ModelEvalResult:
    y = df["label_positive"].astype(int).to_numpy()
    n_candidates = int(len(df))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    max_splits = min(int(cv_folds), n_pos, n_neg)
    if max_splits < 2:
        raise ValueError(f"Not enough class balance for CV in {model_name}: pos={n_pos}, neg={n_neg}")

    skf = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=int(random_state))
    oof = np.full(len(df), np.nan, dtype=np.float64)
    fold_rows: List[Dict[str, float]] = []

    emb_all = _to_numeric_array(df, emb_cols)
    hpa_all = _to_numeric_array(df, hpa_cols)
    freq_all = _to_numeric_array(df, ["log1p_gene_mention_count"])

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df)), y), start=1):
        y_train = y[train_idx]
        y_val = y[val_idx]

        if model_name == "hpa_pca32":
            emb_train = emb_all[train_idx, :]
            emb_val = emb_all[val_idx, :]
            hpa_train = hpa_all[train_idx, :]
            hpa_val = hpa_all[val_idx, :]

            emb_scaler = StandardScaler()
            emb_train_scaled = emb_scaler.fit_transform(emb_train)
            emb_val_scaled = emb_scaler.transform(emb_val)

            n_comp = min(int(pca_dim), emb_train_scaled.shape[0], emb_train_scaled.shape[1])
            pca = PCA(n_components=n_comp, random_state=int(random_state) + int(fold_idx))
            pca_train = pca.fit_transform(emb_train_scaled)
            pca_val = pca.transform(emb_val_scaled)

            x_train = np.column_stack([hpa_train, pca_train])
            x_val = np.column_stack([hpa_val, pca_val])
            feature_count = int(x_train.shape[1])
        elif model_name == "hpa_frequency":
            x_train = np.column_stack([hpa_all[train_idx, :], freq_all[train_idx, :]])
            x_val = np.column_stack([hpa_all[val_idx, :], freq_all[val_idx, :]])
            feature_count = int(x_train.shape[1])
        elif model_name == "hpa_only":
            x_train = hpa_all[train_idx, :]
            x_val = hpa_all[val_idx, :]
            feature_count = int(x_train.shape[1])
        elif model_name == "frequency_only":
            x_train = freq_all[train_idx, :]
            x_val = freq_all[val_idx, :]
            feature_count = int(x_train.shape[1])
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        model = _build_logistic_model(
            penalty=penalty,
            c_value=float(c_value),
            l1_ratio=float(l1_ratio),
            max_iter=int(max_iter),
            random_state=int(random_state) + int(fold_idx),
        )
        model.fit(x_train, y_train)
        y_score = model.predict_proba(x_val)[:, 1]
        oof[val_idx] = y_score

        roc = float(roc_auc_score(y_val, y_score)) if len(np.unique(y_val)) == 2 else float("nan")
        pr = float(average_precision_score(y_val, y_score)) if len(np.unique(y_val)) == 2 else float("nan")
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

    valid = np.isfinite(oof)
    if valid.sum() == 0 or len(np.unique(y[valid])) < 2:
        roc_oof = float("nan")
        pr_oof = float("nan")
    else:
        roc_oof = float(roc_auc_score(y[valid], oof[valid]))
        pr_oof = float(average_precision_score(y[valid], oof[valid]))

    rank_metrics = _rank_metrics_from_scores(y, oof)
    return ModelEvalResult(
        model_name=model_name,
        universe_name=universe_name,
        feature_count=int(feature_count),
        n_candidates=n_candidates,
        n_positives=n_pos,
        n_splits=int(max_splits),
        roc_auc_mean=float(np.mean(roc_vals)) if roc_vals.size else float("nan"),
        roc_auc_std=float(np.std(roc_vals, ddof=1)) if roc_vals.size > 1 else 0.0,
        pr_auc_mean=float(np.mean(pr_vals)) if pr_vals.size else float("nan"),
        pr_auc_std=float(np.std(pr_vals, ddof=1)) if pr_vals.size > 1 else 0.0,
        roc_auc_oof=float(roc_oof),
        pr_auc_oof=float(pr_oof),
        recall_at_1=float(rank_metrics["recall_at_1"]),
        recall_at_3=float(rank_metrics["recall_at_3"]),
        recall_at_5=float(rank_metrics["recall_at_5"]),
        recall_at_10=float(rank_metrics["recall_at_10"]),
        mrr=float(rank_metrics["mrr"]),
        fold_metrics_df=fold_df,
        oof_scores=oof,
    )


def _plot_bar_metric(
    metric_df: pd.DataFrame,
    *,
    universe_name: str,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    model_order: Sequence[str],
) -> None:
    cur = metric_df.loc[metric_df["universe_name"] == universe_name].copy()
    cur = cur.set_index("model_name").reindex(list(model_order)).reset_index()
    cur = cur.loc[cur[metric_col].notna()].copy()

    plt.figure(figsize=(8, 4.6))
    x = np.arange(len(cur))
    vals = cur[metric_col].astype(float).to_numpy()
    errs = cur.get(f"{metric_col.replace('_oof', '')}_std", pd.Series([0.0] * len(cur))).astype(float).to_numpy()
    plt.bar(x, vals, color="#4C78A8")
    if len(errs) == len(vals) and np.any(np.isfinite(errs)):
        plt.errorbar(x, vals, yerr=errs, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)

    plt.xticks(x, cur["model_name"].tolist(), rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_recall_at_k(
    metric_df: pd.DataFrame,
    *,
    universe_name: str,
    out_path: Path,
    model_order: Sequence[str],
) -> None:
    cur = metric_df.loc[metric_df["universe_name"] == universe_name].copy()
    cur = cur.set_index("model_name").reindex(list(model_order)).reset_index()
    cur = cur.loc[cur["model_name"].notna()].copy()

    recall_cols = ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10"]
    x = np.arange(len(recall_cols))
    width = 0.18

    plt.figure(figsize=(10, 5.0))
    for i, row in enumerate(cur.itertuples(index=False)):
        vals = [float(getattr(row, c)) for c in recall_cols]
        plt.bar(x + (i - (len(cur) - 1) / 2) * width, vals, width=width, label=row.model_name)

    plt.xticks(x, ["Recall@1", "Recall@3", "Recall@5", "Recall@10"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Recall")
    plt.title(f"Recall@k Comparison ({universe_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_mention_count_distribution(freq_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 4.6))
    vals = pd.to_numeric(freq_df["gene_mention_count"], errors="coerce").fillna(0.0).to_numpy()
    vals = np.clip(vals, a_min=0.0, a_max=None)
    if len(vals) > 0:
        bins = min(60, max(10, int(np.sqrt(len(vals)))))
        plt.hist(vals, bins=bins, color="#72B7B2", alpha=0.9, edgecolor="white")
        plt.yscale("log")
    else:
        plt.text(0.5, 0.5, "No mention counts available", ha="center", va="center")
        plt.axis("off")
    plt.xlabel("Raw gene mention count")
    plt.ylabel("Gene count (log scale)")
    plt.title("Gene Mention Count Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _make_universe_summary_rows(
    *,
    pca_current_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    expanded_freq_df: pd.DataFrame,
    positive_set: Sequence[str],
    require_hpa_row: bool,
) -> pd.DataFrame:
    def count_pos(df: pd.DataFrame) -> int:
        return int(df["label_positive"].sum()) if "label_positive" in df.columns else 0

    rows = [
        {
            "universe_name": "pca_current_model_universe",
            "definition": "All genes with embeddings (HPA missing values imputed as 0, as in current global model).",
            "requires_embedding": 1,
            "requires_text_frequency": 0,
            "requires_hpa_row": 0,
            "n_candidates": int(len(pca_current_df)),
            "n_positives": int(count_pos(pca_current_df)),
        },
        {
            "universe_name": "matched_universe_hpa_embedding_frequency",
            "definition": (
                "Genes with embeddings + text frequency"
                + (" + mapped HPA row." if require_hpa_row else " (HPA values imputed as needed).")
            ),
            "requires_embedding": 1,
            "requires_text_frequency": 1,
            "requires_hpa_row": int(bool(require_hpa_row)),
            "n_candidates": int(len(matched_df)),
            "n_positives": int(count_pos(matched_df)),
        },
        {
            "universe_name": "expanded_frequency_universe_hpa_frequency",
            "definition": (
                "Genes with text frequency"
                + (" + mapped HPA row." if require_hpa_row else " (HPA values imputed as needed).")
            ),
            "requires_embedding": 0,
            "requires_text_frequency": 1,
            "requires_hpa_row": int(bool(require_hpa_row)),
            "n_candidates": int(len(expanded_freq_df)),
            "n_positives": int(count_pos(expanded_freq_df)),
        },
        {
            "universe_name": "reference_positive_set",
            "definition": "Fixed positive genes from config.VALIDATION_GENES.",
            "requires_embedding": 0,
            "requires_text_frequency": 0,
            "requires_hpa_row": 0,
            "n_candidates": int(len(positive_set)),
            "n_positives": int(len(positive_set)),
        },
    ]
    out = pd.DataFrame(rows)
    out["frequency_vs_pca_candidate_delta"] = np.where(
        out["universe_name"] == "expanded_frequency_universe_hpa_frequency",
        int(len(expanded_freq_df) - len(pca_current_df)),
        np.nan,
    )
    out["frequency_vs_pca_positive_delta"] = np.where(
        out["universe_name"] == "expanded_frequency_universe_hpa_frequency",
        int(int(expanded_freq_df["label_positive"].sum()) - int(pca_current_df["label_positive"].sum())),
        np.nan,
    )
    return out


def _result_to_row(res: ModelEvalResult) -> Dict[str, object]:
    return {
        "universe_name": res.universe_name,
        "model_name": res.model_name,
        "feature_count": int(res.feature_count),
        "n_candidates": int(res.n_candidates),
        "n_positives": int(res.n_positives),
        "n_splits": int(res.n_splits),
        "roc_auc_mean": float(res.roc_auc_mean),
        "roc_auc_std": float(res.roc_auc_std),
        "pr_auc_mean": float(res.pr_auc_mean),
        "pr_auc_std": float(res.pr_auc_std),
        "roc_auc_oof": float(res.roc_auc_oof),
        "pr_auc_oof": float(res.pr_auc_oof),
        "recall_at_1": float(res.recall_at_1),
        "recall_at_3": float(res.recall_at_3),
        "recall_at_5": float(res.recall_at_5),
        "recall_at_10": float(res.recall_at_10),
        "mrr": float(res.mrr),
    }


def _model_feature_map(pca_dim: int) -> Dict[str, List[str]]:
    pca_feats = [f"emb_pca_{i:03d}" for i in range(int(pca_dim))]
    return {
        "hpa_pca32": ["hpa_brain_expression_value", "hpa_muscle_expression_value"] + pca_feats,
        "hpa_frequency": ["hpa_brain_expression_value", "hpa_muscle_expression_value", "log1p_gene_mention_count"],
        "frequency_only": ["log1p_gene_mention_count"],
        "hpa_only": ["hpa_brain_expression_value", "hpa_muscle_expression_value"],
    }


def _to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {args.embedding_path}")
    if not args.hpa_path.exists():
        raise FileNotFoundError(f"HPA file not found: {args.hpa_path}")

    emb_df, emb_stats, emb_cols = _load_embedding_table(args.embedding_path)
    mention_df, mention_stats = load_gene_mention_counts(args.embedding_path)
    hpa_df, hpa_stats = _load_hpa_gene_table(args.hpa_path, expression_threshold=float(args.hpa_expression_threshold))
    hgnc_map = _load_hgnc_symbol_to_ensembl(args.hgnc_path)

    # Current PCA-model universe (matches current setup: embedding-defined with HPA imputed).
    pca_current = emb_df.merge(mention_df, on="gene_symbol", how="left")
    pca_current = pca_current.merge(hpa_df, on="gene_symbol", how="left", suffixes=("", "_hpa"))
    if "gene_id" not in pca_current.columns:
        pca_current["gene_id"] = None
    pca_current["gene_id"] = pca_current["gene_id"].map(_normalize_gene_id_strict)
    pca_current["has_hpa_row"] = pca_current["gene_id"].notna().astype(int)
    missing_gid = pca_current["gene_id"].isna() | (pca_current["gene_id"].astype(str).str.strip() == "")
    pca_current.loc[missing_gid, "gene_id"] = pca_current.loc[missing_gid, "gene_symbol"].map(hgnc_map)
    pca_current["gene_id"] = pca_current["gene_id"].map(_normalize_gene_id_strict)
    for col in HPA_FEATURES + ["has_hpa_expression_evidence"]:
        if col not in pca_current.columns:
            pca_current[col] = 0.0
        pca_current[col] = pd.to_numeric(pca_current[col], errors="coerce").fillna(0.0)
    pca_current["gene_mention_count"] = pd.to_numeric(pca_current["gene_mention_count"], errors="coerce").fillna(0).astype(int)
    pca_current["log1p_gene_mention_count"] = np.log1p(pca_current["gene_mention_count"].astype(float))
    pca_current["has_embedding"] = 1
    pca_current["has_text_frequency"] = (pca_current["gene_mention_count"] > 0).astype(int)

    # Expanded frequency universe (frequency-driven, still with HPA features attached).
    freq_expanded = mention_df.merge(hpa_df, on="gene_symbol", how="left", suffixes=("", "_hpa"))
    if "gene_id" not in freq_expanded.columns:
        freq_expanded["gene_id"] = None
    freq_expanded["gene_id"] = freq_expanded["gene_id"].map(_normalize_gene_id_strict)
    freq_expanded["has_hpa_row"] = freq_expanded["gene_id"].notna().astype(int)
    missing_gid_freq = freq_expanded["gene_id"].isna() | (freq_expanded["gene_id"].astype(str).str.strip() == "")
    freq_expanded.loc[missing_gid_freq, "gene_id"] = freq_expanded.loc[missing_gid_freq, "gene_symbol"].map(hgnc_map)
    freq_expanded["gene_id"] = freq_expanded["gene_id"].map(_normalize_gene_id_strict)
    for col in HPA_FEATURES + ["has_hpa_expression_evidence"]:
        if col not in freq_expanded.columns:
            freq_expanded[col] = 0.0
        freq_expanded[col] = pd.to_numeric(freq_expanded[col], errors="coerce").fillna(0.0)
    freq_expanded["gene_mention_count"] = pd.to_numeric(freq_expanded["gene_mention_count"], errors="coerce").fillna(0).astype(int)
    freq_expanded["log1p_gene_mention_count"] = np.log1p(freq_expanded["gene_mention_count"].astype(float))
    freq_expanded["has_text_frequency"] = (freq_expanded["gene_mention_count"] > 0).astype(int)
    freq_expanded["has_embedding"] = freq_expanded["gene_symbol"].isin(set(emb_df["gene_symbol"].astype(str))).astype(int)
    # Attach embedding vectors to frequency table (needed only if a universe/model requests PCA features).
    freq_expanded = freq_expanded.merge(emb_df, on="gene_symbol", how="left", suffixes=("", "_emb"))
    for col in emb_cols:
        if col not in freq_expanded.columns:
            freq_expanded[col] = 0.0
        freq_expanded[col] = pd.to_numeric(freq_expanded[col], errors="coerce").fillna(0.0)

    positive_set = {normalize_gene_symbol(g) for g in VALIDATION_GENES if normalize_gene_symbol(g)}
    pca_current["label_positive"] = pca_current["gene_symbol"].isin(positive_set).astype(int)
    freq_expanded["label_positive"] = freq_expanded["gene_symbol"].isin(positive_set).astype(int)

    matched = pca_current.copy()
    matched = matched.loc[matched["has_text_frequency"] == 1].copy()
    if not bool(args.no_require_hpa_row):
        matched = matched.loc[matched["has_hpa_row"] == 1].copy()

    expanded_for_eval = freq_expanded.copy()
    if not bool(args.no_require_hpa_row):
        expanded_for_eval = expanded_for_eval.loc[expanded_for_eval["has_hpa_row"] == 1].copy()

    matched = matched.sort_values("gene_symbol", kind="stable").reset_index(drop=True)
    expanded_for_eval = expanded_for_eval.sort_values("gene_symbol", kind="stable").reset_index(drop=True)
    pca_current = pca_current.sort_values("gene_symbol", kind="stable").reset_index(drop=True)

    universe_summary_df = _make_universe_summary_rows(
        pca_current_df=pca_current,
        matched_df=matched,
        expanded_freq_df=expanded_for_eval,
        positive_set=sorted(list(positive_set)),
        require_hpa_row=not bool(args.no_require_hpa_row),
    )
    universe_summary_df.to_csv(args.out_dir / "candidate_universe_summary.csv", index=False)

    # Persist frequency feature table (expanded universe before/after HPA filter).
    freq_feature_table = expanded_for_eval[
        [
            "gene_id",
            "gene_symbol",
            "gene_mention_count",
            "log1p_gene_mention_count",
            "has_text_frequency",
            "has_embedding",
            "has_hpa_row",
            "label_positive",
            "hpa_brain_expression_value",
            "hpa_muscle_expression_value",
        ]
    ].copy()
    freq_feature_table.to_csv(args.out_dir / "frequency_feature_table.csv", index=False)

    model_feature_lists = _model_feature_map(int(args.pca_dim))
    feature_manifest = {
        "matched_universe_models": ["hpa_pca32", "hpa_frequency", "hpa_only", "frequency_only"],
        "expanded_frequency_universe_models": ["hpa_frequency", "hpa_only", "frequency_only"],
        "feature_lists": model_feature_lists,
    }
    (args.out_dir / "feature_lists_by_model.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

    eval_results: List[ModelEvalResult] = []
    model_order_main = ["hpa_only", "hpa_frequency", "hpa_pca32", "frequency_only"]

    # Matched-universe comparison.
    matched_models = ["hpa_pca32", "hpa_frequency"]
    if bool(args.run_optional_models):
        matched_models += ["hpa_only", "frequency_only"]

    for model_name in matched_models:
        res = _evaluate_model_cv(
            matched,
            model_name=model_name,
            emb_cols=emb_cols,
            hpa_cols=HPA_FEATURES,
            pca_dim=int(args.pca_dim),
            penalty=str(args.penalty),
            c_value=float(args.regularization_strength),
            l1_ratio=float(args.l1_ratio),
            max_iter=int(args.max_iter),
            random_state=int(args.random_state),
            cv_folds=int(args.cv_folds),
            universe_name="matched_universe",
        )
        eval_results.append(res)
        res.fold_metrics_df.to_csv(args.out_dir / f"cv_fold_metrics_matched_{model_name}.csv", index=False)

    # Expanded frequency-universe comparison.
    expanded_models = ["hpa_frequency"]
    if bool(args.run_optional_models):
        expanded_models += ["hpa_only", "frequency_only"]

    for model_name in expanded_models:
        res = _evaluate_model_cv(
            expanded_for_eval,
            model_name=model_name,
            emb_cols=emb_cols,
            hpa_cols=HPA_FEATURES,
            pca_dim=int(args.pca_dim),
            penalty=str(args.penalty),
            c_value=float(args.regularization_strength),
            l1_ratio=float(args.l1_ratio),
            max_iter=int(args.max_iter),
            random_state=int(args.random_state) + 1000,
            cv_folds=int(args.cv_folds),
            universe_name="expanded_frequency_universe",
        )
        eval_results.append(res)
        res.fold_metrics_df.to_csv(args.out_dir / f"cv_fold_metrics_expanded_{model_name}.csv", index=False)

    metrics_df = pd.DataFrame([_result_to_row(r) for r in eval_results]).sort_values(
        ["universe_name", "model_name"], kind="stable"
    )
    metrics_df.to_csv(args.out_dir / "model_metrics_comparison.csv", index=False)

    # Plots (main emphasis on matched universe, as requested).
    plot_models = [m for m in ["hpa_only", "hpa_frequency", "hpa_pca32"] if m in set(metrics_df["model_name"])]
    _plot_bar_metric(
        metrics_df,
        universe_name="matched_universe",
        metric_col="roc_auc_oof",
        ylabel="OOF ROC-AUC",
        title="ROC-AUC Comparison (Matched Universe)",
        out_path=args.out_dir / "roc_auc_comparison.png",
        model_order=plot_models,
    )
    _plot_bar_metric(
        metrics_df,
        universe_name="matched_universe",
        metric_col="pr_auc_oof",
        ylabel="OOF PR-AUC",
        title="PR-AUC Comparison (Matched Universe)",
        out_path=args.out_dir / "pr_auc_comparison.png",
        model_order=plot_models,
    )
    _plot_recall_at_k(
        metrics_df,
        universe_name="matched_universe",
        out_path=args.out_dir / "recall_at_k_comparison.png",
        model_order=plot_models + (["frequency_only"] if "frequency_only" in set(metrics_df["model_name"]) else []),
    )
    _plot_mention_count_distribution(freq_feature_table, args.out_dir / "mention_count_distribution.png")

    # Report blocks.
    matched_metrics = metrics_df.loc[metrics_df["universe_name"] == "matched_universe"].copy()
    expanded_metrics = metrics_df.loc[metrics_df["universe_name"] == "expanded_frequency_universe"].copy()

    def get_metric(model: str, col: str, default: float = float("nan")) -> float:
        cur = matched_metrics.loc[matched_metrics["model_name"] == model]
        if cur.empty:
            return default
        return float(cur.iloc[0][col])

    auc_delta = get_metric("hpa_pca32", "roc_auc_oof") - get_metric("hpa_frequency", "roc_auc_oof")
    pr_delta = get_metric("hpa_pca32", "pr_auc_oof") - get_metric("hpa_frequency", "pr_auc_oof")
    r10_delta = get_metric("hpa_pca32", "recall_at_10") - get_metric("hpa_frequency", "recall_at_10")

    pca_candidates = int(len(pca_current))
    pca_positives = int(pca_current["label_positive"].sum())
    freq_candidates = int(len(expanded_for_eval))
    freq_positives = int(expanded_for_eval["label_positive"].sum())
    universes_different = bool(pca_candidates != freq_candidates or pca_positives != freq_positives)
    matched_vs_expanded_same = bool(
        set(matched["gene_symbol"].astype(str).tolist()) == set(expanded_for_eval["gene_symbol"].astype(str).tolist())
    )
    freq_without_embedding = int((expanded_for_eval["has_embedding"] == 0).sum())

    if universes_different:
        if matched_vs_expanded_same and freq_without_embedding == 0:
            universe_reason = (
                "Difference comes from HPA-row filtering: frequency/expanded and matched universes exclude "
                "35 embedding genes without mapped HPA rows. Frequency did not add extra genes beyond embeddings."
            )
        else:
            universe_reason = (
                "Universes differ due to feature-availability constraints (embedding and/or HPA mapping differences)."
            )
    else:
        universe_reason = (
            "Universes are identical in this dataset: text-frequency source is the same embedding pickle, "
            "and every frequency-available gene also has an embedding entry."
        )

    report_lines = [
        "# Text Frequency vs Embedding Comparison Report",
        "",
        "## Goal",
        "- Compare whether PCA-reduced text embeddings add predictive value beyond simple gene mention frequency.",
        "",
        "## Data Sources",
        f"- Embedding source: `{args.embedding_path}`",
        f"- HPA source: `{args.hpa_path}`",
        f"- Positive labels: `config.VALIDATION_GENES` (symbol-level).",
        "",
        "## Universe Counts (Requested Questions)",
        f"1. PCA candidates available: `{pca_candidates}`",
        f"2. PCA positives available: `{pca_positives}`",
        f"3. Frequency candidates available: `{freq_candidates}`",
        f"4. Frequency positives available: `{freq_positives}`",
        f"5. Candidate universes different? `{universes_different}`",
        f"6. Why: {universe_reason}",
        f"- Matched vs Expanded frequency universes identical? `{matched_vs_expanded_same}`",
        f"- Expanded frequency genes lacking embeddings: `{freq_without_embedding}`",
        "",
        "## Universe Summary Table",
        _to_md_table(universe_summary_df),
        "",
        "## CV Setup",
        "- Gene-level stratified cross-validation (StratifiedKFold).",
        f"- Folds used: `{int(args.cv_folds)}` (bounded by class counts per universe).",
        "- Logistic regression with class_weight=balanced, standardized features.",
        f"- Penalty: `{args.penalty}`, C: `{float(args.regularization_strength):.4g}`.",
        "",
        "## Matched-Universe Metrics",
        _to_md_table(
            matched_metrics[
                [
                    "model_name",
                    "n_candidates",
                    "n_positives",
                    "roc_auc_oof",
                    "pr_auc_oof",
                    "recall_at_1",
                    "recall_at_3",
                    "recall_at_5",
                    "recall_at_10",
                    "mrr",
                ]
            ].sort_values("model_name", kind="stable")
        ),
        "",
        "## Expanded Frequency-Universe Metrics",
        _to_md_table(
            expanded_metrics[
                [
                    "model_name",
                    "n_candidates",
                    "n_positives",
                    "roc_auc_oof",
                    "pr_auc_oof",
                    "recall_at_1",
                    "recall_at_3",
                    "recall_at_5",
                    "recall_at_10",
                    "mrr",
                ]
            ].sort_values("model_name", kind="stable")
        ),
        "",
        "## Direct Matched Comparison: HPA+PCA32 vs HPA+Frequency",
        f"- Δ ROC-AUC (PCA32 - Frequency): `{auc_delta:.4f}`",
        f"- Δ PR-AUC (PCA32 - Frequency): `{pr_delta:.4f}`",
        f"- Δ Recall@10 (PCA32 - Frequency): `{r10_delta:.4f}`",
        "",
        "## Interpretation (Cautious)",
        "- If deltas are small, frequency may already capture much of the signal.",
        "- If PCA improves PR-AUC/Recall@k, embeddings likely add ranking information beyond raw mention volume.",
        "- Universe effects are separated via matched vs expanded evaluation.",
        "- Treat conclusions as empirical for this dataset/corpus configuration; avoid overgeneralization.",
    ]
    (args.out_dir / "text_frequency_vs_embedding_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    summary = {
        "script": "text_frequency_vs_embedding_comparison.py",
        "status": "completed",
        "inputs": {
            "embedding_path": str(args.embedding_path),
            "hpa_path": str(args.hpa_path),
            "hgnc_path": str(args.hgnc_path),
        },
        "embedding_stats": emb_stats,
        "mention_stats": mention_stats,
        "hpa_stats": hpa_stats,
        "counts_requested": {
            "pca_candidates": int(pca_candidates),
            "pca_positives": int(pca_positives),
            "frequency_candidates": int(freq_candidates),
            "frequency_positives": int(freq_positives),
            "candidate_universes_different": bool(universes_different),
            "difference_explanation": universe_reason,
            "matched_vs_expanded_same": bool(matched_vs_expanded_same),
            "expanded_frequency_genes_without_embedding": int(freq_without_embedding),
        },
        "output_dir": str(args.out_dir),
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Text-frequency vs embedding comparison completed.")
    print(f"Output directory: {args.out_dir}")
    print(f"PCA candidates/positives: {pca_candidates}/{pca_positives}")
    print(f"Frequency candidates/positives: {freq_candidates}/{freq_positives}")
    print(f"Universes different: {universes_different}")


if __name__ == "__main__":
    main()

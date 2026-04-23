import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import VALIDATION_GENES


umbrella_term = "neurodegenerative_disease"
regularization = "l1"
MODEL_NAME = "word2vec"

PROJECT_ROOT = SRC_DIR.parent
CORPUS_CSV_PATH = PROJECT_ROOT / f"data/corpus/preprocessed/corpus_{umbrella_term}_preprocessed.csv"
GENE_UNIVERSE_CSV_PATH = PROJECT_ROOT / f"data/corpus/extracted_genes/genes_extracted_{umbrella_term}.csv"

TEXT_COL = "text"
YEAR_COL = "year"
START_YEAR = 1970
END_YEAR = 2026
GENE_COL = "gene"

WORD2VEC_VECTOR_SIZE = 200
WORD2VEC_WINDOW = 8
WORD2VEC_MIN_COUNT = 1
WORD2VEC_SG = 1
WORD2VEC_NEGATIVE = 10
WORD2VEC_SAMPLE = 1e-5
WORD2VEC_EPOCHS = 15
WORD2VEC_WORKERS = max(1, (os.cpu_count() or 2) - 1)

OUT_DIR = SRC_DIR / f"scores/{umbrella_term}/all_scores_LR_{regularization}_{MODEL_NAME}_{umbrella_term}/"
CV_METRICS_JSON = OUT_DIR / "cv_metrics.json"
CV_OOF_GOLD_NPZ = OUT_DIR / "scores_oof_gold_only.npz"
FINAL_ALLGENES_NPZ = OUT_DIR / "scores_final_allgenes.npz"
W2V_MODEL_PATH = OUT_DIR / f"word2vec_{umbrella_term}.model"

N_FOLDS = 5
SEED = 42
DEFAULT_C = 0.1
C_SWEEP_VALUES = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0], dtype=np.float64)

C_SWEEP_JSON = OUT_DIR / "c_sweep_metrics.json"
C_SWEEP_CSV = OUT_DIR / "c_sweep_metrics.csv"
C_SWEEP_PLOT = OUT_DIR / "c_sweep_performance_sparsity.png"

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def set_seed(seed: int):
    np.random.seed(seed)


def regex_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def parse_year(raw_value: str) -> Optional[int]:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


class CorpusSentenceIterator:
    def __init__(
        self,
        corpus_csv_path: Path,
        text_col: str,
        year_col: str,
        start_year: int,
        end_year: int,
    ):
        self.corpus_csv_path = corpus_csv_path
        self.text_col = text_col
        self.year_col = year_col
        self.start_year = start_year
        self.end_year = end_year

    def __iter__(self):
        with open(self.corpus_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV sem cabeçalho: {self.corpus_csv_path}")
            missing = {self.text_col, self.year_col} - set(reader.fieldnames)
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(f"CSV sem coluna(s) esperada(s): {missing_str}")

            for row in reader:
                year = parse_year(row.get(self.year_col))
                if year is not None and (year < self.start_year or year > self.end_year):
                    continue

                tokens = regex_tokenize(row.get(self.text_col, ""))
                if tokens:
                    yield tokens


def load_gene_universe_from_csv(path: Path, gene_col: str = GENE_COL) -> Set[str]:
    genes: Set[str] = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or gene_col not in reader.fieldnames:
            raise ValueError(f"Coluna '{gene_col}' não encontrada em {path}")

        for row in reader:
            gene = str(row.get(gene_col, "")).strip().upper()
            if gene:
                genes.add(gene)
    return genes


def train_word2vec(corpus_csv_path: Path):
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError(
            "Dependência ausente: `gensim`. Instale com `pip install gensim` para treinar Word2Vec."
        ) from exc

    sentences = CorpusSentenceIterator(
        corpus_csv_path=corpus_csv_path,
        text_col=TEXT_COL,
        year_col=YEAR_COL,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    print("[info] Training Word2Vec...")
    print(
        f"[info] W2V params: dim={WORD2VEC_VECTOR_SIZE} window={WORD2VEC_WINDOW} "
        f"min_count={WORD2VEC_MIN_COUNT} sg={WORD2VEC_SG} negative={WORD2VEC_NEGATIVE} "
        f"epochs={WORD2VEC_EPOCHS} workers={WORD2VEC_WORKERS}"
    )
    model = Word2Vec(
        sentences=sentences,
        vector_size=WORD2VEC_VECTOR_SIZE,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        sg=WORD2VEC_SG,
        negative=WORD2VEC_NEGATIVE,
        sample=WORD2VEC_SAMPLE,
        epochs=WORD2VEC_EPOCHS,
        workers=WORD2VEC_WORKERS,
        seed=SEED,
    )
    return model


def build_gene_embeddings(word2vec_model, gene_universe: Set[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    gene_embeddings: Dict[str, np.ndarray] = {}
    gene_token_counts: Dict[str, int] = {}

    for gene in sorted(gene_universe):
        token = gene.lower()
        if token not in word2vec_model.wv:
            continue

        gene_embeddings[gene] = word2vec_model.wv[token].astype(np.float32)
        try:
            token_count = int(word2vec_model.wv.get_vecattr(token, "count"))
        except (KeyError, AttributeError, TypeError):
            token_count = 1
        gene_token_counts[gene] = max(token_count, 1)

    return gene_embeddings, gene_token_counts


def get_all_negatives(
    gene_dict: Dict[str, np.ndarray],
    exclude: Set[str],
) -> List[str]:
    return [g for g in gene_dict.keys() if g not in exclude]


def compute_fold_metrics(ranked_genes: List[str], val_gold: Set[str]) -> Dict[str, float]:
    if len(val_gold) == 0:
        return {}

    ranks = {g: i + 1 for i, g in enumerate(ranked_genes)}
    found_ranks = [ranks[g] for g in val_gold if g in ranks]

    def recall_at(k: int) -> float:
        topk = set(ranked_genes[:k])
        return float(len(val_gold & topk) / len(val_gold))

    mrr = sum((1.0 / ranks[g]) for g in val_gold if g in ranks) / len(val_gold)
    mrr20 = sum((1.0 / ranks[g]) for g in val_gold if g in ranks and ranks[g] <= 20) / len(val_gold)

    return {
        "n_val_gold": float(len(val_gold)),
        "auc": float("nan"),
        "recall@10": recall_at(10),
        "recall@50": recall_at(50),
        "recall@100": recall_at(100),
        "mrr": float(mrr),
        "mrr@20": float(mrr20),
        "mean_rank": float(np.mean(found_ranks)) if found_ranks else float("nan"),
        "median_rank": float(np.median(found_ranks)) if found_ranks else float("nan"),
    }


def fold_mean_std(cv_folds: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    vals = [f[key] for f in cv_folds if key in f and not np.isnan(f[key])]
    if len(vals) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def fmt_pm(mean: float, std: float, decimals: int = 3) -> str:
    if np.isnan(mean) or np.isnan(std):
        return "nan"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def build_training_data(
    gene_embeddings: Dict[str, np.ndarray],
    pos_genes: List[str],
    neg_genes: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for g in pos_genes:
        if g in gene_embeddings:
            X.append(gene_embeddings[g])
            y.append(1)

    for g in neg_genes:
        if g in gene_embeddings:
            X.append(gene_embeddings[g])
            y.append(0)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def build_lr_pipeline(c_value: float, random_state: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=regularization,
            solver="liblinear",
            C=float(c_value),
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
    )


def count_selected_features(model, tol: float = 1e-12) -> int:
    coef = model.named_steps["logisticregression"].coef_
    return int(np.count_nonzero(np.abs(coef) > tol))


def list_mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def save_c_sweep_csv(rows: List[Dict[str, float]], out_csv: Path):
    columns = [
        "C",
        "pr_auc_mean",
        "pr_auc_std",
        "roc_auc_mean",
        "roc_auc_std",
        "selected_features_mean",
        "selected_features_std",
        "selected_features_pct_mean",
        "selected_features_pct_std",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_c_sweep(rows: List[Dict[str, float]], out_png: Path, embedding_dim: int):
    c_vals = np.array([r["C"] for r in rows], dtype=np.float64)
    pr_mean = np.array([r["pr_auc_mean"] for r in rows], dtype=np.float64)
    pr_std = np.array([r["pr_auc_std"] for r in rows], dtype=np.float64)
    roc_mean = np.array([r["roc_auc_mean"] for r in rows], dtype=np.float64)
    roc_std = np.array([r["roc_auc_std"] for r in rows], dtype=np.float64)
    sel_mean = np.array([r["selected_features_mean"] for r in rows], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_xscale("log")
    ax1.plot(c_vals, pr_mean, marker="o", label="Avg-PR (CV mean)", color="#1f77b4")
    ax1.plot(c_vals, roc_mean, marker="s", label="ROC-AUC (CV mean)", color="#ff7f0e")
    ax1.fill_between(c_vals, pr_mean - pr_std, pr_mean + pr_std, color="#1f77b4", alpha=0.15)
    ax1.fill_between(c_vals, roc_mean - roc_std, roc_mean + roc_std, color="#ff7f0e", alpha=0.15)
    ax1.set_xlabel("C (inverse regularization strength)")
    ax1.set_ylabel("Validation performance")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(c_vals, sel_mean, marker="^", linestyle="-.", color="black", label="Selected variables (non-zero coef)")
    ax2.set_ylabel(f"Selected variables (max {embedding_dim})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(f"{regularization} Regularization Sweep: Performance vs Selected Variables")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def compute_learning_curve(
    X_train,
    y_train,
    X_val,
    y_val,
    fold_idx,
    out_dir: Path,
    c_value: float = DEFAULT_C,
    max_subsets: int = 10,
):
    train_sizes = np.linspace(0.1, 1.0, max_subsets)
    train_scores, val_scores = [], []

    n_samples = X_train.shape[0]
    for frac in train_sizes:
        n = max(1, int(n_samples * frac))
        idx = np.random.choice(n_samples, n, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]

        model = build_lr_pipeline(c_value=c_value, random_state=SEED + fold_idx)
        model.fit(X_sub, y_sub)

        y_train_pred = model.predict_proba(X_sub)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]

        train_scores.append(average_precision_score(y_sub, y_train_pred))
        val_scores.append(average_precision_score(y_val, y_val_pred))

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes * 100, train_scores, "o-", label="Train AP")
    plt.plot(train_sizes * 100, val_scores, "o-", label="Validation AP")
    plt.xlabel("Percent of Training Data (%)")
    plt.ylabel("Average Precision")
    plt.title(f"Learning Curve Fold {fold_idx + 1}")
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"learning_curve_fold{fold_idx + 1}.png")
    plt.close()


def main():
    if not CORPUS_CSV_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_CSV_PATH}")
    if not GENE_UNIVERSE_CSV_PATH.exists():
        raise FileNotFoundError(f"Gene universe not found: {GENE_UNIVERSE_CSV_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    word2vec_model = train_word2vec(CORPUS_CSV_PATH)
    word2vec_model.save(str(W2V_MODEL_PATH))
    print(f"[info] Saved Word2Vec model to: {W2V_MODEL_PATH}")

    print("[info] Loading candidate gene universe...")
    gene_universe = load_gene_universe_from_csv(GENE_UNIVERSE_CSV_PATH, gene_col=GENE_COL)
    gene_universe |= {str(g).strip().upper() for g in VALIDATION_GENES}

    print("[info] Building single embedding per gene from Word2Vec...")
    gene_embeddings, gene_token_counts = build_gene_embeddings(word2vec_model, gene_universe)
    print(f"[info] Genes with Word2Vec embedding: {len(gene_embeddings)} / {len(gene_universe)}")

    all_genes = sorted(gene_embeddings.keys())
    if not all_genes:
        raise ValueError("No genes with embeddings found in Word2Vec vocabulary.")

    all_genes_set = set(all_genes)
    X_all_genes = np.array([gene_embeddings[g] for g in all_genes], dtype=np.float32)

    gold_all = sorted({str(g).strip().upper() for g in VALIDATION_GENES})
    gold_available = [g for g in gold_all if g in all_genes_set]
    if len(gold_available) < N_FOLDS:
        raise ValueError(
            f"Gold genes available ({len(gold_available)}) is smaller than N_FOLDS ({N_FOLDS})."
        )

    gold_arr = np.array(gold_available)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    gold_oof_probs: Dict[str, float] = {}
    cv_folds: List[Dict[str, float]] = []
    c_sweep_fold_metrics: List[Dict[str, float]] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(gold_arr)):
        set_seed(SEED + fold)
        gold_train = gold_arr[tr_idx].tolist()
        gold_val = gold_arr[va_idx].tolist()
        gold_val_set = set(gold_val)

        print(f"\nCV FOLD {fold + 1}/{N_FOLDS}")

        pos_train = [g for g in gold_train if g in gene_embeddings]
        exclude_train = set(gold_train) | gold_val_set
        neg_train = get_all_negatives(gene_embeddings, exclude=exclude_train)
        neg_val = neg_train.copy()

        X_train, y_train = build_training_data(gene_embeddings, pos_train, neg_train)
        X_val, y_val = build_training_data(gene_embeddings, gold_val, neg_val)

        compute_learning_curve(X_train, y_train, X_val, y_val, fold, OUT_DIR, c_value=DEFAULT_C)

        print(f"[fold {fold + 1}] pos_train = {len(pos_train)} | neg_train = {len(neg_train)}")
        print(f"[fold {fold + 1}] pos_val   = {len(gold_val)} | neg_val   = {len(neg_val)}")
        print(f"[fold {fold + 1}] X_train shape = {X_train.shape} | y_train mean = {y_train.mean():.3f}")

        model = build_lr_pipeline(c_value=DEFAULT_C, random_state=SEED + fold)
        model.fit(X_train, y_train)

        print("[info] Scoring ALL genes for fold ranking...")
        probs = model.predict_proba(X_all_genes)[:, 1]
        scores = {g: prob for g, prob in zip(all_genes, probs)}

        for g in gold_val:
            gold_oof_probs[g] = scores[g]

        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fold_metrics = compute_fold_metrics(ranked, gold_val_set)

        y_true_val, y_score_val = [], []
        for g in gold_val:
            y_true_val.append(1)
            y_score_val.append(scores[g])

        for g in neg_val:
            if g in scores:
                y_true_val.append(0)
                y_score_val.append(scores[g])

        if len(set(y_true_val)) == 2:
            fold_metrics["auc"] = float(average_precision_score(y_true_val, y_score_val))
        else:
            fold_metrics["auc"] = float("nan")

        y_val_pred_default = model.predict_proba(X_val)[:, 1]
        selected_default = count_selected_features(model)
        for c_value in C_SWEEP_VALUES:
            c_value = float(c_value)
            if np.isclose(c_value, DEFAULT_C):
                model_c = model
                y_val_pred = y_val_pred_default
                selected_features = selected_default
            else:
                model_c = build_lr_pipeline(c_value=c_value, random_state=SEED + fold)
                model_c.fit(X_train, y_train)
                y_val_pred = model_c.predict_proba(X_val)[:, 1]
                selected_features = count_selected_features(model_c)

            pr_auc = float(average_precision_score(y_val, y_val_pred))
            if len(np.unique(y_val)) == 2:
                roc_auc = float(roc_auc_score(y_val, y_val_pred))
            else:
                roc_auc = float("nan")

            c_sweep_fold_metrics.append(
                {
                    "C": c_value,
                    "fold": float(fold + 1),
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                    "selected_features": float(selected_features),
                }
            )

        fold_metrics["fold"] = float(fold + 1)
        cv_folds.append(fold_metrics)

    metrics_keys = ["auc", "recall@10", "recall@50", "recall@100", "mrr", "mrr@20", "mean_rank", "median_rank"]
    summary_stats, summary_fmt = {}, {}

    for k in metrics_keys:
        mean, std = fold_mean_std(cv_folds, k)
        summary_stats[k] = {"mean": mean, "std": std}
        summary_fmt[k] = fmt_pm(mean, std, decimals=1 if k == "mean_rank" else 3)

    cv_summary = {
        "folds": cv_folds,
        "summary_mean_std": summary_stats,
        "summary_fmt": summary_fmt,
        "n_folds": N_FOLDS,
        "seed": SEED,
    }

    with open(CV_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)

    print(f"\n[info] Saved CV metrics to: {CV_METRICS_JSON}")
    for k in metrics_keys:
        print(f"  {k:10s}: {summary_fmt[k]}")

    embedding_dim = int(X_all_genes.shape[1])
    c_sweep_summary: List[Dict[str, float]] = []
    for c_value in C_SWEEP_VALUES:
        c_value = float(c_value)
        rows_c = [r for r in c_sweep_fold_metrics if np.isclose(r["C"], c_value)]
        pr_mean, pr_std = list_mean_std([r["pr_auc"] for r in rows_c])
        roc_mean, roc_std = list_mean_std([r["roc_auc"] for r in rows_c])
        sel_mean, sel_std = list_mean_std([r["selected_features"] for r in rows_c])
        sel_pct_mean, sel_pct_std = list_mean_std(
            [(100.0 * r["selected_features"] / embedding_dim) for r in rows_c]
        )
        c_sweep_summary.append(
            {
                "C": c_value,
                "pr_auc_mean": pr_mean,
                "pr_auc_std": pr_std,
                "roc_auc_mean": roc_mean,
                "roc_auc_std": roc_std,
                "selected_features_mean": sel_mean,
                "selected_features_std": sel_std,
                "selected_features_pct_mean": sel_pct_mean,
                "selected_features_pct_std": sel_pct_std,
            }
        )

    c_sweep_payload = {
        "regularization": regularization,
        "default_c": DEFAULT_C,
        "embedding_dim": embedding_dim,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "fold_metrics": c_sweep_fold_metrics,
        "summary": c_sweep_summary,
        "word2vec_params": {
            "vector_size": WORD2VEC_VECTOR_SIZE,
            "window": WORD2VEC_WINDOW,
            "min_count": WORD2VEC_MIN_COUNT,
            "sg": WORD2VEC_SG,
            "negative": WORD2VEC_NEGATIVE,
            "sample": WORD2VEC_SAMPLE,
            "epochs": WORD2VEC_EPOCHS,
            "workers": WORD2VEC_WORKERS,
            "year_start": START_YEAR,
            "year_end": END_YEAR,
        },
    }
    with open(C_SWEEP_JSON, "w", encoding="utf-8") as f:
        json.dump(c_sweep_payload, f, indent=2)
    save_c_sweep_csv(c_sweep_summary, C_SWEEP_CSV)
    plot_c_sweep(c_sweep_summary, C_SWEEP_PLOT, embedding_dim)

    print(f"\n[info] Saved C sweep JSON to: {C_SWEEP_JSON}")
    print(f"[info] Saved C sweep CSV to: {C_SWEEP_CSV}")
    print(f"[info] Saved C sweep plot to: {C_SWEEP_PLOT}")
    for row in c_sweep_summary:
        print(
            f"  C={row['C']:<5g} | PR-AUC={row['pr_auc_mean']:.3f} ± {row['pr_auc_std']:.3f} | "
            f"ROC-AUC={row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f} | "
            f"selected={row['selected_features_mean']:.1f}/{embedding_dim} ({row['selected_features_pct_mean']:.1f}%)"
        )

    oof_genes = sorted(gold_oof_probs.keys())
    oof_scores = np.array([gold_oof_probs[g] for g in oof_genes], dtype=np.float32)
    oof_counts = np.array([gene_token_counts.get(g, 1) for g in oof_genes], dtype=np.int32)

    np.savez_compressed(
        CV_OOF_GOLD_NPZ,
        genes=np.array(oof_genes, dtype=np.str_),
        scores_topm=oof_scores,
        ctx_counts=oof_counts,
        meta=np.array(
            [json.dumps({"method": f"LogReg_{regularization}_Pipeline_word2vec", "folds": N_FOLDS, "seed": SEED})],
            dtype=np.str_,
        ),
    )

    print("\nFINAL MODEL TRAINING (ALL GOLD)")
    set_seed(SEED + 999)

    pos_final = [g for g in gold_available if g in gene_embeddings]
    neg_final = get_all_negatives(
        gene_embeddings,
        exclude=set(gold_available),
    )

    X_train_final, y_train_final = build_training_data(gene_embeddings, pos_final, neg_final)

    model_final = build_lr_pipeline(c_value=DEFAULT_C, random_state=SEED + 999)
    model_final.fit(X_train_final, y_train_final)

    final_probs = model_final.predict_proba(X_all_genes)[:, 1]
    final_counts = [gene_token_counts.get(g, 1) for g in all_genes]

    np.savez_compressed(
        FINAL_ALLGENES_NPZ,
        genes=np.array(all_genes, dtype=np.str_),
        scores_topm=np.array(final_probs, dtype=np.float32),
        ctx_counts=np.array(final_counts, dtype=np.int32),
        meta=np.array([json.dumps({"method": f"LogReg_{regularization}_Pipeline_word2vec", "seed": SEED})], dtype=np.str_),
    )
    print(f"[info] Saved FINAL all-genes NPZ to: {FINAL_ALLGENES_NPZ}")


if __name__ == "__main__":
    main()

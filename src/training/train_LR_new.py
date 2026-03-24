import os
import csv
import json
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import VALIDATION_GENES

umbrella_term = "neurodegenerative_disease"
regularization = 'l2'
MODEL_NAME = "pubmedbert"
FEATURES_PATH = f"../features/featuresUPPER_{MODEL_NAME}_{umbrella_term}/features_ALS_{MODEL_NAME}.pkl"

OUT_DIR = f"../scores/{umbrella_term}UPPER/all_scores_LR_{regularization}_{MODEL_NAME}_{umbrella_term}/"
CV_METRICS_JSON = os.path.join(OUT_DIR, "cv_metrics.json")
CV_OOF_GOLD_NPZ = os.path.join(OUT_DIR, "scores_oof_gold_only.npz")
FINAL_ALLGENES_NPZ = os.path.join(OUT_DIR, "scores_final_allgenes.npz")

N_FOLDS = 5
SEED = 42
MAX_INST_TRAIN = 500
DEFAULT_C = 0.1
C_SWEEP_VALUES = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0], dtype=np.float64)

C_SWEEP_JSON = os.path.join(OUT_DIR, "c_sweep_metrics.json")
C_SWEEP_CSV = os.path.join(OUT_DIR, "c_sweep_metrics.csv")
C_SWEEP_PLOT = os.path.join(OUT_DIR, "c_sweep_performance_sparsity.png")

def set_seed(seed: int):
    np.random.seed(seed)

def subsample(vectors: List[np.ndarray], max_inst: int) -> List[np.ndarray]:
    if len(vectors) <= max_inst:
        return vectors
    idx = np.random.choice(len(vectors), max_inst, replace=False)
    return [vectors[i] for i in idx]

def get_all_negatives(
    gene_dict: Dict[str, List[np.ndarray]],
    exclude: Set[str],
) -> List[str]:
    return [g for g, v in gene_dict.items() if g not in exclude and len(v) > 0]

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

def build_training_data(gene_vectors, pos_genes, neg_genes):
    X, y = [], []
    for g in pos_genes:
        vecs = subsample(gene_vectors[g], MAX_INST_TRAIN)
        X.append(np.mean(vecs, axis=0))
        y.append(1)
        
    for g in neg_genes:
        vecs = subsample(gene_vectors[g], MAX_INST_TRAIN)
        X.append(np.mean(vecs, axis=0))
        y.append(0)
        
    return np.array(X), np.array(y)

def build_lr_pipeline(c_value: float, random_state: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=regularization,
            solver='liblinear',
            C=float(c_value),
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state,
        )
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

def save_c_sweep_csv(rows: List[Dict[str, float]], out_csv: str):
    columns = [
        "C",
        "pr_auc_mean", "pr_auc_std",
        "roc_auc_mean", "roc_auc_std",
        "selected_features_mean", "selected_features_std",
        "selected_features_pct_mean", "selected_features_pct_std",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def plot_c_sweep(rows: List[Dict[str, float]], out_png: str, embedding_dim: int):
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

def compute_learning_curve(X_train, y_train, X_val, y_val, fold_idx, out_dir, c_value: float = DEFAULT_C, max_subsets=10):
    """Computa learning curves usando subsets do treino e salva o gráfico em PNG."""
    train_sizes = np.linspace(0.1, 1.0, max_subsets)
    train_scores, val_scores = [], []

    n_samples = X_train.shape[0]
    for frac in train_sizes:
        n = max(1, int(n_samples * frac))
        idx = np.random.choice(n_samples, n, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]

        model = build_lr_pipeline(c_value=c_value, random_state=SEED + fold_idx)
        model.fit(X_sub, y_sub)

        # Métricas
        y_train_pred = model.predict_proba(X_sub)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]

        train_scores.append(average_precision_score(y_sub, y_train_pred))
        val_scores.append(average_precision_score(y_val, y_val_pred))

    # Salva gráfico
    plt.figure(figsize=(6,4))
    plt.plot(train_sizes * 100, train_scores, 'o-', label='Train AP')
    plt.plot(train_sizes * 100, val_scores, 'o-', label='Validation AP')
    plt.xlabel("Percent of Training Data (%)")
    plt.ylabel("Average Precision")
    plt.title(f"Learning Curve Fold {fold_idx+1}")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"learning_curve_fold{fold_idx+1}.png"))
    plt.close()



def main():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features not found: {FEATURES_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("[info] Loading features...")
    with open(FEATURES_PATH, "rb") as f:
        gene_vectors: Dict[str, List[np.ndarray]] = pickle.load(f)

    all_genes = [g for g, v in gene_vectors.items() if len(v) > 0]
    all_genes_set = set(all_genes)
    
    print("[info] Pre-computing mean vectors for all genes...")
    X_all_genes = np.array([np.mean(gene_vectors[g], axis=0) for g in all_genes])

    

    gold_all = sorted({str(g).strip().upper() for g in VALIDATION_GENES})
    gold_available = [g for g in gold_all if g in all_genes_set]
    
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

        print(f"\nCV FOLD {fold+1}/{N_FOLDS}")

        pos_train = [g for g in gold_train if len(gene_vectors[g]) > 0]
        
        # 1. Negativos para TREINO
        exclude_train = set(gold_train) | gold_val_set
        neg_train = get_all_negatives(gene_vectors, exclude=exclude_train)

        # 2. Usa o mesmo conjunto de negativos na validação
        neg_val = neg_train.copy()

        X_train, y_train = build_training_data(gene_vectors, pos_train, neg_train)
        X_val, y_val = build_training_data(gene_vectors, gold_val, neg_val)

        #compute_learning_curve(X_train, y_train, X_val, y_val, fold, OUT_DIR, c_value=DEFAULT_C)

        print(f"[fold {fold+1}] pos_train = {len(pos_train)} | neg_train = {len(neg_train)}")
        print(f"[fold {fold+1}] pos_val   = {len(gold_val)} | neg_val   = {len(neg_val)}")
        print(f"[fold {fold+1}] X_train shape = {X_train.shape} | y_train mean = {y_train.mean():.3f}")
        
        
        

        # Modelo principal com C padrão (compatível com os artefatos já usados)
        model = build_lr_pipeline(c_value=DEFAULT_C, random_state=SEED + fold)
        model.fit(X_train, y_train)

        print(f"[info] Scoring ALL genes for fold ranking...")
        probs = model.predict_proba(X_all_genes)[:, 1]
        scores = {g: prob for g, prob in zip(all_genes, probs)}

        for g in gold_val:
            gold_oof_probs[g] = scores[g]

        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fold_metrics = compute_fold_metrics(ranked, gold_val_set)

        # AUC agora usa apenas ouro da validação e negativos separados
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

        # Sweep de C para medir trade-off desempenho vs esparsidade (L1)
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

    with open(CV_METRICS_JSON, "w") as f:
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
    }
    with open(C_SWEEP_JSON, "w") as f:
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
    oof_counts = np.array([len(gene_vectors[g]) for g in oof_genes], dtype=np.int32)

    np.savez_compressed(
        CV_OOF_GOLD_NPZ,
        genes=np.array(oof_genes, dtype=np.str_),
        scores_topm=oof_scores,
        ctx_counts=oof_counts,
        meta=np.array([json.dumps({f"method": "LogReg_{regularization}_Pipeline", "folds": N_FOLDS, "seed": SEED})], dtype=np.str_),
    )

    print("\nFINAL MODEL TRAINING (ALL GOLD)")
    set_seed(SEED + 999)
    
    pos_final = [g for g in gold_available if len(gene_vectors[g]) > 0]
    neg_final = get_all_negatives(
        gene_vectors,
        exclude=set(gold_available),
    )

    X_train_final, y_train_final = build_training_data(gene_vectors, pos_final, neg_final)
    
    model_final = build_lr_pipeline(c_value=DEFAULT_C, random_state=SEED + 999)
    model_final.fit(X_train_final, y_train_final)

    final_probs = model_final.predict_proba(X_all_genes)[:, 1]
    final_counts = [len(gene_vectors[g]) for g in all_genes]

    np.savez_compressed(
        FINAL_ALLGENES_NPZ,
        genes=np.array(all_genes, dtype=np.str_),
        scores_topm=np.array(final_probs, dtype=np.float32),
        ctx_counts=np.array(final_counts, dtype=np.int32),
        meta=np.array([json.dumps({f"method": "LogReg_{regularization}_Pipeline", "seed": SEED})], dtype=np.str_),
    )
    print(f"[info] Saved FINAL all-genes NPZ to: {FINAL_ALLGENES_NPZ}")

if __name__ == "__main__":
    main()

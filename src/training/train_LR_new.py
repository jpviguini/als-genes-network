import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import VALIDATION_GENES

umbrella_term = "neurodegenerative_disease"
MODEL_NAME = "pubmedbert"
FEATURES_PATH = f"./features_{MODEL_NAME}_{umbrella_term}/features_ALS_{MODEL_NAME}.pkl"

OUT_DIR = f"./1:4_scores_LR_none_{MODEL_NAME}_{umbrella_term}/"
CV_METRICS_JSON = os.path.join(OUT_DIR, "cv_metrics.json")
CV_OOF_GOLD_NPZ = os.path.join(OUT_DIR, "scores_oof_gold_only.npz")
FINAL_ALLGENES_NPZ = os.path.join(OUT_DIR, "scores_final_allgenes.npz")

N_FOLDS = 5
SEED = 42
NEG_RATIO = 4
RELIABLE_NEG_MAX_BAG = 10
MAX_INST_TRAIN = 500

def set_seed(seed: int):
    np.random.seed(seed)

def subsample(vectors: List[np.ndarray], max_inst: int) -> List[np.ndarray]:
    if len(vectors) <= max_inst:
        return vectors
    idx = np.random.choice(len(vectors), max_inst, replace=False)
    return [vectors[i] for i in idx]

def get_reliable_negatives(
    gene_dict: Dict[str, List[np.ndarray]],
    exclude: Set[str],
    n_needed: int,
    max_bag: int = 5,
) -> List[str]:

    low = [g for g, v in gene_dict.items() if (g not in exclude and 0 < len(v) < max_bag)]

    if len(low) >= n_needed:
        return list(np.random.choice(low, size=n_needed, replace=False))

    remaining = [g for g, v in gene_dict.items() if (g not in exclude and g not in set(low) and len(v) > 0)]
    need = n_needed - len(low)
    if need > 0:
        if need >= len(remaining):
            low.extend(remaining)
        else:
            low.extend(list(np.random.choice(remaining, size=need, replace=False)))
    return low

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

def compute_learning_curve(X_train, y_train, X_val, y_val, fold_idx, out_dir, max_subsets=10):
    """Computa learning curves usando subsets do treino e salva o gráfico em PNG."""
    train_sizes = np.linspace(0.1, 1.0, max_subsets)
    train_scores, val_scores = [], []

    n_samples = X_train.shape[0]
    for frac in train_sizes:
        n = max(1, int(n_samples * frac))
        idx = np.random.choice(n_samples, n, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty=None, solver='lbfgs', C=0.1,
                class_weight='balanced', max_iter=1000, random_state=SEED + fold_idx
            )
        )
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

    for fold, (tr_idx, va_idx) in enumerate(kf.split(gold_arr)):
        set_seed(SEED + fold)
        gold_train = gold_arr[tr_idx].tolist()
        gold_val = gold_arr[va_idx].tolist()
        gold_val_set = set(gold_val)

        print(f"\nCV FOLD {fold+1}/{N_FOLDS}")

        pos_train = [g for g in gold_train if len(gene_vectors[g]) > 0]
        
        # 1. Negativos para TREINO
        exclude_train = set(gold_train) | gold_val_set
        neg_train = get_reliable_negatives(
            gene_vectors,
            exclude=exclude_train,
            n_needed=len(pos_train) * NEG_RATIO,
            max_bag=RELIABLE_NEG_MAX_BAG,
        )

        # 2. Negativos separados para VALIDAÇÃO
        exclude_val = exclude_train | set(neg_train)
        neg_val = get_reliable_negatives(
            gene_vectors,
            exclude=exclude_val,
            n_needed=len(gold_val) * NEG_RATIO,
            max_bag=RELIABLE_NEG_MAX_BAG,
        )

        X_train, y_train = build_training_data(gene_vectors, pos_train, neg_train)
        X_val, y_val = build_training_data(gene_vectors, gold_val, neg_val)

        compute_learning_curve(X_train, y_train, X_val, y_val, fold, OUT_DIR)

        print(f"[fold {fold+1}] pos_train = {len(pos_train)} | neg_train = {len(neg_train)}")
        print(f"[fold {fold+1}] pos_val   = {len(gold_val)} | neg_val   = {len(neg_val)}")
        print(f"[fold {fold+1}] X_train shape = {X_train.shape} | y_train mean = {y_train.mean():.3f}")
        
        
        

        # PIPELINE: StandardScaler + Regressão Logística l1
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty=None, 
                solver='lbfgs', 
                C=0.1, 
                class_weight='balanced',
                max_iter=1000, 
                random_state=SEED + fold
            )
        )
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

    oof_genes = sorted(gold_oof_probs.keys())
    oof_scores = np.array([gold_oof_probs[g] for g in oof_genes], dtype=np.float32)
    oof_counts = np.array([len(gene_vectors[g]) for g in oof_genes], dtype=np.int32)

    np.savez_compressed(
        CV_OOF_GOLD_NPZ,
        genes=np.array(oof_genes, dtype=np.str_),
        scores_topm=oof_scores,
        ctx_counts=oof_counts,
        meta=np.array([json.dumps({"method": "LogReg_l1_Pipeline", "folds": N_FOLDS, "seed": SEED})], dtype=np.str_),
    )

    print("\nFINAL MODEL TRAINING (ALL GOLD)")
    set_seed(SEED + 999)
    
    pos_final = [g for g in gold_available if len(gene_vectors[g]) > 0]
    neg_final = get_reliable_negatives(
        gene_vectors,
        exclude=set(gold_available),
        n_needed=len(pos_final) * NEG_RATIO,
        max_bag=RELIABLE_NEG_MAX_BAG,
    )

    X_train_final, y_train_final = build_training_data(gene_vectors, pos_final, neg_final)
    
    model_final = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=None, 
            solver='lbfgs', 
            C=0.1, 
            class_weight='balanced',
            max_iter=1000, 
            random_state=SEED + 999
        )
    )
    model_final.fit(X_train_final, y_train_final)

    final_probs = model_final.predict_proba(X_all_genes)[:, 1]
    final_counts = [len(gene_vectors[g]) for g in all_genes]

    np.savez_compressed(
        FINAL_ALLGENES_NPZ,
        genes=np.array(all_genes, dtype=np.str_),
        scores_topm=np.array(final_probs, dtype=np.float32),
        ctx_counts=np.array(final_counts, dtype=np.int32),
        meta=np.array([json.dumps({"method": "LogReg_l1_Pipeline", "seed": SEED})], dtype=np.str_),
    )
    print(f"[info] Saved FINAL all-genes NPZ to: {FINAL_ALLGENES_NPZ}")

if __name__ == "__main__":
    main()
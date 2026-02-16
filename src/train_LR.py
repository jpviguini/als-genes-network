import os
import json
import pickle
import numpy as np
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score

from config import VALIDATION_GENES


umbrella_term = "neurodegenerative_disease"
MODEL_NAME = "pubmedbert"
FEATURES_PATH = f"./features_{MODEL_NAME}_{umbrella_term}/features_ALS_{MODEL_NAME}.pkl"

# Alterei o nome da pasta de saída para refletir que é LR (Logistic Regression)
OUT_DIR = f"./scores_LR_{MODEL_NAME}_{umbrella_term}/"
CV_METRICS_JSON = os.path.join(OUT_DIR, "cv_metrics.json")
CV_OOF_GOLD_NPZ = os.path.join(OUT_DIR, "scores_oof_gold_only.npz")
FINAL_ALLGENES_NPZ = os.path.join(OUT_DIR, "scores_final_allgenes.npz")

INPUT_DIM = 1536

# --- PARÂMETROS REMOVIDOS (Não usados na Regressão Logística) ---
# HIDDEN_DIM = 256
# ATTENTION_DIM = 128

N_FOLDS = 5
EPOCHS = 20
LR = 1e-3  # Aumentei levemente pois LR converge mais fácil
WEIGHT_DECAY = 1e-3

NEG_RATIO = 4
RELIABLE_NEG_MAX_BAG = 5
MAX_INST_TRAIN = 100

GRAD_CLIP = 1.0
USE_AMP_TRAIN = True

INFER_CHUNK_SIZE = 512
USE_AMP_INFER = True

SEED = 42


# --- NOVO MODELO: REGRESSÃO LOGÍSTICA (MIL via Mean Pooling) ---
class LogisticRegressionMIL(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # Apenas uma camada linear: Wx + b
        # Sem camadas ocultas, sem ativações não-lineares internas
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, bag_features: torch.Tensor) -> torch.Tensor:
        # Passo 1: Agregação (Mean Pooling)
        # Transforma a "Bag" (N vetores) em 1 vetor representativo do gene
        # Shape entrada: (N_instancias, input_dim) -> Saída: (input_dim,)
        gene_embedding = torch.mean(bag_features, dim=0)
        
        # Passo 2: Classificador Linear
        logit = self.classifier(gene_embedding).view(1) # (1,)
        
        return logit


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


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


@torch.no_grad()
def infer_logit_simple(
    model: LogisticRegressionMIL,
    vectors: List[np.ndarray],
    device: torch.device,
) -> float:
    """
    Inferência simplificada para Regressão Logística.
    Como é apenas uma média, não precisamos de streaming complexo de atenção.
    """
    model.eval()
    model.to(device)

    if len(vectors) == 0:
        return float("nan")

    # Calcula a média dos vetores no lado da CPU (numpy) para economizar VRAM
    # Se a bag for muito grande, np.mean lida bem.
    all_vecs = np.asarray(vectors, dtype=np.float32)
    mean_vec = np.mean(all_vecs, axis=0) # (input_dim,)

    # Passa apenas o vetor médio para a GPU
    x = torch.as_tensor(mean_vec, device=device)
    
    with torch.cuda.amp.autocast(enabled=(USE_AMP_INFER and device.type == "cuda")):
        logit = model.classifier(x).view(1)

    return float(logit.item())


def train_model(
    gene_vectors: Dict[str, List[np.ndarray]],
    gold_train: List[str],
    gold_val_set: Set[str],
    device: torch.device,
    seed: int,
) -> LogisticRegressionMIL:
    set_seed(seed)

    # Instancia o modelo simples
    model = LogisticRegressionMIL(INPUT_DIM).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP_TRAIN and device.type == "cuda"))

    pos = [g for g in gold_train if g in gene_vectors and len(gene_vectors[g]) > 0]
    if len(pos) == 0:
        raise ValueError("No positive genes with features in this split.")

    last_neg = []

    model.train()
    for epoch in range(EPOCHS):
        exclude = set(gold_train) | set(gold_val_set)
        neg = get_reliable_negatives(
            gene_vectors,
            exclude=exclude,
            n_needed=len(pos) * NEG_RATIO,
            max_bag=RELIABLE_NEG_MAX_BAG,
        )
        last_neg = neg 

        genes = pos + neg
        labels = [1.0] * len(pos) + [0.0] * len(neg)

        perm = np.random.permutation(len(genes))
        genes = [genes[i] for i in perm]
        labels = [labels[i] for i in perm]

        total_loss = 0.0
        for g, y in zip(genes, labels):
            vecs = gene_vectors[g]
            if len(vecs) == 0:
                continue
            
            # Subsample ainda é útil para treino rápido e evitar overfitting em bags gigantes
            vecs = subsample(vecs, MAX_INST_TRAIN)

            bag = torch.as_tensor(np.asarray(vecs, dtype=np.float32), device=device)
            target = torch.tensor([y], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP_TRAIN and device.type == "cuda")):
                # Forward simples (agora retorna só logit)
                logit = model(bag)
                loss = criterion(logit, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(genes))
        print(f"Epoch {epoch+1:02d}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")

    model.last_negatives = last_neg
    return model


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


def main():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features not found: {FEATURES_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        # Se for CPU, desativa AMP para evitar erros
        global USE_AMP_TRAIN, USE_AMP_INFER
        USE_AMP_TRAIN = False
        USE_AMP_INFER = False
    
    print(f"[info] Device: {device}")

    print("[info] Loading features...")
    with open(FEATURES_PATH, "rb") as f:
        gene_vectors: Dict[str, List[np.ndarray]] = pickle.load(f)

    all_genes = [g for g, v in gene_vectors.items() if len(v) > 0]
    all_genes_set = set(all_genes)
    print(f"[info] Genes with features: {len(all_genes)}")

    gold_all = sorted({str(g).strip().upper() for g in VALIDATION_GENES})
    gold_available = [g for g in gold_all if g in all_genes_set]
    print(f"[info] Gold genes with features: {len(gold_available)}")

    gold_arr = np.array(gold_available)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    gold_oof_probs: Dict[str, float] = {}
    cv_folds: List[Dict[str, float]] = []

    # cross validation
    for fold, (tr_idx, va_idx) in enumerate(kf.split(gold_arr)):
        gold_train = gold_arr[tr_idx].tolist()
        gold_val = gold_arr[va_idx].tolist()
        gold_val_set = set(gold_val)

        print(f"\nCV FOLD {fold+1}/{N_FOLDS}")
        print(f"[info] train gold={len(gold_train)} val gold={len(gold_val)}")

        model = train_model(
            gene_vectors,
            gold_train=gold_train,
            gold_val_set=gold_val_set,
            device=device,
            seed=SEED + fold,
        )

        print(f"[info] Scoring ALL genes for fold ranking ({len(all_genes)} genes)...")
        scores: Dict[str, float] = {}
        
        # Scoring simplificado (infer_logit_simple)
        for g in tqdm(all_genes, desc=f"Fold {fold+1} scoring", mininterval=1.0):
            logit = infer_logit_simple(
                model=model,
                vectors=gene_vectors[g],
                device=device
            )
            scores[g] = sigmoid(logit)

        # OOF probs for gold_val
        for g in gold_val:
            gold_oof_probs[g] = scores[g]

        # Ranking (descending)
        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Ranking metrics
        fold_metrics = compute_fold_metrics(ranked, gold_val_set)

        # AUC
        y_true, y_score = [], []
        for g in gold_val:
            y_true.append(1)
            y_score.append(scores[g])

        neg_for_auc = getattr(model, "last_negatives", [])
        for g in neg_for_auc:
            if g in scores:
                y_true.append(0)
                y_score.append(scores[g])

        if len(set(y_true)) == 2:
            fold_metrics["auc"] = float(average_precision_score(y_true, y_score))
        else:
            fold_metrics["auc"] = float("nan")

        fold_metrics["fold"] = float(fold + 1)
        cv_folds.append(fold_metrics)

        del model
        torch.cuda.empty_cache()

    # Resumo
    metrics_keys = ["auc", "recall@10", "recall@50", "recall@100", "mrr", "mrr@20", "mean_rank", "median_rank"]
    summary_stats = {}
    summary_fmt = {}

    for k in metrics_keys:
        mean, std = fold_mean_std(cv_folds, k)
        summary_stats[k] = {"mean": mean, "std": std}
       
        if k == "mean_rank":
            summary_fmt[k] = fmt_pm(mean, std, decimals=1)
        else:
            summary_fmt[k] = fmt_pm(mean, std, decimals=3)

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
    print("[info] CV mean ± std (copy to table):")
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
        meta=np.array(
            [json.dumps({"method": "LogisticRegression_5Fold", "folds": N_FOLDS, "seed": SEED})],
            dtype=np.str_,
        ),
    )
    print(f"[info] Saved OOF gold-only NPZ to: {CV_OOF_GOLD_NPZ}")


    print("\nFINAL MODEL TRAINING (ALL GOLD)")
    model_final = train_model(
        gene_vectors,
        gold_train=gold_available,
        gold_val_set=set(),
        device=device,
        seed=SEED + 999,
    )


    print(f"[info] Final scoring ALL genes ({len(all_genes)} genes) ...")
    final_scores = []
    final_counts = []
    for g in tqdm(all_genes, desc="Final scoring", mininterval=1.0):
        logit = infer_logit_simple(
            model=model_final,
            vectors=gene_vectors[g],
            device=device
        )
        final_scores.append(sigmoid(logit))
        final_counts.append(len(gene_vectors[g]))

    np.savez_compressed(
        FINAL_ALLGENES_NPZ,
        genes=np.array(all_genes, dtype=np.str_),
        scores_topm=np.array(final_scores, dtype=np.float32),
        ctx_counts=np.array(final_counts, dtype=np.int32),
        meta=np.array([json.dumps({"method": "LogisticRegression_FinalModel", "seed": SEED})], dtype=np.str_),
    )
    print(f"[info] Saved FINAL all-genes NPZ to: {FINAL_ALLGENES_NPZ}")


if __name__ == "__main__":
    main()
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
from sklearn.metrics import roc_auc_score

from config import VALIDATION_GENES


MODEL_NAME = "scibert"
FEATURES_PATH = f"./features_MIL_{MODEL_NAME}/features_ALS_1970_2026.pkl"


OUT_DIR = f"./scores_mil_MEANPOOL_{MODEL_NAME}/"
CV_METRICS_JSON = os.path.join(OUT_DIR, "cv_metrics.json")
CV_OOF_GOLD_NPZ = os.path.join(OUT_DIR, "scores_oof_gold_only.npz")
FINAL_ALLGENES_NPZ = os.path.join(OUT_DIR, "scores_final_allgenes.npz")

INPUT_DIM = 2305
HIDDEN_DIM = 256

N_FOLDS = 5
EPOCHS = 20
LR = 5e-4
WEIGHT_DECAY = 1e-3

NEG_RATIO = 4
RELIABLE_NEG_MAX_BAG = 5
MAX_INST_TRAIN = 100

GRAD_CLIP = 1.0
USE_AMP_TRAIN = True

INFER_CHUNK_SIZE = 512
USE_AMP_INFER = True

SEED = 42



class MeanPoolingMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
       
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
      
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, bag_features: torch.Tensor) -> torch.Tensor:
        h = self.feature_extractor(bag_features)  # (N, hidden_dim)
        
        bag_rep = torch.mean(h, dim=0)  # (hidden_dim,)
        
        logit = self.classifier(bag_rep).view(1)  # (1,)
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
def infer_logit_mean_pooling(
    model: MeanPoolingMIL,
    vectors: List[np.ndarray],
    device: torch.device,
    chunk_size: int = 512,
    use_amp: bool = True,
) -> float:
    model.eval()
    model.to(device)

    N = len(vectors)
    if N == 0:
        return float("nan")

    # Accumulate sum of vectors to compute mean later
    running_sum = None

    for s in range(0, N, chunk_size):
        chunk = np.asarray(vectors[s:s + chunk_size], dtype=np.float32)
        x = torch.as_tensor(chunk, device=device)
        
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            h = model.feature_extractor(x) # (Chunk, Hidden)
            chunk_sum = torch.sum(h, dim=0) # (Hidden,)

        if running_sum is None:
            running_sum = chunk_sum
        else:
            running_sum += chunk_sum

    # Compute Mean
    bag_rep = running_sum / N

    # Classification
    with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
        logit = model.classifier(bag_rep).view(1)

    return float(logit.item())



def train_model(
    gene_vectors: Dict[str, List[np.ndarray]],
    gold_train: List[str],
    gold_val_set: Set[str],
    device: torch.device,
    seed: int,
) -> MeanPoolingMIL:
    set_seed(seed)

    model = MeanPoolingMIL(INPUT_DIM, HIDDEN_DIM).to(device)
    
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
            vecs = subsample(vecs, MAX_INST_TRAIN)

            bag = torch.as_tensor(np.asarray(vecs, dtype=np.float32), device=device)
            target = torch.tensor([y], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP_TRAIN and device.type == "cuda")):
                logit = model(bag) # forward only returns logits (without attention weights)
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
        print("Warning: Running on CPU. This might be slow.")

    print(f"[info] Device: {device}")
    print(f"[info] Output Dir: {OUT_DIR}")

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

        print(f"\n CV FOLD {fold+1}/{N_FOLDS}")
        print(f"[info] train gold={len(gold_train)} val gold={len(gold_val)}")

        model = train_model(
            gene_vectors,
            gold_train=gold_train,
            gold_val_set=gold_val_set,
            device=device,
            seed=SEED + fold,
        )

        # score ALL genes
        print(f"[info] Scoring ALL genes for fold ranking...")
        scores: Dict[str, float] = {}
        for g in tqdm(all_genes, desc=f"Fold {fold+1} scoring", mininterval=1.0):
            
            logit = infer_logit_mean_pooling(
                model=model,
                vectors=gene_vectors[g],
                device=device,
                chunk_size=INFER_CHUNK_SIZE,
                use_amp=USE_AMP_INFER,
            )
            scores[g] = sigmoid(logit)

        # OOF probs
        for g in gold_val:
            gold_oof_probs[g] = scores[g]

        # Ranking
        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
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
            fold_metrics["auc"] = float(roc_auc_score(y_true, y_score))
        else:
            fold_metrics["auc"] = float("nan")

        fold_metrics["fold"] = float(fold + 1)
        cv_folds.append(fold_metrics)

        del model
        torch.cuda.empty_cache()

    # summary
    metrics_keys = ["auc", "recall@10", "recall@50", "recall@100", "mrr", "mrr@20", "mean_rank", "median_rank"]
    summary_stats = {}
    summary_fmt = {}

    for k in metrics_keys:
        mean, std = fold_mean_std(cv_folds, k)
        summary_stats[k] = {"mean": mean, "std": std}
        decimals = 1 if "rank" in k else 3
        summary_fmt[k] = fmt_pm(mean, std, decimals=decimals)

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
    print("[info] CV mean ± std:")
    for k in metrics_keys:
        print(f"  {k:15s}: {summary_fmt[k]}")


    print("\nFINAL MODEL TRAINING (ALL GOLD)")
    model_final = train_model(
        gene_vectors,
        gold_train=gold_available,
        gold_val_set=set(),
        device=device,
        seed=SEED + 999,
    )

    print(f"[info] Final scoring ALL genes...")
    final_scores = []
    final_counts = []
    for g in tqdm(all_genes, desc="Final scoring", mininterval=1.0):
        logit = infer_logit_mean_pooling(
            model=model_final,
            vectors=gene_vectors[g],
            device=device,
            chunk_size=INFER_CHUNK_SIZE,
            use_amp=USE_AMP_INFER,
        )
        final_scores.append(sigmoid(logit))
        final_counts.append(len(gene_vectors[g]))

    np.savez_compressed(
        FINAL_ALLGENES_NPZ,
        genes=np.array(all_genes, dtype=np.str_),
        scores_topm=np.array(final_scores, dtype=np.float32),
        ctx_counts=np.array(final_counts, dtype=np.int32),
        meta=np.array([json.dumps({"method": "MIL_MEANPOOL_FinalModel", "seed": SEED})], dtype=np.str_),
    )
    print(f"[info] Saved FINAL all-genes NPZ to: {FINAL_ALLGENES_NPZ}")


if __name__ == "__main__":
    main()
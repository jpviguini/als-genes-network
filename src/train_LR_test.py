import os
import json
import pickle
import numpy as np
from typing import Dict, List, Set, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

from config import VALIDATION_GENES

# ==========================
# CONFIGURAÇÃO GERAL
# ==========================
umbrella_term = "neurodegenerative_disease"
MODEL_NAME = "pubmedbert"
FEATURES_PATH = f"./features_{MODEL_NAME}_{umbrella_term}/features_ALS_{MODEL_NAME}.pkl"
OT_JSON = "OT-MONDO_0004976-associated-targets-2_12_2026-v25_12.json"
VALIDATION_SOURCE = "eva"

OUT_DIR = f"./scores_LR_{MODEL_NAME}_{umbrella_term}_test/"
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_DIM = 1536
BOTTLENECK_LIST = [None, 768, 384, 192, 96, 48, 24, 16] 
N_FOLDS = 5
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-3
NEG_RATIO = 4
RELIABLE_NEG_MAX_BAG = 5
MAX_INST_TRAIN = 100
GRAD_CLIP = 1.0
USE_AMP_TRAIN = True
USE_AMP_INFER = True
SEED = 42


# ==========================
# FUNÇÕES ÚTEIS
# ==========================
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


def load_ot_target(json_path: str, target_source: str) -> Set[str]:
    """Retorna apenas genes com score >= 0.5 na fonte"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = []
    for entry in data:
        symbol = entry.get("symbol", "").strip().upper()
        score_raw = entry.get(target_source, 0)

        # converte para float com segurança
        try:
            score = float(score_raw)
        except (ValueError, TypeError):
            score = 0.0

        if score >= 0.5:
            df.append(symbol)
    return set(df)


# ==========================
# MODELO
# ==========================
class LogisticRegressionMIL(nn.Module):
    """MIL mean pooling + linear classifier, opcional bottleneck"""

    def __init__(self, input_dim: int, bottleneck_dim: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        if bottleneck_dim is None:
            self.proj = None
            self.classifier = nn.Linear(input_dim, 1)
        else:
            self.proj = nn.Linear(input_dim, bottleneck_dim, bias=False)
            self.classifier = nn.Linear(bottleneck_dim, 1)

    def forward(self, bag_features: torch.Tensor) -> torch.Tensor:
        gene_embedding = torch.mean(bag_features, dim=0)
        if self.proj is not None:
            gene_embedding = self.proj(gene_embedding)
        logit = self.classifier(gene_embedding).view(1)
        return logit


@torch.no_grad()
def infer_logit_simple(model: LogisticRegressionMIL, vectors: List[np.ndarray], device: torch.device) -> float:
    if len(vectors) == 0:
        return float("nan")
    all_vecs = np.asarray(vectors, dtype=np.float32)
    mean_vec = np.mean(all_vecs, axis=0)
    x = torch.as_tensor(mean_vec, device=device)
    with torch.cuda.amp.autocast(enabled=(USE_AMP_INFER and device.type == "cuda")):
        if model.proj is not None:
            x = model.proj(x)
        logit = model.classifier(x).view(1)
    return float(logit.item())


def train_model(
    gene_vectors: Dict[str, List[np.ndarray]],
    gold_train: List[str],
    gold_val_set: Set[str],
    device: torch.device,
    bottleneck_dim: Optional[int],
    seed: int,
) -> LogisticRegressionMIL:
    set_seed(seed)
    model = LogisticRegressionMIL(INPUT_DIM, bottleneck_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP_TRAIN and device.type == "cuda"))

    pos = [g for g in gold_train if g in gene_vectors and len(gene_vectors[g]) > 0]
    last_neg = []

    model.train()
    for epoch in range(EPOCHS):
        exclude = set(gold_train) | gold_val_set
        neg = get_reliable_negatives(gene_vectors, exclude=exclude, n_needed=len(pos) * NEG_RATIO, max_bag=RELIABLE_NEG_MAX_BAG)
        last_neg = neg

        genes = pos + neg
        labels = [1.0] * len(pos) + [0.0] * len(neg)
        perm = np.random.permutation(len(genes))
        genes = [genes[i] for i in perm]
        labels = [labels[i] for i in perm]

        total_loss = 0.0
        for g, y in zip(genes, labels):
            vecs = subsample(gene_vectors[g], MAX_INST_TRAIN)
            bag = torch.as_tensor(np.asarray(vecs, dtype=np.float32), device=device)
            target = torch.tensor([y], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP_TRAIN and device.type == "cuda")):
                logit = model(bag)
                loss = criterion(logit, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(genes))
        print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f} (bottleneck={bottleneck_dim})")

    model.last_negatives = last_neg
    return model


# ==========================
# TREINAMENTO E AUC POR BOTTLENECK
# ==========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        global USE_AMP_TRAIN, USE_AMP_INFER
        USE_AMP_TRAIN = False
        USE_AMP_INFER = False
    print(f"[info] Device: {device}")

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features não encontrado: {FEATURES_PATH}")

    print("[info] Carregando features...")
    with open(FEATURES_PATH, "rb") as f:
        gene_vectors: Dict[str, List[np.ndarray]] = pickle.load(f)
    all_genes = [g for g, v in gene_vectors.items() if len(v) > 0]
    all_genes_set = set(all_genes)

    # Gold genes baseado no OpenTargets EVA >= 0.5
    gold_ot = load_ot_target(OT_JSON, VALIDATION_SOURCE)
    gold_ot = gold_ot - set([g.strip().upper() for g in VALIDATION_GENES])  # remove VALIDATION_GENES
    gold_available = [g for g in gold_ot if g in all_genes_set]
    print(f"[info] Gold genes EVA >=0.5: {len(gold_available)}")

    gold_arr = np.array(gold_available)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    auc_results = {}

    for bottleneck_dim in BOTTLENECK_LIST:
        print(f"\n=== TREINANDO BOTTLENECK {bottleneck_dim} ===")
        gold_oof_probs = {}
        cv_folds = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(gold_arr)):
            gold_train = gold_arr[tr_idx].tolist()
            gold_val = gold_arr[va_idx].tolist()
            gold_val_set = set(gold_val)

            model = train_model(gene_vectors, gold_train, gold_val_set, device, bottleneck_dim, SEED + fold)

            # Infer all gold val genes + negatives
            scores = {}
            for g in gold_val:
                logit = infer_logit_simple(model, gene_vectors[g], device)
                scores[g] = sigmoid(logit)
            neg_for_auc = getattr(model, "last_negatives", [])
            for g in neg_for_auc:
                if g in gene_vectors:
                    scores[g] = sigmoid(infer_logit_simple(model, gene_vectors[g], device))

            y_true = [1] * len(gold_val) + [0] * len(neg_for_auc)
            y_score = [scores[g] for g in gold_val + neg_for_auc]

            if len(set(y_true)) == 2:
                auc = float(average_precision_score(y_true, y_score))
            else:
                auc = float("nan")
            cv_folds.append(auc)

            del model
            torch.cuda.empty_cache()

        mean_auc = float(np.mean(cv_folds))
        std_auc = float(np.std(cv_folds, ddof=1)) if len(cv_folds) > 1 else 0.0
        print(f"[Bottleneck={bottleneck_dim}] CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        auc_results[bottleneck_dim if bottleneck_dim is not None else "None"] = mean_auc

    # ==========================
    # PLOT AUC POR BOTTLENECK
    # ==========================
    plt.figure(figsize=(8, 5))
    keys = list(auc_results.keys())
    values = [auc_results[k] for k in keys]
    plt.plot(keys, values, marker='o')
    plt.xlabel("Bottleneck Dim")
    plt.ylabel("PR-AUC")
    plt.title(f"Cross-Validated PR-AUC vs Bottleneck ({VALIDATION_SOURCE})")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, f"auc_vs_bottleneck_{VALIDATION_SOURCE}.png"), dpi=300)
    plt.show()
    print(f"[info] Gráfico salvo em: {OUT_DIR}auc_vs_bottleneck_{VALIDATION_SOURCE}.png")


if __name__ == "__main__":
    main()

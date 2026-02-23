import os
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

from config import VALIDATION_GENES


# =========================
# CONFIG
# =========================
umbrella_term = "neurodegenerative_disease"
MODEL_NAME = "pubmedbert"

FEATURES_PATH = f"./features_{MODEL_NAME}_{umbrella_term}/features_ALS_{MODEL_NAME}.pkl"
OUT_DIR = f"./dimensionality_analysis_{MODEL_NAME}_{umbrella_term}/"

ORIGINAL_INPUT_DIM = 1536

# começa em 1536 e vai dividindo por 2 até 8
DIMENSIONS_TO_TEST = []
d = ORIGINAL_INPUT_DIM
while d >= 8:
    DIMENSIONS_TO_TEST.append(d)
    d //= 2

# Experimento
N_SPLITS = 10
TEST_SIZE = 0.2
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-3
SEED = 42


# =========================
# MODEL
# =========================
class BottleneckLogistic(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()

        if target_dim < input_dim:
            self.bottleneck = nn.Linear(input_dim, target_dim, bias=False)
            self.classifier = nn.Linear(target_dim, 1)
        else:
            self.bottleneck = nn.Identity()
            self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        logit = self.classifier(z).view(-1)
        return logit


# =========================
# UTILS
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data(features_path: str):
    """
    Carrega pickle (dict: gene -> list[vectors 1536]) e aplica mean pooling
    para virar 1 vetor por gene.
    """
    print(f"[info] Loading features: {features_path}")
    with open(features_path, "rb") as f:
        gene_vectors = pickle.load(f)

    gold_genes = set([str(g).strip().upper() for g in VALIDATION_GENES])

    # positivos
    pos_genes = [g for g in gene_vectors if g in gold_genes and len(gene_vectors[g]) > 0]

    # negativos
    neg_genes_all = [g for g in gene_vectors if g not in gold_genes and len(gene_vectors[g]) > 0]

    # 4x negativos
    np.random.seed(SEED)
    neg_sample_size = min(len(neg_genes_all), len(pos_genes) * 4)
    neg_genes = np.random.choice(neg_genes_all, size=neg_sample_size, replace=False)

    final_genes = pos_genes + list(neg_genes)

    X_list, y_list = [], []

    for gene in final_genes:
        vecs = gene_vectors[gene]  # list[np.array(1536)]
        mean_vec = np.mean(np.stack(vecs), axis=0)

        label = 1.0 if gene in gold_genes else 0.0
        X_list.append(mean_vec)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


def train_eval_loop(X, y, target_dim, device):
    cv = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)
    aucs = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        t_X_train = torch.tensor(X_train, device=device)
        t_y_train = torch.tensor(y_train, device=device)
        t_X_test = torch.tensor(X_test, device=device)

        model = BottleneckLogistic(ORIGINAL_INPUT_DIM, target_dim).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()

        # treino
        model.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            logits = model(t_X_train)
            loss = criterion(logits, t_y_train)
            loss.backward()
            optimizer.step()

        # avaliação
        model.eval()
        with torch.no_grad():
            test_logits = model(t_X_test)
            test_probs = torch.sigmoid(test_logits).detach().cpu().numpy()

        try:
            auc = roc_auc_score(y_test, test_probs)
            aucs.append(auc)
        except ValueError:
            # acontece se y_test tiver só 0 ou só 1 no split
            pass

    if len(aucs) == 0:
        return float("nan"), float("nan")

    return float(np.mean(aucs)), float(np.std(aucs))


# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features not found: {FEATURES_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")

    X, y = prepare_data(FEATURES_PATH)
    print(f"[info] X shape: {X.shape} | y shape: {y.shape}")
    print(f"[info] Testing dimensions: {DIMENSIONS_TO_TEST}")

    results = []

    print("\n=== DIMENSIONALITY LOOP ===")
    for dim in tqdm(DIMENSIONS_TO_TEST, desc="Testing dims"):
        mean_auc, std_auc = train_eval_loop(X, y, dim, device)

        print(f"Dim: {dim:4d} | AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        results.append(
            {
                "dim": dim,
                "auc_mean": mean_auc,
                "auc_std": std_auc,
                "n_splits": N_SPLITS,
                "test_size": TEST_SIZE,
                "epochs": EPOCHS,
            }
        )

    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, "dimensionality_results.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n[info] Saved results: {out_csv}")


if __name__ == "__main__":
    main()

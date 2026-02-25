import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns


FEATURES_WITH_META_PKL = "./features_TEST_pubmedbert/features_ALS_1970_2026.pkl"
CORPUS_CSV_PATH = "../data/corpus_als_general_pmc_preprocessed3.csv"
TEXT_COL = "text"
YEAR_COL = "year"


START_YEAR = 1970
END_YEAR = 2026

MODEL_CKPT = "./scores_mil_pubmedbert/model_final.pt"

INPUT_DIM = 2305
HIDDEN_DIM = 256
ATTENTION_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)  # logit

    def forward(self, bag_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.feature_extractor(bag_features)                      # (N, H)
        a = self.attention_weights(self.attention_V(h) * self.attention_U(h)).squeeze(1)  # (N,)
        A = torch.softmax(a, dim=0)                                   # (N,)
        bag_rep = torch.sum(A.unsqueeze(1) * h, dim=0)                # (H,)
        logit = self.classifier(bag_rep).view(1)
        return logit, A


def highlight_gene(text: str, gene: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").strip()
    return re.sub(
        rf"\b({re.escape(gene)})\b",
        r"\\textbf{\1}",
        text,
        flags=re.IGNORECASE,
    )

def sanity_checks(text: str, gene: str):
    has_gene = re.search(rf"\b{re.escape(gene)}\b", text, flags=re.IGNORECASE) is not None
    has_dis = ("als_disease_token" in text.lower())
    return has_gene, has_dis

def ascii_bar(value_0_1: float, width: int = 28) -> str:
    value_0_1 = float(np.clip(value_0_1, 0.0, 1.0))
    filled = int(round(value_0_1 * width))
    return "█" * filled + "░" * (width - filled)

def weight_scales(A: np.ndarray, eps: float = 1e-12):
    """
    Returns:
      - rel_to_max: A / max(A)
      - logit_norm: normalized log(A) between min_nonzero and max
    """
    A = A.astype(np.float64)
    amax = float(A.max() + eps)
    rel_to_max = A / amax

    nonzero = A[A > eps]
    if len(nonzero) == 0:
        logit_norm = np.zeros_like(A, dtype=np.float64)
        return rel_to_max, logit_norm

    lo = float(nonzero.min())
    hi = float(nonzero.max())
    # log scale to expand tiny probabilities
    logA = np.log(np.clip(A, lo, hi))
    log_lo = float(np.log(lo))
    log_hi = float(np.log(hi))
    denom = (log_hi - log_lo) if (log_hi > log_lo) else 1.0
    logit_norm = (logA - log_lo) / denom
    logit_norm = np.clip(logit_norm, 0.0, 1.0)

    return rel_to_max, logit_norm

def format_weight(A: np.ndarray, i: int, rel_to_max: np.ndarray, logit_norm: np.ndarray) -> str:
    w = float(A[i])
    pct = 100.0 * w
    rel = float(rel_to_max[i])
    lgn = float(logit_norm[i])
    bar_rel = ascii_bar(rel, width=18)
    bar_log = ascii_bar(lgn, width=18)
    return (
        f"raw={w:.6f} | pct={pct:.3f}% | "
        f"rel={rel:.3f} {bar_rel} | "
        f"log={lgn:.3f} {bar_log}"
    )


def plot_attention_distribution(A, gene_name, output_path="attention_dist.png"):
    """
    histogram and decay curve plots
    """
   
    weights = np.sort(A)[::-1] 
    log_weights = np.log10(weights + 1e-12) 

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # histogram
    sns.histplot(log_weights, bins=50, ax=axes[0], color="#2c3e50", kde=True)
    axes[0].set_title(f"Distribution of Attention Weights (Log Scale) - {gene_name}")
    axes[0].set_xlabel("Log10(Attention Weight)")
    axes[0].set_ylabel("Number of Articles")
    axes[0].axvline(x=np.log10(1/len(A)), color='r', linestyle='--', label="Uniform Attention (Baseline)")
    axes[0].legend()

    # cumulative mass
    cumsum = np.cumsum(weights)
    axes[1].plot(np.arange(len(weights)), cumsum, color="#e74c3c", linewidth=2)
    axes[1].set_title(f"Cumulative Attention Mass - {gene_name}")
    axes[1].set_xlabel("Number of Articles (Sorted by Importance)")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].axhline(y=0.9, color='k', linestyle=':', label="90% Mass")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[info] Plot saved to {output_path}")



@torch.no_grad()
def main(
    gene: str = "sod1",
    top_k: int = 2,
    low_nonzero_k: int = 2,
    eps_zero: float = 1e-12,
    show_full_abstract: bool = True,
    max_chars: int = 0,  # 0 = no limit; else truncate for terminal
):
    gene = gene.strip().upper()

    print("[info] loading features w/ meta...")
    with open(FEATURES_WITH_META_PKL, "rb") as f:
        bags = pickle.load(f)

    if gene not in bags or len(bags[gene]) == 0:
        raise ValueError(f"Gene {gene} not found or empty bag.")

    print("[info] loading corpus csv...")
    df = pd.read_csv(CORPUS_CSV_PATH, escapechar="\\")
    print(f"[info] Filtering years {START_YEAR}-{END_YEAR} to match extraction ID...")
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df = df[(df[YEAR_COL] >= START_YEAR) & (df[YEAR_COL] <= END_YEAR)].copy()
    df = df.reset_index(drop=True)

    inst = bags[gene]
    if not isinstance(inst[0], dict) or ("feat" not in inst[0]) or ("row_id" not in inst[0]):
        raise ValueError("Expected bag format: list of dicts with keys {'feat','row_id'}")

    feats = np.stack([x["feat"] for x in inst], axis=0).astype(np.float32)
    row_ids = np.array([int(x["row_id"]) for x in inst], dtype=np.int64)

    print(f"[info] gene={gene} | instances={len(inst)} | feat_dim={feats.shape[1]}")

    model = GatedAttentionMIL(INPUT_DIM, HIDDEN_DIM, ATTENTION_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))
    model.eval()

    bag = torch.as_tensor(feats, device=DEVICE)
    logit, A = model(bag)
    A = A.detach().cpu().numpy().astype(np.float64)

    score = 1.0 / (1.0 + np.exp(-float(logit.item())))
    print(f"[info] model score for {gene}: {score:.4f}")

    # Diagnostics: concentration + near-zero counts
    n = len(A)
    n_zero = int(np.sum(A <= eps_zero))
    n_nonzero = n - n_zero
    print(f"[info] attention near-zero (<= {eps_zero:.0e}): {n_zero}/{n} ({100*n_zero/n:.2f}%) | nonzero: {n_nonzero}")

    A_sorted = np.sort(A)[::-1]
    top1 = float(A_sorted[0]) if n > 0 else 0.0
    top10 = float(A_sorted[:10].sum()) if n >= 10 else float(A_sorted.sum())
    top50 = float(A_sorted[:50].sum()) if n >= 50 else float(A_sorted.sum())
    print(f"[info] attention mass | top1={100*top1:.2f}% | top10={100*top10:.2f}% | top50={100*top50:.2f}%")
    
    plot_attention_distribution(A, gene)

    rel_to_max, logit_norm = weight_scales(A, eps=eps_zero)

    order_desc = np.argsort(-A)

    # lowest but nonzero: pick from indices where A > eps
    nz_idx = np.where(A > eps_zero)[0]
    if len(nz_idx) == 0:
        order_low_nz = np.array([], dtype=int)
    else:
        order_low_nz = nz_idx[np.argsort(A[nz_idx])]  # ascending among nonzero

    def get_text(rid: int) -> str:
        if 0 <= rid < len(df):
            return str(df.loc[rid, TEXT_COL])
        return "[ERROR: Index out of bounds]"

    def maybe_truncate(s: str) -> str:
        if not show_full_abstract:
            return s[:500] + ("..." if len(s) > 500 else "")
        if max_chars and len(s) > max_chars:
            return s[:max_chars] + "..."
        return s


    # TOP-K
    print("\nTOP ARTICLES (HIGHEST ATTENTION)")
    for k in range(min(top_k, len(order_desc))):
        i = int(order_desc[k])
        rid = int(row_ids[i])
        raw_text = get_text(rid)
        text = highlight_gene(raw_text, gene)

        has_gene, has_dis = sanity_checks(raw_text, gene)
        if not has_gene or not has_dis:
            print(f"[WARN] row_id={rid} mismatch? GeneInText={has_gene} DisInText={has_dis}")

        print(f"\n[Top #{k+1}] idx={i} | row_id={rid} | {format_weight(A, i, rel_to_max, logit_norm)}")
        print(maybe_truncate(text))

  
    print("\nLOW ARTICLES (LOW ATTENTION, NONZERO)")
    if len(order_low_nz) == 0:
        print("[info] No nonzero attention instances found (all <= eps).")
    else:
        for k in range(min(low_nonzero_k, len(order_low_nz))):
            i = int(order_low_nz[k])
            rid = int(row_ids[i])
            raw_text = get_text(rid)
            text = highlight_gene(raw_text, gene)

            has_gene, has_dis = sanity_checks(raw_text, gene)
            if not has_gene or not has_dis:
                print(f"[WARN] row_id={rid} mismatch? GeneInText={has_gene} DisInText={has_dis}")

            print(f"\n[Low(nonzero) #{k+1}] idx={i} | row_id={rid} | {format_weight(A, i, rel_to_max, logit_norm)}")
            print(maybe_truncate(text))

    print("\n(INFO) SMALLEST ATTENTION (MAY BE ZERO)")
    order_asc = np.argsort(A)
    for k in range(min(2, len(order_asc))):
        i = int(order_asc[k])
        rid = int(row_ids[i])
        print(f"[min #{k+1}] idx={i} row_id={rid} raw={A[i]:.12f}")





if __name__ == "__main__":
    main(
        gene="sod1",
        top_k=2,
        low_nonzero_k=2,
        eps_zero=1e-12,
        show_full_abstract=True,
        max_chars=0,  
    )

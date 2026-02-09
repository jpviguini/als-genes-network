import numpy as np
import matplotlib.pyplot as plt


SCORES_NPZ = "./scores_word2vec/scores_word2vec_allgenes.npz"
OUTPUT_FIG = "./freq_vs_score_scatter2.pdf"

try:
    from config import VALIDATION_GENES
    GOLD_SET = {g.strip().upper() for g in VALIDATION_GENES}
except Exception:
    GOLD_SET = None


def main():
    print(f"[info] Loading scores from {SCORES_NPZ}")
    data = np.load(SCORES_NPZ, allow_pickle=True)

    genes = data["genes"]
    scores = data["scores_topm"]  
    counts = data["ctx_counts"]   

    genes = np.array([g.upper() for g in genes])
    scores = scores.astype(float)
    counts = counts.astype(int)

    print(f"[info] Loaded {len(genes)} genes")

    if GOLD_SET is not None:
        is_gold = np.array([g in GOLD_SET for g in genes])
    else:
        is_gold = np.zeros(len(genes), dtype=bool)


    plt.figure(figsize=(7, 5))

    plt.scatter(
        counts[~is_gold],
        scores[~is_gold],
        alpha=0.35,
        s=18,
        label="Non-gold genes"
    )

    # Gold
    if is_gold.any():
        plt.scatter(
            counts[is_gold],
            scores[is_gold],
            alpha=0.9,
            s=40,
            marker="x",
            label="Gold genes"
        )

    plt.xscale("log")
    plt.xlabel("Number of articles mentioning the gene (log scale)")
    plt.ylabel("Gene–disease association score (MIL)")
    plt.title("Impact of mention frequency vs MIL score")

    plt.legend(frameon=False)
    plt.tight_layout()

    print(f"[info] Saving figure to {OUTPUT_FIG}")
    plt.savefig(OUTPUT_FIG, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

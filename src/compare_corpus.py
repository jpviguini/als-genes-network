import numpy as np
import pandas as pd
from scipy.stats import spearmanr

UMBRELLAS = [
    "neurodegenerative_disease",  # broad
    "neuromuscular_disease",      # broad
    "motor_neuron_disease",       # specific
]

MODEL = "pubmedbert"

NPZ_MAP = {
    u: f"./logits_MIL_{MODEL}_{u}/scores_final_allgenes.npz"
    for u in UMBRELLAS
}

TOPK_LIST = [10, 20, 50, 100, 200]


def load_npz(npz_path: str, tag: str) -> pd.DataFrame:
    d = np.load(npz_path, allow_pickle=True)
    df = pd.DataFrame({
        "gene": d["genes"].astype(str),
        tag: d["scores_topm"].astype(float),
        f"{tag}_ctx": d["ctx_counts"].astype(int),
    })
    df["gene"] = df["gene"].str.strip().str.upper()
    return df


def topk_overlap(df: pd.DataFrame, col_a: str, col_b: str, k: int) -> tuple[int, float]:
    top_a = set(df.sort_values(col_a, ascending=False).head(k)["gene"])
    top_b = set(df.sort_values(col_b, ascending=False).head(k)["gene"])
    inter = len(top_a & top_b)
    return inter, inter / k


def pairwise_report(a: str, b: str, dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    col_a = f"mil_{a}"
    col_b = f"mil_{b}"

    merged = dfs[a].merge(dfs[b], on="gene", how="inner")
    print(f"\n=== {a} vs {b} ===")
    print(f"[info] overlap genes: {len(merged):,}")

    rho, p = spearmanr(merged[col_a].values, merged[col_b].values)
    print(f"Spearman rho = {rho:.4f}  (p={p:.3e})")

    print("Top-K overlap:")
    for k in TOPK_LIST:
        inter, frac = topk_overlap(merged, col_a, col_b, k)
        print(f"  Top-{k:3d}: {inter:3d}/{k} ({100*frac:.1f}%)")

    return merged


def report_specific_only(
    merged: pd.DataFrame,
    specific: str,
    general: str,
    top_n: int = 20,
    min_ctx_specific: int = 5,
    general_rank_floor_pct: float = 0.50,  # bottom 50% in general
):
    col_s = f"mil_{specific}"
    col_g = f"mil_{general}"
    ctx_s = f"mil_{specific}_ctx"

    df = merged.copy()

    # ranks (1 = best)
    df[f"rank_{specific}"] = df[col_s].rank(ascending=False, method="average")
    df[f"rank_{general}"] = df[col_g].rank(ascending=False, method="average")

    # percentile ranks (0 = best, 1 = worst) for readability
    df[f"pct_{specific}"] = df[f"rank_{specific}"] / len(df)
    df[f"pct_{general}"] = df[f"rank_{general}"] / len(df)

    # "specific-only": top in specific AND low in general
    cand = df[
        (df[ctx_s] >= min_ctx_specific) &
        (df[f"pct_{specific}"] <= 0.05) &            # top 5% in specific
        (df[f"pct_{general}"] >= general_rank_floor_pct)  # bottom X% in general
    ].copy()

    cand = cand.sort_values([col_s, col_g], ascending=[False, True]).head(top_n)

    print(f"\n--- Specific-only candidates: {specific} high, {general} low ---")
    if len(cand) == 0:
        print("[info] none found with current thresholds.")
        return

    show_cols = ["gene", col_s, col_g, ctx_s, f"pct_{specific}", f"pct_{general}"]
    print(cand[show_cols].to_string(index=False, justify="left", col_space=12))


def main():
    # load all
    dfs = {}
    for u, path in NPZ_MAP.items():
        dfs[u] = load_npz(path, tag=f"mil_{u}")
        print(f"[info] loaded {u}: {len(dfs[u]):,} genes")

    # pairwise comparisons
    # broad vs broad
    m_nd_nm = pairwise_report("neurodegenerative_disease", "neuromuscular_disease", dfs)

    # specific vs broad
    m_mnd_nd = pairwise_report("motor_neuron_disease", "neurodegenerative_disease", dfs)
    m_mnd_nm = pairwise_report("motor_neuron_disease", "neuromuscular_disease", dfs)

    # specific-only listings (motor neuron disease vs each broad)
    report_specific_only(m_mnd_nd, specific="motor_neuron_disease", general="neurodegenerative_disease")
    report_specific_only(m_mnd_nm, specific="motor_neuron_disease", general="neuromuscular_disease")


if __name__ == "__main__":
    main()

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

umbrella_term = "motor_neuron_disease"

MIL_NPZ = f"./logits_MIL_pubmedbert_{umbrella_term}/scores_final_allgenes.npz"
OT_JSON = "OT-MONDO_0004976-associated-targets-2_10_2026-v25_12.json"


def load_mil(npz_path: str) -> pd.DataFrame:
    d = np.load(npz_path, allow_pickle=True)
    df = pd.DataFrame({
        "gene": d["genes"].astype(str),
        "mil_score": d["scores_topm"].astype(float),
        "ctx_count": d["ctx_counts"].astype(int),
    })
    df["gene"] = df["gene"].str.strip().str.upper()
    return df


def load_ot(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    needed = {"symbol", "globalScore", "europepmc"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"JSON não tem as chaves esperadas: faltando {missing}. Tem: {df.columns.tolist()}")

    df = df[["symbol", "globalScore", "europepmc"]].copy()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    # globalScore: numérico
    df["globalScore"] = pd.to_numeric(df["globalScore"], errors="coerce")

    # europepmc: pode ser número ou "No data"
    # pd.to_numeric transforma "No data" -> NaN automaticamente com errors="coerce"
    df["europepmc"] = pd.to_numeric(df["europepmc"], errors="coerce")

    df = df.dropna(subset=["symbol", "globalScore"])  # exige globalScore
    df = df.rename(columns={
        "symbol": "gene",
        "globalScore": "ot_globalScore",
        "europepmc": "ot_europepmc",
    })

    # se houver duplicatas por gene: pega o maior score em cada coluna
    df = df.groupby("gene", as_index=False).agg({
        "ot_globalScore": "max",
        "ot_europepmc": "max",
    })
    return df


def topk_overlap(df: pd.DataFrame, k: int, ot_col: str) -> tuple[float, int]:
    top_mil = set(df.sort_values("mil_score", ascending=False).head(k)["gene"])
    top_ot = set(df.sort_values(ot_col, ascending=False).head(k)["gene"])
    inter = len(top_mil & top_ot)
    return inter / k, inter


def report_compare(merged: pd.DataFrame, ot_col: str, label: str):
    sub = merged.dropna(subset=[ot_col]).copy()
    print(f"\n=== {label} ===")
    print(f"[info] Overlap (genes with {label}): {len(sub):,}")

    if len(sub) < 10:
        print("[warn] overlap muito pequeno; Spearman fica instável.")
        return

    rho, p = spearmanr(sub["mil_score"].values, sub[ot_col].values)
    print(f"Spearman rho = {rho:.4f}  (p={p:.3e})")

    print(f"Top-K Overlap (MIL vs {label}):")
    for k in [10, 20, 50, 100, 200]:
        frac, inter = topk_overlap(sub, k, ot_col=ot_col)
        print(f"  Top-{k:3d}: overlap = {inter:3d}/{k}  ({100*frac:.1f}%)")

    print("\nTop 15 MIL genes (with OT cols):")
    print(sub.sort_values("mil_score", ascending=False)
            .head(15)[["gene", "mil_score", "ot_globalScore", "ot_europepmc"]])

    print(f"\nTop 15 {label} genes:")
    print(sub.sort_values(ot_col, ascending=False)
            .head(15)[["gene", "mil_score", "ot_globalScore", "ot_europepmc"]])


def main():
    mil = load_mil(MIL_NPZ)
    ot = load_ot(OT_JSON)

    merged = mil.merge(ot, on="gene", how="inner")

    print(f"[info] MIL genes: {len(mil):,}")
    print(f"[info] OT genes:  {len(ot):,}")
    print(f"[info] Overlap:   {len(merged):,}")

    # 1) Comparação com globalScore
    report_compare(merged, ot_col="ot_globalScore", label="OpenTargets globalScore")

    # 2) Comparação com europepmc (literature-only)
    report_compare(merged, ot_col="ot_europepmc", label="OpenTargets europepmc")


if __name__ == "__main__":
    main()

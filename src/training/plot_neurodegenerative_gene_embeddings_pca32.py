#!/usr/bin/env python3
"""
Generate a 2D scatter plot from neurodegenerative-disease gene embeddings.

Pipeline
1) Load gene embeddings from the PubMedBERT pickle file.
2) Build one embedding vector per gene (mean across vectors when needed).
3) Standardize vectors and run PCA to 32 dimensions.
4) Plot the first two dimensions of the 32D representation (PC1 vs PC2).
5) Save figure and coordinate table.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


INPUT_PICKLE = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/features/"
    "featuresUPPER_pubmedbert_neurodegenerative_disease/features_ALS_pubmedbert.pkl"
)

OUTPUT_DIR = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/results/embedding_pca32_neurodegenerative_disease"
)
OUTPUT_PNG = OUTPUT_DIR / "gene_embeddings_pca32_pc1_pc2_scatter.png"
OUTPUT_COORDS_CSV = OUTPUT_DIR / "gene_embeddings_pca32_pc_coords.csv"
OUTPUT_INFO_TXT = OUTPUT_DIR / "run_summary.txt"

TARGET_PCA_DIM = 32


def _to_1d_float_array(value: object) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # If a matrix appears, use mean across rows as a single vector.
        return np.nanmean(arr, axis=0)
    return None


def _candidate_vectors(value: object) -> List[np.ndarray]:
    vectors: List[np.ndarray] = []

    if isinstance(value, (list, tuple)):
        for item in value:
            vec = _to_1d_float_array(item)
            if vec is not None:
                vectors.append(vec)
        return vectors

    if isinstance(value, dict):
        for key in ("embedding", "embeddings", "vector", "vectors", "value", "values"):
            if key in value:
                return _candidate_vectors(value[key])
        return vectors

    vec = _to_1d_float_array(value)
    if vec is not None:
        vectors.append(vec)
    return vectors


def build_gene_embedding_table(raw_obj: object) -> pd.DataFrame:
    if not isinstance(raw_obj, dict):
        raise ValueError(f"Expected dictionary-like pickle structure, got {type(raw_obj)}")

    rows: List[Dict[str, object]] = []
    skipped = 0

    for gene, payload in raw_obj.items():
        vectors = _candidate_vectors(payload)
        if not vectors:
            skipped += 1
            continue

        lengths = [v.shape[0] for v in vectors]
        target_len = max(set(lengths), key=lengths.count)
        vectors = [v for v in vectors if v.shape[0] == target_len]
        if not vectors:
            skipped += 1
            continue

        mean_vec = np.nanmean(np.vstack(vectors), axis=0)
        if not np.all(np.isfinite(mean_vec)):
            mean_vec = np.nan_to_num(mean_vec, nan=0.0, posinf=0.0, neginf=0.0)

        row = {"gene_symbol": str(gene), "n_source_vectors": int(len(vectors))}
        for i, val in enumerate(mean_vec):
            row[f"emb_{i:04d}"] = float(val)
        rows.append(row)

    if not rows:
        raise ValueError("No valid gene embeddings were extracted from input pickle.")

    df = pd.DataFrame(rows)
    df = df.sort_values("gene_symbol", kind="stable").reset_index(drop=True)
    df.attrs["skipped_genes"] = skipped
    return df


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_PICKLE.open("rb") as f:
        raw_obj = pickle.load(f)

    gene_df = build_gene_embedding_table(raw_obj)
    emb_cols = [c for c in gene_df.columns if c.startswith("emb_")]
    x = gene_df[emb_cols].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    n_components = min(TARGET_PCA_DIM, x_scaled.shape[0], x_scaled.shape[1])
    if n_components < 2:
        raise ValueError(
            f"PCA needs at least 2 components for plotting; got n_components={n_components}."
        )

    pca = PCA(n_components=n_components, random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    coords_df = pd.DataFrame(
        {
            "gene_symbol": gene_df["gene_symbol"].astype(str),
            "n_source_vectors": gene_df["n_source_vectors"].astype(int),
            "pc1": x_pca[:, 0],
            "pc2": x_pca[:, 1],
        }
    )

    for i in range(n_components):
        coords_df[f"pca32_dim_{i+1:02d}"] = x_pca[:, i]
    coords_df.to_csv(OUTPUT_COORDS_CSV, index=False)

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    ax.scatter(coords_df["pc1"], coords_df["pc2"], s=12, alpha=0.75, linewidths=0)
    ax.set_title("Neurodegenerative-disease Gene Embeddings (PCA32 -> PC1 vs PC2)")
    ax.set_xlabel(
        f"PC1 of 32D embedding ({100.0 * pca.explained_variance_ratio_[0]:.2f}% variance)"
    )
    ax.set_ylabel(
        f"PC2 of 32D embedding ({100.0 * pca.explained_variance_ratio_[1]:.2f}% variance)"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=160)
    plt.close(fig)

    summary = [
        "Neurodegenerative-disease embedding PCA run",
        f"input_pickle={INPUT_PICKLE}",
        f"output_png={OUTPUT_PNG}",
        f"output_coords_csv={OUTPUT_COORDS_CSV}",
        f"genes_with_embedding={len(coords_df)}",
        f"skipped_genes={int(gene_df.attrs.get('skipped_genes', 0))}",
        f"original_embedding_dim={x.shape[1]}",
        f"pca_dim_used={n_components}",
        f"explained_variance_pc1={pca.explained_variance_ratio_[0]:.8f}",
        f"explained_variance_pc2={pca.explained_variance_ratio_[1]:.8f}",
        f"explained_variance_first2_sum={pca.explained_variance_ratio_[:2].sum():.8f}",
    ]
    OUTPUT_INFO_TXT.write_text("\n".join(summary), encoding="utf-8")

    print(f"[ok] genes_with_embedding={len(coords_df)}")
    print(f"[ok] original_embedding_dim={x.shape[1]}")
    print(f"[ok] pca_dim_used={n_components}")
    print(f"[ok] plot_saved={OUTPUT_PNG}")
    print(f"[ok] coords_saved={OUTPUT_COORDS_CSV}")


if __name__ == "__main__":
    run()


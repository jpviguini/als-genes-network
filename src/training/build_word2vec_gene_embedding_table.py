#!/usr/bin/env python3
"""Build one Word2Vec embedding per gene (clean standalone script)."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


DEFAULT_W2V_MODEL = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/word2vec_tmp/models/word2vec/"
    "word2vec_neurodegenerative_disease.model"
)
DEFAULT_GENE_UNIVERSE_CSV = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/data/corpus/extracted_genes/genes_extracted_neurodegenerative_disease.csv"
)
DEFAULT_CANDIDATE_TABLE = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/"
    "GCST90027164_cs_gene_candidate_feature_table_neurodegenerative_disease.csv"
)
DEFAULT_OUT_DIR = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/word2vec_tmp/embeddings"
)
DEFAULT_GENE_COL = "gene"
DEFAULT_CANDIDATE_GENE_COL = "gene_symbol"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one Word2Vec embedding vector per gene.")
    parser.add_argument("--word2vec-model", type=Path, default=DEFAULT_W2V_MODEL)
    parser.add_argument("--gene-universe-csv", type=Path, default=DEFAULT_GENE_UNIVERSE_CSV)
    parser.add_argument("--gene-col", type=str, default=DEFAULT_GENE_COL)
    parser.add_argument("--candidate-table", type=Path, default=DEFAULT_CANDIDATE_TABLE)
    parser.add_argument("--candidate-gene-col", type=str, default=DEFAULT_CANDIDATE_GENE_COL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def normalize_gene_symbol(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().upper()
    return s if s else None


def load_keyed_vectors(model_path: Path):
    from gensim.models import KeyedVectors, Word2Vec

    try:
        model = Word2Vec.load(str(model_path))
        return model.wv
    except Exception:
        pass

    try:
        return KeyedVectors.load(str(model_path))
    except Exception as exc:
        raise ValueError(f"Could not load Word2Vec model from {model_path}") from exc


def resolve_vocab_key(wv, gene_symbol: str) -> Optional[str]:
    candidates = [gene_symbol.lower(), gene_symbol, gene_symbol.upper()]
    seen: Set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        if key in wv:
            return key
    return None


def get_vec_count(wv, key: str) -> int:
    try:
        return max(1, int(wv.get_vecattr(key, "count")))
    except Exception:
        pass

    try:
        return max(1, int(wv.vocab[key].count))
    except Exception:
        return 1


def load_gene_universe(path: Path, gene_col: str) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Gene universe CSV not found: {path}")
    df = pd.read_csv(path)
    if gene_col not in df.columns:
        raise ValueError(f"Column '{gene_col}' not found in gene universe CSV: {path}")

    genes = set()
    for raw in df[gene_col].tolist():
        g = normalize_gene_symbol(raw)
        if g is not None:
            genes.add(g)
    return genes


def load_candidate_gene_set(path: Path, gene_col: str) -> Set[str]:
    if not path.exists():
        return set()

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, usecols=[gene_col])
    elif suffix == ".parquet":
        df = pd.read_parquet(path, columns=[gene_col])
    else:
        raise ValueError(f"Unsupported candidate table format: {path}")

    out: Set[str] = set()
    for raw in df[gene_col].tolist():
        g = normalize_gene_symbol(raw)
        if g is not None:
            out.add(g)
    return out


def build_embeddings(
    wv,
    genes: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rows: List[Dict[str, object]] = []
    embed_map: Dict[str, np.ndarray] = {}

    for gene in sorted(genes):
        key = resolve_vocab_key(wv, gene)
        if key is None:
            rows.append(
                {
                    "gene_symbol": gene,
                    "has_word2vec_embedding": 0,
                    "word2vec_token": "",
                    "word2vec_token_count": 0,
                }
            )
            continue

        vec = np.asarray(wv[key], dtype=np.float32)
        embed_map[gene] = vec
        rows.append(
            {
                "gene_symbol": gene,
                "has_word2vec_embedding": 1,
                "word2vec_token": str(key),
                "word2vec_token_count": int(get_vec_count(wv, key)),
            }
        )

    return pd.DataFrame(rows), embed_map


def save_embedding_artifacts(
    out_dir: Path,
    embedding_table: pd.DataFrame,
    embed_map: Dict[str, np.ndarray],
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = out_dir / "word2vec_gene_embedding_table.csv"
    pkl_path = out_dir / "word2vec_gene_embeddings.pkl"
    npz_path = out_dir / "word2vec_gene_embeddings.npz"

    embedding_table.to_csv(table_path, index=False)

    with pkl_path.open("wb") as f:
        pickle.dump(embed_map, f)

    genes = np.array(sorted(embed_map.keys()), dtype=np.str_)
    if genes.size == 0:
        matrix = np.zeros((0, 0), dtype=np.float32)
    else:
        matrix = np.vstack([embed_map[g] for g in genes]).astype(np.float32)
    np.savez_compressed(npz_path, genes=genes, embeddings=matrix)

    return {
        "table_path": table_path,
        "pickle_path": pkl_path,
        "npz_path": npz_path,
    }


def main() -> None:
    args = parse_args()

    if not args.word2vec_model.exists():
        raise FileNotFoundError(f"Word2Vec model not found: {args.word2vec_model}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Loading Word2Vec model: {args.word2vec_model}")
    wv = load_keyed_vectors(args.word2vec_model)

    gene_universe = load_gene_universe(args.gene_universe_csv, args.gene_col)
    candidate_genes = load_candidate_gene_set(args.candidate_table, args.candidate_gene_col)

    combined = set(gene_universe)
    combined.update(candidate_genes)

    print(f"[info] Universe genes: {len(gene_universe)}")
    print(f"[info] Candidate-table genes: {len(candidate_genes)}")
    print(f"[info] Combined genes to score: {len(combined)}")

    emb_table, embed_map = build_embeddings(wv, sorted(combined))

    artifacts = save_embedding_artifacts(out_dir, emb_table, embed_map)

    embedding_dim = 0
    if embed_map:
        embedding_dim = int(len(next(iter(embed_map.values()))))

    candidate_cov = 0
    if candidate_genes:
        candidate_cov = int(sum(g in embed_map for g in candidate_genes))

    summary = {
        "word2vec_model": str(args.word2vec_model),
        "gene_universe_csv": str(args.gene_universe_csv),
        "candidate_table": str(args.candidate_table),
        "gene_counts": {
            "universe": int(len(gene_universe)),
            "candidate_table_unique": int(len(candidate_genes)),
            "combined": int(len(combined)),
            "with_word2vec_embedding": int(len(embed_map)),
            "without_word2vec_embedding": int(len(combined) - len(embed_map)),
            "candidate_table_with_word2vec_embedding": int(candidate_cov),
            "candidate_table_without_word2vec_embedding": int(len(candidate_genes) - candidate_cov),
        },
        "embedding_dimension": int(embedding_dim),
        "single_embedding_per_gene_method": (
            "Single static Word2Vec token vector per normalized gene symbol (no per-document averaging)."
        ),
        "outputs": {k: str(v) for k, v in artifacts.items()},
    }

    summary_path = out_dir / "word2vec_gene_embedding_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[info] Genes with embeddings: {len(embed_map)} / {len(combined)}")
    print(f"[info] Embedding dimension: {embedding_dim}")
    print(f"[info] Candidate genes with embeddings: {candidate_cov} / {len(candidate_genes)}")
    print(f"[info] Wrote: {artifacts['table_path']}")
    print(f"[info] Wrote: {artifacts['pickle_path']}")
    print(f"[info] Wrote: {artifacts['npz_path']}")
    print(f"[info] Wrote: {summary_path}")


if __name__ == "__main__":
    main()

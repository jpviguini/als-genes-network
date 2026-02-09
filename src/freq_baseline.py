#!/usr/bin/env python3
import os
import re
import argparse
from typing import List, Dict, Set
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def regex_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text))

def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_gene_universe_from_csv(path: str, gene_col: str = "gene") -> List[str]:
    df = pd.read_csv(path)
    genes = (
        df[gene_col]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .str.upper()
        .unique()
        .tolist()
    )
    return genes

def main():
    ap = argparse.ArgumentParser(
        description="Build frequency (co-occurrence) baseline and export as NPZ compatible with metrics.py."
    )
    ap.add_argument("--corpus", required=True, help="CSV corpus path (must contain text and year columns).")
    ap.add_argument("--genes", required=True, help="CSV gene universe path.")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--year-col", default="year")
    ap.add_argument("--start-year", type=int, default=1970)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--gene-col", default="gene")
    ap.add_argument("--target-term", default="als_disease_token")
    ap.add_argument("--outdir", default="./scores_frequency_baseline/")
    ap.add_argument("--outfile", default="scores_frequency_cooc_ALS_1970_2026.npz")
    ap.add_argument(
        "--score",
        choices=["cooc_doc_count", "log1p"],
        default="cooc_doc_count",
        help="Which frequency score to export.",
    )
    args = ap.parse_args()

    target = args.target_term.lower()

    print("[info] Loading gene universe...")
    genes_list = load_gene_universe_from_csv(args.genes, gene_col=args.gene_col)
    gene_set: Set[str] = set(genes_list)
    print(f"[info] #genes in universe: {len(gene_set)}")

    print("[info] Reading corpus...")
    df = pd.read_csv(args.corpus, escapechar="\\")
    df[args.year_col] = df[args.year_col].astype(int)
    df = df[(df[args.year_col] >= args.start_year) & (df[args.year_col] <= args.end_year)].copy()
    texts = df[args.text_col].astype(str).tolist()
    print(f"[info] #docs in range: {len(texts)}")

    cooc_doc_count: Dict[str, int] = defaultdict(int)
    n_als_docs = 0

    print("[info] Counting co-occurrence in ALS-anchored docs...")
    for t in tqdm(texts, desc="Counting", mininterval=0.5):
        toks = regex_tokenize(t)
        if not toks:
            continue

        toks_lower = [x.lower() for x in toks]
        if target not in toks_lower:
            continue

        n_als_docs += 1

        toks_upper = [x.upper() for x in toks]
        present_genes = set(tok_u for tok_u in toks_upper if tok_u in gene_set)
        for g in present_genes:
            cooc_doc_count[g] += 1

    print(f"[info] ALS-anchored docs: {n_als_docs}")
    print(f"[info] genes with >=1 cooc: {sum(1 for g in gene_set if cooc_doc_count.get(g, 0) > 0)}")

    # export scores for all genes in the universe (missing => 0)
    scores = np.zeros(len(genes_list), dtype=np.float32)
    for i, g in enumerate(genes_list):
        c = float(cooc_doc_count.get(g, 0))
        if args.score == "log1p":
            c = np.log1p(c)
        scores[i] = c

    safe_makedirs(args.outdir)
    out_path = os.path.join(args.outdir, args.outfile)


    np.savez_compressed(out_path, genes=np.array(genes_list, dtype=object), scores_topm=scores)

    print(f"[info] Saved NPZ baseline to: {out_path}")
 
    top_idx = np.argsort(scores)[::-1][:10]
    print("[info] Top-10 by score:")
    for r, idx in enumerate(top_idx, 1):
        print(f"  {r:02d}. {genes_list[idx]}  score={scores[idx]:.3f}")

if __name__ == "__main__":
    main()

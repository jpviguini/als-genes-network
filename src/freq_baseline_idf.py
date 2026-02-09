#!/usr/bin/env python3
import os
import re
import argparse
from typing import List, Set, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer


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
        description="Build TF-IDF baseline on ALS-anchored docs and export NPZ compatible with metrics.py."
    )
    ap.add_argument("--corpus", required=True, help="CSV corpus path (must contain text and year columns).")
    ap.add_argument("--genes", required=True, help="CSV gene universe path.")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--year-col", default="year")
    ap.add_argument("--start-year", type=int, default=1970)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--gene-col", default="gene")
    ap.add_argument("--target-term", default="als_disease_token")
    ap.add_argument("--outdir", default="./scores_tfidf_baseline/")
    ap.add_argument("--outfile", default="scores_tfidf_ALS_anchored_1970_2026.npz")

    ap.add_argument(
        "--score",
        choices=["sum", "mean", "max", "topm_sum", "topm_mean"],
        default="sum",
        help="How to aggregate TF-IDF over documents for each gene.",
    )
    ap.add_argument(
        "--topm",
        type=int,
        default=50,
        help="M for topm_* aggregations (only used if score is topm_sum or topm_mean).",
    )

    # TF-IDF settings
    ap.add_argument("--min-df", type=int, default=2, help="Ignore terms that appear in < min_df docs.")
    ap.add_argument("--max-df", type=float, default=1.0, help="Ignore terms that appear in > max_df fraction of docs.")
    ap.add_argument("--sublinear-tf", action="store_true", help="Use sublinear tf scaling (log(1+tf)).")
    ap.add_argument("--use-idf", action="store_true", help="Use IDF reweighting (default True).")
    ap.add_argument("--smooth-idf", action="store_true", help="Smooth IDF (default True).")
    ap.add_argument("--norm", choices=["l2", "l1", "none"], default="l2", help="Vector norm per document.")
    ap.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase text before tokenization. (Default False to preserve gene symbols.)",
    )

    args = ap.parse_args()

    use_idf = True if args.use_idf else True
    smooth_idf = True if args.smooth_idf else True
    norm = None if args.norm == "none" else args.norm

    target = args.target_term.lower()

    print("[info] Loading gene universe...")
    genes_list = load_gene_universe_from_csv(args.genes, gene_col=args.gene_col)
    gene_set: Set[str] = set(genes_list)
    print(f"[info] #genes in universe: {len(gene_set)}")

    print("[info] Reading corpus...")
    df = pd.read_csv(args.corpus, escapechar="\\")
    df[args.year_col] = df[args.year_col].astype(int)
    df = df[(df[args.year_col] >= args.start_year) & (df[args.year_col] <= args.end_year)].copy()
    texts_all = df[args.text_col].astype(str).tolist()
    print(f"[info] #docs in range: {len(texts_all)}")


    print("[info] Filtering ALS-anchored docs...")
    anchored_texts: List[str] = []
    anchored_present_genes: List[Set[str]] = [] 
    for t in tqdm(texts_all, desc="Filter", mininterval=0.5):
        toks = regex_tokenize(t if not args.lowercase else str(t).lower())
        if not toks:
            continue
        toks_lower = [x.lower() for x in toks]
        if target not in toks_lower:
            continue
        anchored_texts.append(t)

    n_docs = len(anchored_texts)
    print(f"[info] ALS-anchored docs: {n_docs}")
    if n_docs == 0:
        raise SystemExit("[error] No ALS-anchored docs found. Check target-term / preprocessing.")

    # build TF-IDF on anchored docs.
    print("[info] Building TF-IDF matrix (sparse)...")

    def tok_func(s: str) -> List[str]:
        if args.lowercase:
            s = str(s).lower()
        return regex_tokenize(s)

    vectorizer = TfidfVectorizer(
        tokenizer=tok_func,
        preprocessor=None,
        token_pattern=None,  # must be None when providing tokenizer
        lowercase=False,     # we handle it ourselves (or not) in tok_func
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=args.sublinear_tf,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        norm=norm,
        dtype=np.float32,
    )

    X = vectorizer.fit_transform(anchored_texts)  # shape: (n_docs, n_terms)
    vocab = vectorizer.vocabulary_  # term -> column index
    print(f"[info] TF-IDF matrix shape: {X.shape}  nnz={X.nnz}")

    # map genes to TF-IDF columns.
    gene_to_col: Dict[str, int] = {}
    missing = 0
    for g in genes_list:
        if g in vocab:
            gene_to_col[g] = vocab[g]
        else:
            # fallback: if tokenizer produced lowercase tokens for some reason
            g_lower = g.lower()
            if g_lower in vocab:
                gene_to_col[g] = vocab[g_lower]
            else:
                missing += 1

    print(f"[info] Genes with TF-IDF column: {len(gene_to_col)} / {len(genes_list)} (missing={missing})")

    # aggregate per gene
    scores = np.zeros(len(genes_list), dtype=np.float32)

    if args.score in ("sum", "mean", "max"):
 
        for i, g in enumerate(tqdm(genes_list, desc="Scoring", mininterval=0.5)):
            col = gene_to_col.get(g, None)
            if col is None:
                continue
            vec = X[:, col]
            if vec.nnz == 0:
                continue
            if args.score == "sum":
                scores[i] = float(vec.sum())
            elif args.score == "mean":
                scores[i] = float(vec.sum() / n_docs)
            else:  # max
                scores[i] = float(vec.max())

    else:
 
        M = max(1, int(args.topm))
        for i, g in enumerate(tqdm(genes_list, desc="Scoring(topM)", mininterval=0.5)):
            col = gene_to_col.get(g, None)
            if col is None:
                continue
            vec = X[:, col]
            if vec.nnz == 0:
                continue
            data = vec.data  # nonzero values
            if data.size == 0:
                continue
            if data.size <= M:
                top = data
            else:
                # partial selection without full sort
                top = np.partition(data, -M)[-M:]
            s = float(top.sum())
            if args.score == "topm_mean":
                s = s / float(top.size)
            scores[i] = s

    safe_makedirs(args.outdir)
    out_path = os.path.join(args.outdir, args.outfile)


    np.savez_compressed(out_path, genes=np.array(genes_list, dtype=object), scores_topm=scores)

    print(f"[info] Saved NPZ baseline to: {out_path}")


    top_idx = np.argsort(scores)[::-1][:10]
    print("[info] Top-10 by TF-IDF score:")
    for r, idx in enumerate(top_idx, 1):
        print(f"  {r:02d}. {genes_list[idx]}  score={scores[idx]:.6f}")

if __name__ == "__main__":
    main()

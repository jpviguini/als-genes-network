import os
import json
import math
import heapq
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


MODEL_MAP = {
    "bert-base": "bert-base-uncased",
    "biobert": "dmis-lab/biobert-v1.1",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "scibert": "allenai/scibert_scivocab_uncased",
}

SELECTED_MODEL_KEY = "biobert"
OVERRIDE_MODEL_NAME = "./biobert_als_adapted_model"  # local dir (fine tuned) or "" to use MODEL_MAP

CORPUS_CSV_PATH = "../data/corpus_als_general_pmc_preprocessed3.csv"
TEXT_COL = "text"
YEAR_COL = "year"
START_YEAR = 1970
END_YEAR = 2026

GENE_UNIVERSE_CSV_PATH = "../data/genes_extracted_validated_general_pmc3.csv"
GENE_COL = "gene"

TARGET_TERMS = ("als_disease_token",)

BATCH_SIZE = 16
MAX_LEN = 512

USE_LAST4_AVG = True # true: mean of last 4 hidden layers  -- false: use last hidden layer
MENTION_L2NORM = True
USE_AMP_ON_CUDA = True

TOP_M = 0 # top-m (10,20,50). 0 takes all
MIN_CTX_FOR_RANK = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = f"./scores_{SELECTED_MODEL_KEY}_top{TOP_M}/"
OUTPUT_FILENAME = f"scores_top{TOP_M}_ALS_{START_YEAR}_{END_YEAR}.npz"

TQDM_MININTERVAL = 0.5


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < eps else (v / n)


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_gene_universe_from_csv(path: str, gene_col: str = "gene") -> Set[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"gene universe csv not found: {path}")
    df = pd.read_csv(path)
    if gene_col not in df.columns:
        raise ValueError(f"gene universe csv must have column '{gene_col}'. found: {df.columns.tolist()}")
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
    return set(genes)


def push_topm(heap: List[float], value: float, m: int) -> None:
    if m <= 0:
        return
    if len(heap) < m:
        heapq.heappush(heap, value)
    else:
        if value > heap[0]:
            heapq.heapreplace(heap, value)


def resolve_local_dir(path: str) -> str:
    
    if not path:
        return ""
    path = os.path.expanduser(path)

    candidates = [path]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, path))
    candidates.append(os.path.join(os.path.dirname(script_dir), path))

    for p in candidates:
        p_abs = os.path.abspath(p)
        if os.path.isdir(p_abs):
            return p_abs

    return ""


def resolve_model_name_and_mode() -> Tuple[str, bool]:

    # returns (model_name, local_files_only)

    name = (OVERRIDE_MODEL_NAME or "").strip()
    if name:
        local_dir = resolve_local_dir(name)
        if local_dir:
            return local_dir, True
        raise FileNotFoundError(
            f"local model dir not found: '{name}'. tried cwd, script dir, and project root."
        )
    return MODEL_MAP[SELECTED_MODEL_KEY], False


@torch.inference_mode()
def extract_topm_cosine_scores(
    texts: List[str],
    tokenizer,
    model,
    valid_candidates: Set[str],
    target_terms: Tuple[str, ...],
    batch_size: int,
    max_len: int,
    use_last4_avg: bool,
    mention_l2norm: bool,
    use_amp_on_cuda: bool,
    top_m: int,
) -> Tuple[Dict[str, List[float]], Dict[str, int]]:

    # each abstract is treated as a single context; tokenizer truncation handles >512 tokens
    
    device = next(model.parameters()).device
    target_terms_set = set(t.lower() for t in target_terms)

    gene_topm: Dict[str, List[float]] = defaultdict(list)
    gene_ctx_count: Dict[str, int] = defaultdict(int)

    total_batches = math.ceil(len(texts) / batch_size) if texts else 0
    pbar = tqdm(
        range(0, len(texts), batch_size),
        total=total_batches,
        desc="extracting + scoring",
        unit="batch",
        mininterval=TQDM_MININTERVAL,
    )

    for start in pbar:
        batch_texts = texts[start : start + batch_size]
        batch_words = [str(t).split() for t in batch_texts]

        encoded = tokenizer(
            batch_words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        if use_amp_on_cuda and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**encoded, output_hidden_states=True)
        else:
            outputs = model(**encoded, output_hidden_states=True)

        if use_last4_avg:
            stacked = torch.stack(outputs.hidden_states[-4:])
            token_embeddings = torch.mean(stacked, dim=0)
        else:
            token_embeddings = outputs.last_hidden_state

        token_embeddings = token_embeddings.float().cpu().numpy()

        for b_idx, words in enumerate(batch_words):
            if not words:
                continue
            if not any(w.lower() in target_terms_set for w in words):
                continue

            wids = encoded.word_ids(batch_index=b_idx)
            vecs = token_embeddings[b_idx]

            by_word: Dict[int, List[np.ndarray]] = {}
            for t_idx, w_idx in enumerate(wids):
                if w_idx is None:
                    continue
                by_word.setdefault(w_idx, []).append(vecs[t_idx])

            als_vecs: List[np.ndarray] = []
            gene_vecs_in_ctx: List[Tuple[str, np.ndarray]] = []

            for w_idx, sub_vecs in by_word.items():
                if w_idx >= len(words):
                    continue

                w = words[w_idx].strip(".,;:()[]{}<>\"'")
                if not w:
                    continue

                wl = w.lower()
                wu = w.upper()

                v = np.mean(np.stack(sub_vecs, axis=0), axis=0)
                if mention_l2norm:
                    v = l2_normalize(v)

                if wl in target_terms_set:
                    als_vecs.append(v)
                elif wu in valid_candidates:
                    gene_vecs_in_ctx.append((wu, v))

            if not als_vecs or not gene_vecs_in_ctx:
                continue

            als_ctx = np.mean(np.stack(als_vecs, axis=0), axis=0)
            if mention_l2norm:
                als_ctx = l2_normalize(als_ctx)

            for g, gv in gene_vecs_in_ctx:
                score = float(np.dot(gv, als_ctx)) if mention_l2norm else float(
                    np.dot(l2_normalize(gv), l2_normalize(als_ctx))
                )


                if top_m > 0:
                    push_topm(gene_topm[g], score, top_m)
                else: # M == all
                    gene_topm[g].append(score)
                
                gene_ctx_count[g] += 1

        pbar.set_postfix({"genes_seen": len(gene_ctx_count)})

    return gene_topm, gene_ctx_count


def save_scores_npz(
    gene_topm: Dict[str, List[float]],
    gene_ctx_count: Dict[str, int],
    output_path: str,
    meta: dict,
    min_ctx_for_rank: int,
) -> None:
    genes, scores, ctx_counts = [], [], []

    for g, heap_scores in gene_topm.items():
        c = gene_ctx_count.get(g, 0)
        if c < min_ctx_for_rank:
            continue
        if not heap_scores:
            continue
        genes.append(g)
        scores.append(float(np.mean(heap_scores)))
        ctx_counts.append(int(c))

    if not genes:
        raise RuntimeError("No genes to save (empty after filtering).")

    np.savez_compressed(
        output_path,
        genes=np.array(genes, dtype=np.str_),
        scores_topm=np.array(scores, dtype=np.float32),
        ctx_counts=np.array(ctx_counts, dtype=np.int32),
        meta=np.array([json.dumps(meta, ensure_ascii=False)], dtype=np.str_),
    )


def main():
    print(f"[info] Device: {DEVICE}")

    model_name, local_only = resolve_model_name_and_mode()
    print(f"[info] Loading model: {model_name} (local_files_only={local_only})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_only).to(DEVICE)
    model.eval()

    safe_makedirs(OUTPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    print(f"[info] Loading gene universe from: {GENE_UNIVERSE_CSV_PATH}")
    valid_candidates = load_gene_universe_from_csv(GENE_UNIVERSE_CSV_PATH, gene_col=GENE_COL)
    print(f"[info] Candidate genes loaded: {len(valid_candidates)}")

    if not os.path.exists(CORPUS_CSV_PATH):
        raise FileNotFoundError(f"Corpus not found: {CORPUS_CSV_PATH}")

    print("[info] Reading corpus csv...")
    df = pd.read_csv(CORPUS_CSV_PATH, escapechar="\\")
    if YEAR_COL not in df.columns or TEXT_COL not in df.columns:
        raise ValueError(f"csv must have columns '{TEXT_COL}' and '{YEAR_COL}'. found: {df.columns.tolist()}")

    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df = df[(df[YEAR_COL] >= START_YEAR) & (df[YEAR_COL] <= END_YEAR)].copy()

    n_docs = len(df)
    print(f"[info] Docs in range {START_YEAR}-{END_YEAR}: {n_docs}")
    if n_docs == 0:
        raise RuntimeError("No documents in the selected year range.")

    texts = df[TEXT_COL].astype(str).tolist()

    print(f"[info] extracting + scoring with top-{TOP_M} mean...")
    gene_topm, gene_ctx_count = extract_topm_cosine_scores(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        valid_candidates=valid_candidates,
        target_terms=TARGET_TERMS,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        use_last4_avg=USE_LAST4_AVG,
        mention_l2norm=MENTION_L2NORM,
        use_amp_on_cuda=USE_AMP_ON_CUDA,
        top_m=TOP_M,
    )

    meta = {
        "model_name": model_name,
        "model_key": SELECTED_MODEL_KEY,
        "year_range": [START_YEAR, END_YEAR],
        "n_docs": int(n_docs),
        "pooling": "avg_last4" if USE_LAST4_AVG else "last_layer",
        "mention_l2norm": bool(MENTION_L2NORM),
        "max_len": int(MAX_LEN),
        "batch_size": int(BATCH_SIZE),
        "target_terms": list(TARGET_TERMS),
        "n_candidate_genes": int(len(valid_candidates)),
        "top_m": int(TOP_M),
        "min_ctx_for_rank": int(MIN_CTX_FOR_RANK),
    }

    print(f"[info] Saving scores to: {out_path}")
    save_scores_npz(
        gene_topm=gene_topm,
        gene_ctx_count=gene_ctx_count,
        output_path=out_path,
        meta=meta,
        min_ctx_for_rank=MIN_CTX_FOR_RANK,
    )

    print("[info] Done.")
    print(f"[info] Output: {out_path}")


if __name__ == "__main__":
    main()

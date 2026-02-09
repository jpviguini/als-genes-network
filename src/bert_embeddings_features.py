import os
import re
import math
import pickle
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

SELECTED_MODEL_KEY = "scibert"
OVERRIDE_MODEL_NAME = f"./{SELECTED_MODEL_KEY}_als_adapted_model"

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
USE_LAST4_AVG = True
USE_AMP_ON_CUDA = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = f"./features_NOT_{SELECTED_MODEL_KEY}/"
OUTPUT_FILENAME = f"features_ALS_{START_YEAR}_{END_YEAR}.pkl"
TQDM_MININTERVAL = 0.5

# regex tokenization
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def regex_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text))


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_local_dir(path: str) -> str:
    if not path:
        return ""
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        return path
    return ""


def resolve_model_name_and_mode() -> Tuple[str, bool]:
    name = (OVERRIDE_MODEL_NAME or "").strip()
    if name:
        local_dir = resolve_local_dir(name)
        if local_dir:
            return local_dir, True
    return MODEL_MAP[SELECTED_MODEL_KEY], False


def load_gene_universe_from_csv(path: str, gene_col: str = "gene") -> Set[str]:
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
    return set(genes)


@torch.inference_mode()
def extract_article_level_features(
    texts: List[str],
    tokenizer,
    model,
    valid_candidates: Set[str],
    target_terms: Tuple[str, ...],
    batch_size: int,
    max_len: int,
    use_last4_avg: bool,
    use_amp_on_cuda: bool,
) -> Dict[str, List[np.ndarray]]:

    device = next(model.parameters()).device
    target_terms_set = set(t.lower() for t in target_terms)

    gene_article_bags: Dict[str, List[np.ndarray]] = defaultdict(list)

    total_batches = math.ceil(len(texts) / batch_size) if texts else 0
    pbar = tqdm(
        range(0, len(texts), batch_size),
        total=total_batches,
        desc="Extracting Article Features",
        mininterval=TQDM_MININTERVAL,
    )

    for start in pbar:
        batch_texts = texts[start: start + batch_size]


        batch_words = [regex_tokenize(t) for t in batch_texts]

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
            token_embeddings = torch.mean(torch.stack(outputs.hidden_states[-4:]), dim=0)
        else:
            token_embeddings = outputs.last_hidden_state

        token_embeddings = token_embeddings.float().cpu().numpy()

        for b_idx, words in enumerate(batch_words):
            if not words:
                continue

            # filter als_disease_token articles
            if not any(w.lower() in target_terms_set for w in words):
                continue

            wids = encoded.word_ids(batch_index=b_idx)
            vecs = token_embeddings[b_idx]

            by_word: Dict[int, List[np.ndarray]] = {}
            for t_idx, w_idx in enumerate(wids):
                if w_idx is None:
                    continue
                by_word.setdefault(w_idx, []).append(vecs[t_idx])

            disease_vectors = []
            gene_vectors_map = defaultdict(list)

            for w_idx, sub_vecs in by_word.items():
                if w_idx >= len(words):
                    continue

                w = words[w_idx]
                if not w:
                    continue

                wl = w.lower()
                wu = w.upper()

                word_vec = np.mean(np.stack(sub_vecs, axis=0), axis=0)

                if wl in target_terms_set:
                    disease_vectors.append(word_vec)
                elif wu in valid_candidates:
                    gene_vectors_map[wu].append(word_vec)

            if not disease_vectors:
                continue

            article_disease_vec = np.mean(np.stack(disease_vectors, axis=0), axis=0)

            for gene, g_vecs in gene_vectors_map.items():
                article_gene_vec = np.mean(np.stack(g_vecs, axis=0), axis=0)

                g_norm = article_gene_vec / (np.linalg.norm(article_gene_vec) + 1e-9)
                d_norm = article_disease_vec / (np.linalg.norm(article_disease_vec) + 1e-9)
                cosine_sim = float(np.dot(g_norm, d_norm))

                interaction = article_gene_vec * article_disease_vec

                final_feature = np.concatenate(
                    [
                        article_gene_vec,
                        article_disease_vec,
                        #interaction,
                        #np.array([cosine_sim], dtype=np.float32),
                    ],
                    axis=0,
                )

                gene_article_bags[gene].append(final_feature)
                # gene_article_bags[gene].append({
                #     "feat": final_feature.astype(np.float32),
                #     "row_id": int(start + b_idx),   
                #     # "year": int(df_year),         
                # })

    return gene_article_bags


def main():
    print(f"[info] Device: {DEVICE}")
    model_name, local_only = resolve_model_name_and_mode()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_only).to(DEVICE)
    model.eval()

    print("[info] Loading candidates...")
    valid_candidates = load_gene_universe_from_csv(GENE_UNIVERSE_CSV_PATH, gene_col=GENE_COL)

    print("[info] Reading corpus...")
    df = pd.read_csv(CORPUS_CSV_PATH, escapechar="\\")
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df = df[(df[YEAR_COL] >= START_YEAR) & (df[YEAR_COL] <= END_YEAR)].copy()
    texts = df[TEXT_COL].astype(str).tolist()

    print("[info] Extracting ARTICLE-LEVEL interaction features...")
    gene_bags = extract_article_level_features(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        valid_candidates=valid_candidates,
        target_terms=TARGET_TERMS,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        use_last4_avg=USE_LAST4_AVG,
        use_amp_on_cuda=USE_AMP_ON_CUDA,
    )

    safe_makedirs(OUTPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    print(f"[info] Saving {len(gene_bags)} gene bags to {out_path}...")

    with open(out_path, "wb") as f:
        pickle.dump(gene_bags, f)



    print("[info] Done. Ready for MIL training.")


if __name__ == "__main__":
    main()

import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import sys
import os


W2V_MODEL_PATH = "./word2vec_models/model_ALS_v200_a0p05_n15.model"
GENE_FREQ_CSV = "../data/genes_extracted_validated_general_pmc3.csv"
OUT_NPZ = "./scores_word2vec/scores_word2vec_allgenes.npz"

GENE_COL = "gene"
DISEASE_TOKEN = "als_disease_token" 


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def main():
    print("[info] Loading Word2Vec model...")
    try:
        model = Word2Vec.load(W2V_MODEL_PATH)
        wv = model.wv
    except:
        wv = KeyedVectors.load(W2V_MODEL_PATH)

    print(f"[info] Loading gene CSV: {GENE_FREQ_CSV}")
    try:
        df = pd.read_csv(GENE_FREQ_CSV)
    except FileNotFoundError:
        print(f"[ERROR] CSV not found at {GENE_FREQ_CSV}")
        sys.exit(1)

    if GENE_COL not in df.columns:
        print(f"[ERROR] Column '{GENE_COL}' not found. Available: {df.columns.tolist()}")
        sys.exit(1)

    if DISEASE_TOKEN not in wv:
        raise ValueError(f"Disease token '{DISEASE_TOKEN}' not in Vocabulary")

    disease_vec = wv[DISEASE_TOKEN]

    genes = []
    scores = []
    counts = []
    

    seen_genes = set()

    print("[info] Computing scores and extracting counts from model...")

    for _, row in df.iterrows():
        raw_gene = str(row[GENE_COL]).strip()
        
        if raw_gene in wv:
            key = raw_gene
        elif raw_gene.upper() in wv:
            key = raw_gene.upper()
        elif raw_gene.lower() in wv:
            key = raw_gene.lower()
        else:
            continue 
            
        if key in seen_genes:
            continue
        seen_genes.add(key)

        score = cosine(wv[key], disease_vec)

 
        try:
            cnt = wv.get_vecattr(key, "count")
        except:
            try:
                cnt = wv.vocab[key].count
            except:
                cnt = 1

        genes.append(key) 
        scores.append(score)
        counts.append(cnt)

    print(f"[info] Scored {len(genes)} unique genes found in model.")

    if len(genes) == 0:
        print("[ERROR] No genes matched. Check casing or CSV content.")
        sys.exit(1)


    output_dir = os.path.dirname(OUT_NPZ)
    if output_dir and not os.path.exists(output_dir):
        print(f"[info] Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        OUT_NPZ,
        genes=np.array(genes, dtype=np.str_),
        scores_topm=np.array(scores, dtype=np.float32),
        ctx_counts=np.array(counts, dtype=np.int32),
        meta=np.array(["word2vec_cosine_extracted_counts"], dtype=np.str_)
    )

    print(f"[info] Saved to {OUT_NPZ}")

if __name__ == "__main__":
    main()
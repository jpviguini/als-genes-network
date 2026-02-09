import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler, StandardScaler


MODELS_DIR = "./models_word2vec_timespan_general/"
OUTPUT_DIR = "./validation/per_gene/w2v/"
GENES_FILE = "../data/core_genes.txt"   # file with one gene per line
TARGET_WORD = "als_disease_token"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_genes(genes_file):
    with open(genes_file, "r") as f:
        genes = [line.strip().lower() for line in f if line.strip()]
    return genes

def normalize_values(values):
    """
    normalize a list of values with three different methods
        - Minmax
        - zcscore
        - Softmax
    """
    values = np.array(values).reshape(-1, 1)
    minmax = MinMaxScaler().fit_transform(values).flatten() # we are mainly using this
    zscore = StandardScaler().fit_transform(values).flatten()
    softmaxed = softmax(values).flatten()
    return minmax, zscore, softmaxed

def main():
    print("Starting script of generating dot products (genes x 'als_disease_token')\n")

    genes = load_genes(GENES_FILE)
    print(f"Total of genes: {len(genes)}")

    # loads all available models
    model_files = sorted([
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".model")
    ])

    years = []
    for f in model_files:
        # extracts the last year: model_ALS_1923_1930.model -> 1930
        year = int(f.split("_")[-1].replace(".model", ""))
        years.append(year)

    print(f"Found models ({len(years)}): {years}\n")

    # main loop
    for gene in genes:
        results = {"year": [], "dot": []}

        for file, year in zip(model_files, years):
            model_path = os.path.join(MODELS_DIR, file)
            try:
                model = Word2Vec.load(model_path)
                vocab = model.wv.key_to_index

                if TARGET_WORD in vocab and gene in vocab:
                    dot_value = np.dot(model.wv[gene], model.wv[TARGET_WORD])
                    results["year"].append(year)
                    results["dot"].append(dot_value)
                else:
                    # gene or 'als' are not in the vocab
                    results["year"].append(year)
                    results["dot"].append(np.nan)
            except Exception as e:
                print(f"Error when loading {file}: {e}")
                continue

        # creates dataframe and normalizes
        df = pd.DataFrame(results)
        if df["dot"].notna().sum() == 0:
            print(f"No valid value for gene {gene}, ignoring.")
            continue


        df["dot_minmax"], df["dot_zscore"], df["dot_softmax"] = normalize_values(
            df["dot"].fillna(0)
        )

        output_path = os.path.join(OUTPUT_DIR, f"{gene}.csv")
        df.to_csv(output_path, index=False)
        print(f"Gene {gene} saved in {output_path}")

    print("\nEND.")

if __name__ == "__main__":
    main()

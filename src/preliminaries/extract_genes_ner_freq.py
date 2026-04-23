import os
from pathlib import Path
import scispacy
import spacy
import mygene
import pandas as pd
from tqdm import tqdm
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # forces only CPU execution (skips gpu warnings)

umbrella_term = "motor_neuron_disease"

project_root = Path(__file__).resolve().parents[2]
input_path = project_root / "data" / "corpus" / "raw" / f"corpus_{umbrella_term}.csv"
output_dir = project_root / "data" / "corpus" / "extracted_genes"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)  # original text (not preprocessed)
texts = df['text'].dropna().tolist()
print(f"Total of texts to process: {len(texts)}")

print("Loading model 'en_ner_bionlp13cg_md'...")
nlp = spacy.load("en_ner_bionlp13cg_md", disable=["parser", "lemmatizer"])
print("Model loaded successfully")

genes_extracted = set()

gene_counts = Counter()


def is_full_uppercase_mention(text):
    mention = text.strip()
    letters = [ch for ch in mention if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)


print("Extracting genes with scispaCy...")

# nlp.pipe --> processing batches
for doc in tqdm(nlp.pipe(texts, batch_size=8), total=len(texts)):
    for ent in doc.ents:
        if ent.label_ == "GENE_OR_GENE_PRODUCT":
            if not is_full_uppercase_mention(ent.text):
                continue
            g = ent.text.lower().strip()
            genes_extracted.add(g)      
            gene_counts[g] += 1         

print(f"Total extracted by NER: {len(genes_extracted)} unique entities")

# validation with mygene
mg = mygene.MyGeneInfo()
genes_list = list(genes_extracted)
print("Consulting mygene for validation...")

valid_genes = set()
batch_size = 1000

for i in tqdm(range(0, len(genes_list), batch_size)):
    batch = genes_list[i:i + batch_size]
    try:
        results = mg.querymany(
            batch,
            scopes="symbol,alias,name",
            fields="symbol,name,taxid",
            species="human",
            as_dataframe=False,
            verbose=False
        )
        for r in results:
            if not r.get("notfound") and "symbol" in r:
                valid_genes.add(r["symbol"].lower())
    except Exception as e:
        print(f"Error when consulting batch {i // batch_size + 1}: {e}")
        continue

genes = list(valid_genes)
print(f"Validated genes with MyGene: {len(genes)} / {len(genes_list)}")


output_path = output_dir / f"genesUPPER_extracted_{umbrella_term}.csv"
pd.DataFrame({"gene": genes}).to_csv(output_path, index=False)
print(f"Saved file in: {output_path}")
print(f"Example of validated genes: {genes[:20]}")

genes_with_counts = [(g, int(gene_counts.get(g, 0))) for g in genes]

output_path_freq = output_dir / f"genesUPPER_extracted_{umbrella_term}_with_freq.csv"
pd.DataFrame(genes_with_counts, columns=["gene", "count"]).to_csv(output_path_freq, index=False)

print(f"Saved frequency file in: {output_path_freq}")
print(pd.DataFrame(genes_with_counts, columns=["gene", "count"]).sort_values("count", ascending=False).head(20))

import os
import scispacy
import spacy
import mygene
import pandas as pd
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # forces only CPU execution (skips gpu warnings)


df = pd.read_csv("../data/corpus_als_general_pmc3.csv")
texts = df['text'].dropna().tolist()
print(f"Total of texts to process: {len(texts)}")


print("Loading model 'en_ner_bionlp13cg_md'...")
nlp = spacy.load("en_ner_bionlp13cg_md", disable=["parser", "lemmatizer"])
print("Model loaded successfully")


genes_extracted = set()

print("Extracting genes with scispaCy...")

# nlp.pipe --> processing batches
for doc in tqdm(nlp.pipe(texts, batch_size=8), total=len(texts)):
    for ent in doc.ents:
        if ent.label_ == "GENE_OR_GENE_PRODUCT":
            genes_extracted.add(ent.text.lower().strip())

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


output_path = "../data/genes_extracted_validated_general_pmc3.csv"
pd.DataFrame({"gene": genes}).to_csv(output_path, index=False)

print(f"Saved file in: {output_path}")
print(f"Example of validated genes: {genes[:20]}")



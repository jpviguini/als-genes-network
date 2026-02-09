import pandas as pd
import operator

# MODEL_PATH = "../models/word2vec_model_2025_19322_biased_updated.bin"
CORPUS_PATH = "../data/corpus_als_improved.csv"
OUTPUT_PATH = "../data/analogies_als_corrected"
FREQUENCY_THRESHOLD = 50



def get_ALS_genes_symptoms():
    """
    Returns a dictionary with genes and their main symptoms (from OMIM)  https://omim.org/.
    """

    return {
        'ANXA11': ['myopathy'],
        'C9ORF72': ['dementia'],
        'CHCHD10': ['myopathy', 'dementia', 'atrophy'],
        'EPHA4': [],
        'FUS': ['tremor', 'dementia'],
        'HNRNPA1': ['myopathy', 'paget', 'dementia'],
        'KIF5A': ['myoclonus', 'paraplegia'],
        'NEK1': ['dysplasia', 'polydactyly'],
        'OPTN': ['dementia', 'glaucoma'],
        'PFN1': [],
        'SOD1': ['tetraplegia', 'hypotonia'],
        'TARDBP': ['dementia', 'degeneration'],
        'TDP-43': [],
        'TBK1': ['encephalopathy', 'arthritis', 'vasculitis', 'dementia'],
        'UBQLN2': ['dementia'],
        'UNC13A': [],
        'VAPB': ['atrophy', 'finkel'],
        'VCP': ['dementia', 'myopathy', 'paget']
    }



def get_words_frequency(corpus_path):
    """
    returns token frequency within a corpus
    """
    df = pd.read_csv(corpus_path)
    abstracts = df['text'].astype(str).tolist()
    abstracts = [x.split() for x in abstracts]

    freq = {}
    for tokens in abstracts:
        for word in tokens:
            freq[word.lower()] = freq.get(word.lower(), 0) + 1

    return dict(sorted(freq.items(), key=operator.itemgetter(1), reverse=True))


def generate_analogies(freq_dict, freq_threshold=100):
    """
    Generate analogies, filtering by a min frequency
    """
    
    analogies = []
    als_dict = get_ALS_genes_symptoms()

    for gene1, symp_list1 in als_dict.items():
        for gene2, symp_list2 in als_dict.items():
            if gene1 == gene2:
                continue

            for symp1 in symp_list1:
                # 1. gene1 : sympton_gene1 :: gene2 : als
                if all([
                    freq_dict.get(gene1.lower(), 0) >= freq_threshold,
                    freq_dict.get(symp1.lower(), 0) >= freq_threshold,
                    freq_dict.get(gene2.lower(), 0) >= freq_threshold,
                    freq_dict.get("als", 0) >= freq_threshold
                ]):
                    analogies.append(f"{gene1.lower()} {symp1.lower()} {gene2.lower()} als\n")

                # 2. gene1 : sympton_gene1 :: gene2 : sympton_gene2
                for symp2 in symp_list2:
                    if all([
                        freq_dict.get(gene1.lower(), 0) >= freq_threshold,
                        freq_dict.get(symp1.lower(), 0) >= freq_threshold,
                        freq_dict.get(gene2.lower(), 0) >= freq_threshold,
                        freq_dict.get(symp2.lower(), 0) >= freq_threshold
                    ]):
                        analogies.append(f"{gene1.lower()} {symp1.lower()} {gene2.lower()} {symp2.lower()}\n")

                        # 3. sympton_gene1 : gene1 :: sympton_gene2 : gene2
                        analogies.append(f"{symp1.lower()} {gene1.lower()} {symp2.lower()} {gene2.lower()}\n")

    return analogies


if __name__ == "__main__":
  
    print("Calculating word frequencies...")
    freq_dict = get_words_frequency(CORPUS_PATH)

    print("Generating analogies...")
    analogies = generate_analogies(freq_dict, freq_threshold=FREQUENCY_THRESHOLD)

    print(f"Saving {len(analogies)} ALS analogies in {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        for a in analogies:
            f.write(a)

    print("END.")

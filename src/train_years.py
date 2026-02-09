# this is for training one model per year (dot product validation afterwards)

import os, re, sys, shutil, itertools, gensim
from gensim.models import Word2Vec, FastText
import pandas as pd
from train_hyperparameter import list_from_txt, keep_target_als_genes

if __name__ == '__main__':
    print('Starting script')

    MODEL_TYPE = 'ft'  # 'w2v' or 'ft'


    if MODEL_TYPE == 'w2v':
        output_dir = './models_word2vec_timespan_general/'
        training_params = [200, 0.05, 15]
    else:
        output_dir = './models_fasttext_timespan_general/'
        training_params = [200, 0.0025, 5] 


    os.makedirs(output_dir, exist_ok=True)

    # loading corpus
    print('Reading DataFrame of papers...')
    df = pd.read_csv('../data/corpus_als_general_pmc_preprocessed3.csv', escapechar='\\')
    
    df = df[df["year"] >= 1970] # filtering to start in 1970 (there are very few articles before that)

    years = sorted(df.year.unique().tolist())
    first_year = years[0]
    ranges = [years[:i+1] for i in range(len(years))]

    for r in ranges:
        print(f"Training model from {r[0]} to {r[-1]}")
        abstracts = df[df.year.isin(r)]['text'].astype(str).tolist()
        abstracts = [x.split() for x in abstracts]
        print(f"Number of abstracts: {len(abstracts)}\n")

        if MODEL_TYPE == 'w2v':
            model = Word2Vec(
                sentences=abstracts,
                min_count=5,
                sg=1,
                hs=0,
                epochs=15,
                trim_rule=keep_target_als_genes,
                vector_size=training_params[0],
                alpha=training_params[1],
                negative=training_params[2],
                workers=16
            )
        else:
            model = FastText(
                sentences=abstracts,
                min_count=5,
                sg=1,
                hs=0,
                epochs=15,
                trim_rule=keep_target_als_genes,
                vector_size=training_params[0],
                alpha=training_params[1],
                negative=training_params[2],
                workers=16
            )

        model_path = os.path.join(output_dir, f"model_ALS_{first_year}_{r[-1]}.model")
        model.save(model_path)

    print('END!')

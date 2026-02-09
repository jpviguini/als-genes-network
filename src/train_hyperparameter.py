

import os, re, sys, shutil, itertools, gensim
import pandas as pd
import numpy as np
import config
from gensim.utils import RULE_KEEP, RULE_DEFAULT
from gensim.models import Word2Vec, FastText
from pathlib import Path
from os import listdir


def list_from_txt(file_path):
    """
    Creates a list of itens based on a .txt file, each line becomes an item.
    """
    strings_list = []
    with open (file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            strings_list.append(line.rstrip('\n'))
    return strings_list


def get_target_als_genes():
    """
    Return the list of target genes that we want to keep in the vocabulary (even if min_count is too low)
    """
   
    genes = config.VALIDATION_GENES
    return sorted(list(set([g.lower() for g in genes])))


def keep_target_als_genes(word, count, min_count):
 
    if word.lower() in get_target_als_genes():
        return gensim.utils.RULE_KEEP
    else:
        return gensim.utils.RULE_DEFAULT

if __name__ == '__main__':
    print('Starting script')

    TRAINING_FASTTETX_MODELS = True # false --> word2vec
    DATA_CSV_PATH = '../data/corpus_als_general_pmc_preprocessed3.csv' 
    TEXT_COLUMN_NAME = 'text' 
    
    # grid search
    # parm_dict = {
    #     'size': (100, 200, 300), 
    #     'alpha': (0.0025, 0.025, 0.05), 
    #     'negative': (5, 10, 15)
    # }

    parm_dict = {
        'size': ([200]), 
        'alpha': ([0.0025]), 
        'negative': ([5])
    }
    
    MIN_COUNT = 5 # if the word is mentioned at least MIN_COUNT times 
    ITERATIONS = 15 
    SG = 1           # 1 for Skip-Gram, 0 for CBOW
    HS = 0           # 0 for Negative Sampling, 1 for Hierarchical Softmax
    

    models_already_trained = []

    if TRAINING_FASTTETX_MODELS:
        MODEL_DIR = '../fasttext_models/'
        model_type = 'FastText'
    else:
        MODEL_DIR = './word2vec_models/' 
        model_type = 'Word2Vec'
        
    os.makedirs(MODEL_DIR, exist_ok=True)
    models_already_trained = [x for x in os.listdir(MODEL_DIR) if x.endswith('.model')]
    
    print(f"Training models of type: {model_type}")
    print(f"Models will be saved in: {MODEL_DIR}")

    
    print(f"Reading DataFrame of articles: {DATA_CSV_PATH}")
    if not os.path.exists(DATA_CSV_PATH):
        print(f"ERROR: File not found: {DATA_CSV_PATH}")
        print("Verify the variable 'DATA_CSV_PATH'.")
        sys.exit()
        
    df = pd.read_csv(DATA_CSV_PATH, escapechar='\\')
    
   
    if TEXT_COLUMN_NAME not in df.columns:
        print(f"ERROR: Column '{TEXT_COLUMN_NAME}' not found in CSV.")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit()

    
    df = df.dropna(subset=[TEXT_COLUMN_NAME])
    abstracts_raw = df[TEXT_COLUMN_NAME].to_list()
    
    print('Processing and tokenizing texts...')
    
    abstracts_tokenized = [str(x).lower().split() for x in abstracts_raw]
    
    print(f'Number of abstracts to train: {len(abstracts_tokenized)}\n')


    # grid search and training
    size, alpha, negative = [tup for k, tup in parm_dict.items()]
    parm_combo = list(itertools.product(size, alpha, negative))
    total_models = len(parm_combo)

    print(f'Starting Gridsearch for {total_models} hyperparameters combinations...')
    
    for index, parms in enumerate(parm_combo):
        v, a, n = parms # v = vector size, a = alpha, n = negative
        
        alpha_str = str(a).replace('.', 'p') # converts '0.025' to '0p025' for the filename
        
        print(f"Model {index+1}: vector_size={v}, alpha={a}, negative={n}")
        # format: model_ALS_v100_a0p025_n5.model
        model_name = f'model_ALS_v{v}_a{alpha_str}_n{n}.model'
        
        model_save_path = os.path.join(MODEL_DIR, model_name)
        
        
        if model_name in models_already_trained:
            print(f"Skipping {index+1}/{total_models}: {model_name} (already exists)")
            continue

        else:
            print(f'Training model {index+1}/{total_models}: {model_name}')

            if TRAINING_FASTTETX_MODELS:
                model = FastText(
                    sentences=abstracts_tokenized,
                    sorted_vocab=True,
                    min_count=MIN_COUNT,
                    sg=SG,
                    hs=HS,
                    epochs=ITERATIONS,
                    trim_rule=keep_target_als_genes,
                    vector_size=v,
                    alpha=a,
                    negative=n,
                    workers=16
                )
            else: # word2vec
                model = Word2Vec(
                    sentences=abstracts_tokenized,
                    sorted_vocab=True,
                    min_count=MIN_COUNT,
                    sg=SG,
                    hs=HS,
                    epochs=ITERATIONS,
                    trim_rule=keep_target_als_genes,
                  
                    vector_size=v,
                    alpha=a,
                    negative=n,
                    workers=16
                )
            
            print(f"Saving model in: {model_save_path}")
            model.save(model_save_path)

    print('\nEND!')

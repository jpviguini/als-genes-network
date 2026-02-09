import os
from gensim import models
from gensim.models import Word2Vec, FastText
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import re 


def list_from_txt(file_path):
    '''
    Creates a list of itens based on a .txt file, each line becomes an item.
    '''
    
    strings_list = []
    with open (file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            strings_list.append(line.rstrip('\n'))
    return strings_list

def score_func(model, analogies_als, number_of_analogies_als, analogies_grammar, number_of_analogies_grammar, analogies_biomedical, number_of_analogies_biomedical, topn):
    """ 
    Computes the performance of the model in each analogies categories.
    """
    
    number_of_analogies_all = number_of_analogies_als + number_of_analogies_grammar + number_of_analogies_biomedical

    number_of_correct_analogies_all = 0
    number_of_correct_analogies_als = 0
    number_of_correct_analogies_grammar = 0
    number_of_correct_analogies_biomedical = 0

    
    for analogie in analogies_als:
        words = analogie.split(' ')
        try:
            # verify if the words exist before running the analogy
            if not all(w in model.wv for w in words):
                continue
            for pair in model.wv.most_similar(positive=[model.wv[words[1]], model.wv[words[2]]], negative=[model.wv[words[0]]], topn=topn):
                if words[3] == pair[0]:
                    number_of_correct_analogies_all += 1
                    number_of_correct_analogies_als += 1
        except Exception:
            continue

    # loop for 'analogies_grammar'
    for analogie in analogies_grammar:
        words = analogie.split(' ')
        try:
            if not all(w in model.wv for w in words):
                continue
            for pair in model.wv.most_similar(positive=[model.wv[words[1]], model.wv[words[2]]], negative=[model.wv[words[0]]], topn=topn):
                if words[3] == pair[0]:
                    number_of_correct_analogies_all += 1
                    number_of_correct_analogies_grammar += 1
        except Exception:
            continue

    # loop for 'analogies_biomedical'
    for analogie in analogies_biomedical:
        words = analogie.split(' ')
        try:
            if not all(w in model.wv for w in words):
                continue
            for pair in model.wv.most_similar(positive=[model.wv[words[1]], model.wv[words[2]]], negative=[model.wv[words[0]]], topn=topn):
                if words[3] == pair[0]:
                    number_of_correct_analogies_all += 1
                    number_of_correct_analogies_biomedical += 1
        except Exception:
            continue
            
 
    score_all = (number_of_correct_analogies_all * 100) / number_of_analogies_all if number_of_analogies_all > 0 else 0
    score_als = (number_of_correct_analogies_als * 100) / number_of_analogies_als if number_of_analogies_als > 0 else 0
    score_grammar = (number_of_correct_analogies_grammar * 100) / number_of_analogies_grammar if number_of_analogies_grammar > 0 else 0
    score_biomedical = (number_of_correct_analogies_biomedical * 100) / number_of_analogies_biomedical if number_of_analogies_biomedical > 0 else 0

    return (
        score_all,
        score_als,
        score_grammar,
        score_biomedical
    )

def contains(string, unwanted_words):
    for x in string.split(' '):
        if x in unwanted_words:
            return True
    
    return False

def get_valid_analogies(model):
    
    analogies_als_path = '../data/analogies_als.txt'
    analogies_grammar_path = '../data/analogies_grammar.txt'
    analogies_biomedical_path = '../data/analogies_biomedical.txt'
    
    analogies_als = list_from_txt(analogies_als_path) if os.path.exists(analogies_als_path) else []
    analogies_grammar = list_from_txt(analogies_grammar_path) if os.path.exists(analogies_grammar_path) else []
    analogies_biomedical = list_from_txt(analogies_biomedical_path) if os.path.exists(analogies_biomedical_path) else []

    # analogies ALS filter
    analogie_words_present_in_model_vocab = set()
    remove_analogies_with_the_words = []
    for analogie in analogies_als:
        words = [x.lower() for x in analogie.split(' ')]
        if ':' in words: continue
        for w in words:
            if w not in analogie_words_present_in_model_vocab:
                if w in model.wv: 
                    analogie_words_present_in_model_vocab.add(w)
                else:
                    remove_analogies_with_the_words.append(w)
    analogies_als = [x for x in analogies_als if not contains(x.lower(), remove_analogies_with_the_words)]
    
    # analogies grammar filter
    analogie_words_present_in_model_vocab = set()
    remove_analogies_with_the_words = []
    for analogie in analogies_grammar:
        words = [x.lower() for x in analogie.split(' ')]
        if ':' in words: continue
        for w in words:
            if w not in analogie_words_present_in_model_vocab:
                if w in model.wv:
                    analogie_words_present_in_model_vocab.add(w)
                else:
                    remove_analogies_with_the_words.append(w)
    analogies_grammar = [x for x in analogies_grammar if not contains(x.lower(), remove_analogies_with_the_words)]

    # analogies biomedical filter
    analogie_words_present_in_model_vocab = set()
    remove_analogies_with_the_words = []
    for analogie in analogies_biomedical:
        words = [x.lower() for x in analogie.split(' ')]
        if ':' in words: continue
        for w in words:
            if w not in analogie_words_present_in_model_vocab:
                if w in model.wv:
                    analogie_words_present_in_model_vocab.add(w)
                else:
                    remove_analogies_with_the_words.append(w)
    analogies_biomedical = [x for x in analogies_biomedical if not contains(x.lower(), remove_analogies_with_the_words)]

    # removing duplicates:
    analogies_als = list(dict.fromkeys(analogies_als))
    analogies_grammar = list(dict.fromkeys(analogies_grammar))
    analogies_biomedical = list(dict.fromkeys(analogies_biomedical))
    
    return analogies_als, analogies_grammar, analogies_biomedical



# plot
def get_performance_bar_plot_from_df(df, colors):
    models_names = df['Model name'].to_list()
    performance = {
        'All': df['All'].to_list(),
        'ALS': df['ALS'].to_list(),
        'Grammar': df['Grammar'].to_list(),
        'Biomedical': df['Biomedical'].to_list(),
    }

    x = np.arange(len(models_names))  # label locations
    width = 0.2  # width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(25, 8))

    for attribute, measurement in performance.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[multiplier])
        multiplier += 1

    ax.set_ylabel('Score (%)')
    ax.set_xlabel('Models')
    ax.set_xticks(x + width, models_names)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', ncols=len(performance))

    return fig

if __name__ == '__main__':
    print('Starting')
    
    TESTING_DEFAULT_MODEL = True
    ANALYZING_FASTTEXT_MODELS = True # false --> word2vec
    
    if ANALYZING_FASTTEXT_MODELS:
      
        MODELS_PATH = '../fasttext_models'
        model_type = 'FastText'
    else:
      
        MODELS_PATH = './word2vec_models' 
        model_type = 'Word2Vec'
    
   
    if not os.path.exists(MODELS_PATH):
        print(f"ERROR: Model's folder not found: {MODELS_PATH}")
        print("Verify MODEL_PATH.")
        exit()
        
    MODELS = sorted([f.path for f in os.scandir(MODELS_PATH) if f.name.endswith('.model')])
    
    if not MODELS:
        print(f"ERROR: There is no .model file: {MODELS_PATH}")
        exit()
        
    TOPN_VALUES = [10, 20]
    COLORS = ['blue', 'mediumseagreen', 'orange', 'hotpink']
    
    # Output folder
    dat_str = date.today().strftime("%Y%m%d")
    output_dir = f'./hyperparameter_reports_ALS_{model_type}_{dat_str}/'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")
    
 
    data_models_files = { 'model name': [], 'filepath': [] }
    for index, model_name in enumerate(MODELS):
        
        clean_name = os.path.basename(model_name).lower()
        if 'default' in clean_name:
            data_models_files['model name'].append('Default')
            TESTING_DEFAULT_MODEL = True
        else:
            # extract hyperparameters from filename for model naming
            model_name_splitted = os.path.basename(clean_name).split('.model')[0].split('_')

            v, a, n = np.nan, np.nan, np.nan
            for part in model_name_splitted:
                if part.startswith('v') and part[1:].isdigit():
                    v = int(part[1:])
                elif part.startswith('a') and 'p' in part:   # a0p025
                    a = float(part[1:].replace('p', '.'))
                elif part.startswith('n') and part[1:].isdigit():
                    n = int(part[1:])

            model_label = f'{{{v}; {a}; {n}}}'
            data_models_files['model name'].append(model_label)

        data_models_files['filepath'].append(model_name)
        
    df_models_files = pd.DataFrame(data=data_models_files)
    df_models_files.to_csv(os.path.join(output_dir, 'models_files.csv'), index=False)

   
    # hyperparameter table
    data_optimization_hyperparameters = {
        'model name': [], 'vector size': [], 'learning rate': [], 'negative sampling': [],
    }


    for index, model_name in enumerate(MODELS):
        model_name_lower = model_name.lower()
        
        
        model_name_key = df_models_files[df_models_files['filepath'] == model_name]['model name'].values[0]
        data_optimization_hyperparameters['model name'].append(model_name_key)

        # try to extract the hyperparameters from the file name
        try:
    
            model_name_splitted = os.path.basename(model_name_lower).split('.model')[0].split('_')
            
            # search for parts that start with 'v', 'a', 'n' (vector, alpha, negative)
            v, a, n = np.nan, np.nan, np.nan
            
            for part in model_name_splitted:
                if part.startswith('v') and part[1:].isdigit():
                    v = float(part[1:])
                elif part.startswith('a') and re.match(r"a\d+p\d+", part): # 'a0p025'
                    v_str = part[1:].replace('p', '.')
                    a = float(v_str)
                elif part.startswith('a') and re.match(r"a\d+", part): # 'a0025' (a0025 = 0.025)
                
                    a = float(part[1:]) / (10** (len(part)-1) )
                elif part.startswith('n') and part[1:].isdigit():
                    n = float(part[1:])

    
            if 'default' in model_name_lower:
                v, a, n = np.nan, np.nan, np.nan

          
            data_optimization_hyperparameters['vector size'].append(v)
            data_optimization_hyperparameters['learning rate'].append(a)
            data_optimization_hyperparameters['negative sampling'].append(n)

        except Exception as e:
    
            print(f"Warning: Could not extract hyperparameter of {model_name}. Error: {e}")
            data_optimization_hyperparameters['vector size'].append(np.nan)
            data_optimization_hyperparameters['learning rate'].append(np.nan)
            data_optimization_hyperparameters['negative sampling'].append(np.nan)

    df_optimization_hyperparameters = pd.DataFrame(data=data_optimization_hyperparameters)
    
    df_optimization_hyperparameters.to_csv(os.path.join(output_dir, 'optimization_hyperparameters.csv'), index=False)


    # scores dictionary (ALS, grammar)
    models_scores = {model_name: {topn: {'All': 0, 'ALS': 0, 'Grammar': 0, 'Biomedical': 0} for topn in TOPN_VALUES} for model_name in MODELS}
    number_of_times_models_were_loaded = 0
    
    analogies_als, analogies_grammar, analogies_biomedical = [], [], []

    for topn in TOPN_VALUES:
        print(f'\n--- Starting Topn value: {topn} ---')
        for index, model_name in enumerate(MODELS):
            
            print(f'Loading model {index+1}/{len(MODELS)}: {os.path.basename(model_name)}')
            if ANALYZING_FASTTEXT_MODELS:
                model = FastText.load(model_name)
            else:
                model = Word2Vec.load(model_name)


            if number_of_times_models_were_loaded == 0:
                print('Getting valid analogies (only first time)')
                analogies_als, analogies_grammar, analogies_biomedical = get_valid_analogies(model)
                df_analogies = pd.DataFrame(data={
                    'Analogies': ['All', 'ALS', 'Grammar', 'Biomedical'],
                    'Amount': [
                        len(analogies_als)+len(analogies_grammar)+len(analogies_biomedical),
                        len(analogies_als),
                        len(analogies_grammar),
                        len(analogies_biomedical)
                    ]
                })
                df_analogies.to_csv(os.path.join(output_dir, 'analogies_summary.csv'), index=False)
                print(f"Valid analogies: All={len(analogies_als)+len(analogies_grammar)+len(analogies_biomedical)}, ALS={len(analogies_als)}, Grammar={len(analogies_grammar)}, Biomedical={len(analogies_biomedical)}")

            print(f'Computing score for model {index+1}/{len(MODELS)} (topn={topn})')
            model_score = score_func(model, analogies_als, len(analogies_als), analogies_grammar, len(analogies_grammar), analogies_biomedical, len(analogies_biomedical), topn=topn)

            models_scores[model_name][topn]['All'] = model_score[0]
            models_scores[model_name][topn]['ALS'] = model_score[1]
            models_scores[model_name][topn]['Grammar'] = model_score[2]
            models_scores[model_name][topn]['Biomedical'] = model_score[3]

            number_of_times_models_were_loaded += 1


    print('\nGenerating performance tables...')
    performance_tables = [] # list of dataframes
    for i, topn_val in enumerate(TOPN_VALUES):
       
        data = {
            'Model name': [], 'All': [], 'ALS': [], 'Grammar': [], 'Biomedical': [],
        }

        for model_path in MODELS:
          
            model_name_key = df_models_files[df_models_files['filepath'] == model_path]['model name'].values[0]
            
            score_dict = models_scores[model_path][topn_val]

            data['Model name'].append(model_name_key)
            data['All'].append(score_dict['All'])
            data['ALS'].append(score_dict['ALS'])
            data['Grammar'].append(score_dict['Grammar'])
            data['Biomedical'].append(score_dict['Biomedical'])
        
        df_perf = pd.DataFrame(data=data)
        performance_tables.append(df_perf)
        
      
        df_filename = os.path.join(output_dir, f'performance_topn_{topn_val}.csv')
        df_perf.to_csv(df_filename, index=False)
        print(f'Saved: {df_filename}')


    print('\nCreating and saving plots as PNG...')
    for i, df in enumerate(performance_tables):
        topn_val = TOPN_VALUES[i]
        fig = get_performance_bar_plot_from_df(df, COLORS)
        

        fig.suptitle(f'Performance of {model_type} models over the analogies, topn={topn_val}', fontsize=16)
        
        
        plot_filename = os.path.join(output_dir, f'performance_plot_topn_{topn_val}.pdf')
        fig.savefig(plot_filename, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved plot: {plot_filename}')

    print(f'\nAll CSV reports and figures (PNG) were saved in: {output_dir}')
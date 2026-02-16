import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- CONFIGURAÇÕES ---
UMBRELLA_TERM = "motor_neuron_disease"
MODEL_NAME = "pubmedbert"

# Caminhos (Ajuste se necessário)
FEATURES_PATH = f"./features_{MODEL_NAME}_{UMBRELLA_TERM}/features_ALS_{MODEL_NAME}.pkl"
OT_JSON = "OT-MONDO_0004976-associated-targets-2_12_2026-v25_12.json"
DATA_DIR = "../data/"

# A fonte independente que o supervisor pediu
TARGET_SOURCE = "eva" 
THRESHOLD = 0.5

# Passos de dimensionalidade para testar
# O input original é 1536 (768 gene + 768 disease). Vamos reduzindo pela metade.
DIMENSIONS_TO_TEST = [1536, 1024, 512, 256, 128, 64, 32, 16]

# Configuração da Validação (Bootstrap/Monte Carlo)
N_SPLITS = 50   # Quantas vezes vamos repetir o teste por dimensão
TEST_SIZE = 0.2 # 20% dos genes "escondidos" (witheld set)
SEED = 42

def load_ot_target(json_path: str, target_source: str) -> pd.DataFrame:
    """Carrega o JSON do OpenTargets e extrai o gabarito da source EVA"""
    print(f"[Info] Carregando Ground Truth de {target_source}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["gene"] = df["symbol"].astype(str).str.strip().str.upper()
    
    if target_source in df.columns:
        df = df[["gene", target_source]].copy()
        df[target_source] = pd.to_numeric(df[target_source], errors="coerce").fillna(0)
        # Binariza: 1 se score >= 0.5, senão 0
        df["target"] = (df[target_source] >= THRESHOLD).astype(int)
    else:
        raise ValueError(f"Source '{target_source}' não encontrada no JSON.")
        
    return df[["gene", "target"]].groupby("gene", as_index=False).max()

def prepare_data(features_path: str, df_truth: pd.DataFrame):
    """
    1. Carrega os vetores (Bags)
    2. Faz 'Mean Pooling' (Média dos vetores) para ter 1 vetor por gene
    3. Cruza com o gabarito (y)
    """
    print("[Info] Carregando features extraídas...")
    with open(features_path, "rb") as f:
        gene_bags = pickle.load(f)
    
    X_list = []
    y_list = []
    genes_list = []
    
    # Dicionário rápido para lookup do target
    truth_dict = dict(zip(df_truth["gene"], df_truth["target"]))
    
    for gene, vectors in gene_bags.items():
        if len(vectors) == 0:
            continue
            
        # Passo Crucial: Como o supervisor pediu "Logistic Model", 
        # precisamos de 1 vetor fixo por gene.
        # Tiramos a média dos vetores da bag (Mean Pooling).
        # Isso simula o embedding "final" do gene.
        mean_vector = np.mean(np.stack(vectors), axis=0)
        
        # Define o label (1 se está no EVA, 0 caso contrário/desconhecido)
        label = truth_dict.get(gene, 0)
        
        X_list.append(mean_vector)
        y_list.append(label)
        genes_list.append(gene)
        
    return np.array(X_list), np.array(y_list), genes_list

def evaluate_dimensionality(X, y, dims):
    """
    Loop principal sugerido pelo supervisor:
    1. Reduz dimensão (PCA)
    2. Repete N vezes (Split -> Train LR -> Test AUC)
    """
    results = []
    
    # Normaliza antes do PCA (Boa prática)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    total_features = X.shape[1]
    
    print(f"\n[Info] Iniciando análise de complexidade.")
    print(f"Total Genes: {X.shape[0]} | Positivos ({TARGET_SOURCE}): {sum(y)}")
    
    for dim in tqdm(dims, desc="Testando Dimensionalidades"):
        current_X = X_scaled
        
        # 1. Redução de Dimensionalidade (Steps of two-folds)
        if dim < total_features:
            pca = PCA(n_components=dim, random_state=SEED)
            current_X = pca.fit_transform(X_scaled)
        
        # 2. Validação Robusta (Repeated Sub-sampling)
        cv = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)
        
        dim_aucs = []
        
        for train_idx, test_idx in cv.split(current_X, y):
            X_train, X_test = current_X[train_idx], current_X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 3. Treina Logistic Model (Rápido e Elegante)
            # class_weight='balanced' ajuda pois temos poucos positivos
            clf = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
            clf.fit(X_train, y_train)
            
            # 4. Computa AUC no conjunto "Withheld" (Teste)
            y_probs = clf.predict_proba(X_test)[:, 1]
            
            try:
                score = roc_auc_score(y_test, y_probs)
                dim_aucs.append(score)
            except ValueError:
                pass # Ignora fold se der erro (ex: só uma classe no teste)
        
        # Salva a média e desvio padrão para essa dimensão
        mean_auc = np.mean(dim_aucs)
        std_auc = np.std(dim_aucs)
        
        results.append({
            "dimension": dim,
            "mean_auc": mean_auc,
            "std_auc": std_auc
        })
        
    return pd.DataFrame(results)

def plot_complexity_curve(df_results):
    plt.figure(figsize=(10, 6))
    
    # Plot linha com erro sombreado
    plt.plot(df_results["dimension"], df_results["mean_auc"], marker='o', color='navy', label='Mean AUC')
    plt.fill_between(
        df_results["dimension"], 
        df_results["mean_auc"] - df_results["std_auc"], 
        df_results["mean_auc"] + df_results["std_auc"], 
        color='navy', alpha=0.2, label='Std Dev'
    )
    
    plt.xscale('log') # Escala logarítmica fica melhor para potências de 2
    plt.xlabel('Model Dimensionality (Log Scale)', fontsize=12)
    plt.ylabel(f'ROC-AUC on {TARGET_SOURCE} (Independent Set)', fontsize=12)
    plt.title(f'Model Complexity vs. Performance\nTarget: {TARGET_SOURCE}', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Anotação de Dimensões
    for i, row in df_results.iterrows():
        plt.annotate(f"{int(row['dimension'])}", 
                     (row['dimension'], row['mean_auc']),
                     textcoords="offset points", xytext=(0,10), ha='center')

    plt.legend()
    plt.tight_layout()
    output_file = f"{DATA_DIR}complexity_analysis_{TARGET_SOURCE}.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n[Info] Gráfico salvo em: {output_file}")
    plt.show()

if __name__ == "__main__":
    # 1. Carregar Gabarito
    df_truth = load_ot_target(OT_JSON, TARGET_SOURCE)
    
    # 2. Carregar e Preparar Dados (Mean Pooling)
    X, y, _ = prepare_data(FEATURES_PATH, df_truth)
    
    # 3. Rodar Loop de Dimensionalidade
    df_results = evaluate_dimensionality(X, y, DIMENSIONS_TO_TEST)
    
    # 4. Mostrar Tabela e Plotar
    print("\n=== Resultados da Análise de Complexidade ===")
    print(df_results.sort_values("dimension", ascending=False).to_string(index=False))
    
    plot_complexity_curve(df_results)
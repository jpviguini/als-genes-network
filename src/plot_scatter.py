import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score
from config import VALIDATION_GENES

# Tenta importar a biblioteca de ajuste de texto. Se não tiver, avisa.
try:
    from adjustText import adjust_text
except ImportError:
    print("AVISO: Biblioteca 'adjustText' não encontrada. As labels podem ficar sobrepostas.")
    print("Instale usando: pip install adjustText")
    adjust_text = None

# --- CONFIGURAÇÕES ---
# Caminhos dos modelos
PATH_ALS = "./scores_MIL_t_pubmedbert_motor_neuron_disease/scores_final_allgenes.npz"
PATH_NEURODEG = "./scores_MIL_t_pubmedbert_neurodegenerative_disease/scores_final_allgenes.npz"
# PATH_NEUROMUSC = "./scores_MIL_t_pubmedbert_neuromuscular_disease/scores_final_allgenes.npz"

OT_JSON = "OT-MONDO_0004976-associated-targets-2_12_2026-v25_12.json"
DATA_DIR = "../data/"
VALIDATION_SOURCE = 'eva' 

def load_scores(npz_path: str, col_name: str) -> pd.DataFrame:
    try:
        d = np.load(npz_path, allow_pickle=True)
        scores = d["scores_topm"].astype(float) # Já são probabilidades
        df = pd.DataFrame({"gene": d["genes"].astype(str), col_name: scores})
        df["gene"] = df["gene"].str.strip().str.upper()
        return df.groupby("gene", as_index=False)[col_name].max()
    except FileNotFoundError:
        print(f"[Aviso] Arquivo não encontrado: {npz_path}")
        return pd.DataFrame(columns=["gene", col_name])

def load_ot_target(json_path: str, target_source: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["gene"] = df["symbol"].astype(str).str.strip().str.upper()
    
    if target_source in df.columns:
        df = df[["gene", target_source]].copy()
        df[target_source] = pd.to_numeric(df[target_source], errors="coerce").fillna(0)
    else:
        print(f"[Erro] Source '{target_source}' não existe no JSON.")
        return pd.DataFrame()
        
    return df.groupby("gene", as_index=False).max()

def plot_specificity_scatter(df, x_col, y_col, hue_col):
    """
    X-axis: Modelo Específico (ALS)
    Y-axis: Modelo Genérico (NeuroDeg)
    Cor: Se é validado na source (ex: EVA >= 0.5)
    """
    # Aumentei um pouco o tamanho para dar respiro aos textos
    plt.figure(figsize=(10, 10))
    
    # 1. Background: Todos os genes
    plt.scatter(df[x_col], df[y_col], c='#e0e0e0', alpha=0.4, s=15, label='Other Genes', edgecolors='none')
    
    # 2. Highlight: Genes validados na source
    positives = df[df[hue_col] >= 0.5]
    
    texts_to_adjust = [] # Lista para guardar os objetos de texto
    
    if not positives.empty:
        plt.scatter(positives[x_col], positives[y_col], c='#d62728', alpha=0.8, s=40, 
                    label=f'Validated in {hue_col}', edgecolors='white', linewidth=0.5)
        
        # Lógica de seleção de Labels:
        # Prioriza genes onde (Score X > Score Y) E (Score X é alto)
        # Calcula a distância até a diagonal (quanto maior, mais "específico" é o gene)
        positives = positives.copy()
        positives['diff'] = positives[x_col] - positives[y_col]
        
        # Filtra para rotular apenas os mais interessantes (diferença positiva relevante ou ambos muito altos)
        # Critério 1: Score no modelo específico > genérico + margem
        # Critério 2: Ambos muito altos (top hits consensuais)
        interesting = positives[
            ((positives['diff'] > 0.1) & (positives[x_col] > 0.5)) | 
            ((positives[x_col] > 0.8) & (positives[y_col] > 0.8))
        ]
        
        # Limita a 25 labels para não poluir demais, priorizando os maiores scores no modelo específico
        top_interesting = interesting.sort_values(x_col, ascending=False)
        
        for _, row in top_interesting.iterrows():
            t = plt.text(row[x_col], row[y_col], row['gene'], 
                         fontsize=9, fontweight='bold', color='#333333')
            texts_to_adjust.append(t)

    # Linha Diagonal (x=y)
    plt.plot([0, 1], [0, 1], ls="--", c="black", alpha=0.5, label="Equal Performance")
    
    # Região de Interesse (Texto fixo no gráfico)
    plt.text(0.85, 0.15, "Specific Signal\n(Better in ALS Model)", 
             ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.xlabel(f"Prob. {x_col} (Specific)", fontsize=12)
    plt.ylabel(f"Prob. {y_col} (Generic)", fontsize=12)
    plt.title(f"Model Specificity Check\nTarget Source: {hue_col}", fontsize=14)
    plt.legend(loc="upper left")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    

    if adjust_text and texts_to_adjust:
        print("Ajustando posição das labels (pode demorar alguns segundos)...")
        adjust_text(texts_to_adjust, 
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                    expand_points=(1.2, 1.2)) # Força empurrar mais para longe dos pontos
    
    out_name = f"{DATA_DIR}scatter_{x_col}_vs_{y_col}_by_{hue_col}.png"
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)
    plt.close()
    print(f"[Plot] Salvo: {out_name}")

if __name__ == "__main__":
    # 1. Carregar Dados
    print("Carregando probabilidades...")
    df_als = load_scores(PATH_ALS, "ALS_Model")
    df_neuro = load_scores(PATH_NEURODEG, "NeuroDeg_Model")
    ot = load_ot_target(OT_JSON, VALIDATION_SOURCE)
    
    df_full = df_als.merge(ot, on="gene", how="inner")
    df_full = df_full.merge(df_neuro, on="gene", how="left").fillna(0)
    
    # 2. REMOVER DATA LEAKAGE
    train_genes = set([str(g).strip().upper() for g in VALIDATION_GENES])
    df_eval = df_full[~df_full["gene"].isin(train_genes)].copy()
    
    print(f"Genes para validação (Hold-out): {len(df_eval)}")

    # 3. Gerar Scatter Plot
    if "NeuroDeg_Model" in df_eval.columns:
        plot_specificity_scatter(df_eval, "ALS_Model", "NeuroDeg_Model", VALIDATION_SOURCE)
    else:
        print("Modelo NeuroDeg não encontrado ou vazio.")

    # 4. Estatísticas
    positives = df_eval[df_eval[VALIDATION_SOURCE] >= 0.5]
    if not positives.empty:
        mean_als = positives["ALS_Model"].mean()
        mean_neuro = positives["NeuroDeg_Model"].mean()
        print("\n=== Estatísticas nos Genes Validados (Positivos) ===")
        print(f"Média Score ALS Model: {mean_als:.4f}")
        print(f"Média Score NeuroDeg Model: {mean_neuro:.4f}")
    else:
        print("Nenhum positivo encontrado para calcular médias.")
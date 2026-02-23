import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score

# Importa a lista de genes usados no treino para removê-los da validação
from config import VALIDATION_GENES

OT_JSON = "OT-MONDO_0004976-associated-targets-2_12_2026-v25_12.json"
UMBRELLA_TERM = "neurodegenerative_disease"
GENE_UNIVERSE_CSV = f"../data/genes_extracted_{UMBRELLA_TERM}_with_freq.csv"
MIL_NPZ = f"./1:4_scores_LR_pubmedbert_{UMBRELLA_TERM}/scores_final_allgenes.npz"
DATA_DIR = "../data/"

THRESHOLD = 0.5

# Sources que servirão como Ground Truth (alvos)
GT_SOURCES = ['eva', 'chembl', 'clingen', 'crispr', 'crisprScreen', 'expressionAtlas', 
              'geneBurden', 'gene2Phenotype', 'genomicsEngland', 'impc', 'orphanet', 
              'gwasCredibleSets', 'reactome', 'uniprotLiterature', 'uniprotVariants']

# Modelos que tentaremos usar para recuperar esses genes
PREDICTOR_SOURCES = ['mil_score', 'europepmc']

def load_ot_wide(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["gene"] = df["symbol"].astype(str).str.strip().str.upper()
    # Garante que as colunas existam mesmo se não estiverem no JSON
    existing_cols = [c for c in df.columns if c in GT_SOURCES + ['europepmc']]
    cols_to_keep = ["gene"] + existing_cols
    df = df[cols_to_keep]
    for c in df.columns:
        if c == "gene": continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.groupby("gene", as_index=False).max()

def load_mil_scores(npz_path: str) -> pd.DataFrame:
    d = np.load(npz_path, allow_pickle=True)
    df = pd.DataFrame({"gene": d["genes"].astype(str), "mil_score": d["scores_topm"].astype(float)})
    df["gene"] = df["gene"].str.strip().str.upper()
    return df.groupby("gene", as_index=False)["mil_score"].max()

def plot_auc_comparison(results_list, metric_name, filename):
    df_res = pd.DataFrame(results_list)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_res, x="gt_source", y=metric_name, hue="predictor", palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.title(f"[{UMBRELLA_TERM}] Generalization Performance (Gold genes excluded)\nSource score >= {THRESHOLD}")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}_1:4_{filename}.png", dpi=300)
    plt.close()

def plot_top_genes_heatmap(df_eval, top_n=50):
    # Seleciona top genes pelo BERT (Excluindo genes de treino)
    top_genes = df_eval.sort_values("mil_score", ascending=False).head(top_n)
    
    # Prepara matriz para o heatmap
    # Filtra colunas que realmente existem no df
    cols_to_plot = [c for c in GT_SOURCES + ['europepmc', 'mil_score'] if c in df_eval.columns]
    
    heatmap_data = top_genes.set_index("gene")[cols_to_plot]
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(heatmap_data, annot=False, cmap="YlGnBu", cbar_kws={'label': 'Score'})
    plt.title(f"[{UMBRELLA_TERM}] Top {top_n} Novel Predictions (Train Genes Excluded)\nBERT Score vs Evidence Streams")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}heatmap_top_{top_n}_novel_genes_BERT_{UMBRELLA_TERM}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # 1. Carga de dados
    print("Carregando dados...")
    ot = load_ot_wide(OT_JSON)
    mil = load_mil_scores(MIL_NPZ)
    
    # Merge inicial
    df_full = ot.merge(mil, on="gene", how="inner")
    print(f"Total de genes (Interseção OT + Modelo): {len(df_full)}")

    # 2. REMOÇÃO DOS GENES DE TREINO (VALIDATION_GENES)
    train_genes = set([str(g).strip().upper() for g in VALIDATION_GENES])
    
    # Cria o dataframe de avaliação (df_eval) excluindo o treino
    df_eval = df_full[~df_full["gene"].isin(train_genes)].copy()
    
    n_removed = len(df_full) - len(df_eval)
    print(f"Genes de treino removidos da análise: {n_removed}")
    print(f"Genes restantes para validação (Hold-out): {len(df_eval)}")

    # 3. Loop de Avaliação
    all_metrics = []
    
    print(f"\nCalculando métricas (Hold-out validation) threshold {THRESHOLD}...")
    
    for gt_src in GT_SOURCES:
        if gt_src not in df_eval.columns: 
            print(f"[Aviso] Source {gt_src} não encontrada no JSON.")
            continue
        
        # Define quem são os positivos para ESTA source NO CONJUNTO DE TESTE
        y_true = (df_eval[gt_src] >= THRESHOLD).astype(int)
        
        n_pos = y_true.sum()
        # Só calcula se houver genes positivos suficientes FORA do treino
        if n_pos < 5: 
            print(f"PULANDO {gt_src}: Apenas {n_pos} genes positivos restantes após remover treino.")
            continue
            
        for pred_src in PREDICTOR_SOURCES:
            if pred_src not in df_eval.columns: continue

            y_score = df_eval[pred_src]
            
            try:
                roc = roc_auc_score(y_true, y_score)
                pr = average_precision_score(y_true, y_score)
                
                all_metrics.append({
                    "gt_source": gt_src,
                    "predictor": "Our BERT" if pred_src == 'mil_score' else "EuropePMC",
                    "roc_auc": roc,
                    "pr_auc": pr,
                    "n_positives": n_pos
                })
            except ValueError as e:
                print(f"Erro ao calcular métrica para {gt_src}: {e}")
                continue

    # 4. Gerar Gráficos e Resumos
    if all_metrics:
        # Salva gráficos
        plot_auc_comparison(all_metrics, "pr_auc", f"PR_AUC_holdout_validation_{UMBRELLA_TERM}_LR")
        plot_auc_comparison(all_metrics, "roc_auc", f"ROC_AUC_holdout_validation_{UMBRELLA_TERM}_LR")
        print("\n[info] Gráficos de barra (Hold-out) salvos.")

        # Resumo Tabela
        res_df = pd.DataFrame(all_metrics)
        summary = res_df.pivot(index="gt_source", columns="predictor", values="roc_auc")
        print("\n=== (ROC-AUC) - GENERALIZATION TASK ===")
        print(summary)
        
        # Salva CSV com métricas brutas
        res_df.to_csv(f"{DATA_DIR}metrics_holdout_{UMBRELLA_TERM}.csv", index=False)
    else:
        print("\n[Aviso] Nenhuma métrica foi calculada. Verifique se sobraram positivos após a filtragem.")


    print("\n=== Quantidade de Gold Genes por Evidence Stream ===")
    for gt_src in GT_SOURCES:
        if gt_src not in df_eval.columns:
            print(f"{gt_src}: não encontrada")
            continue
        n_gold = (df_eval[gt_src] >= THRESHOLD).sum()
        print(f"{gt_src}: {n_gold} gold genes")


    # 5. Gerar Heatmap (apenas dos genes de teste)
    plot_top_genes_heatmap(df_eval, top_n=50)
    print("[info] Heatmap dos top 50 genes 'novos' (não treino) salvo.")
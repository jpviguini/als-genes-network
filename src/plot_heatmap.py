import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

file_path = 'OT-MONDO_0004976-associated-targets-2_12_2026-v25_12.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    print("\n--- Colunas disponíveis no arquivo JSON ---")
    print(df.columns.tolist())
    print("-------------------------------------------\n")
    
except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    exit()


# evidence_columns = [
#     'cancerGeneCensus', 'intogen', 'evaSomatic', 'cancerBiomarkers',
#     'chembl', 'crisprScreen', 'crispr', 'reactome',
#     'europepmc', 'expressionAtlas', 'impc', 'globalScore'
# ]

evidence_columns = ['globalScore','eva','chembl','clingen','crispr','crisprScreen','europepmc','expressionAtlas','geneBurden','gene2phenotype','genomicsEngland','impc','orphanet','gwasCredibleSets','reactome','uniprotLiterature','uniprotVariants']





available_columns = [col for col in evidence_columns if col in df.columns]


df_plot = df.set_index('symbol')


if 'globalScore' in df_plot.columns:
    df_plot = df_plot.sort_values('globalScore', ascending=False)

df_plot = df_plot[available_columns]

df_plot = df_plot.replace('No data', 0)
for col in df_plot.columns:
    df_plot[col] = pd.to_numeric(df_plot[col])

# Plot 1: Top 50 Genes
plt.figure(figsize=(12, 10))
sns.heatmap(df_plot.head(50), cmap='viridis', linewidths=.5, annot=False)
plt.title('Heatmap of evidence scores (Top 50 Genes)')
plt.xlabel('Source')
plt.ylabel('Gene')
plt.tight_layout()
plt.savefig('../data/heatmap_top50_genes.png', dpi=300)
print("Salvo: heatmap_top50_genes.png")

# # --- Plot 2: Todos os Genes (Visão Geral) ---
# plt.figure(figsize=(10, 20))
# sns.heatmap(df_plot, cmap='viridis', yticklabels=False)
# plt.title('Heatmap de Scores de Evidência (Todos os Genes)')
# plt.xlabel('Fonte de Evidência')
# plt.ylabel('Genes (ordenados por Global Score)')
# plt.tight_layout()
# plt.savefig('../data/heatmap_all_genes.png', dpi=300)
# print("Salvo: heatmap_all_genes.png")
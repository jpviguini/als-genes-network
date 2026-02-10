import pandas as pd
import matplotlib.pyplot as plt

umbrella_term = "motor_neuron_disease"

csv_path = f"../data/genes_extracted_{umbrella_term}_with_freq.csv"
top_n = 100   # quantos genes mostrar


df = pd.read_csv(csv_path)

print("Loaded file:", csv_path)
print("Total genes:", len(df))

# ordenar pelos mais frequentes
df = df.sort_values("count", ascending=False)

# pegar top N genes
df_top = df.head(top_n)

print("\nTop genes:")
print(df_top)


plt.figure(figsize=(12,6))
plt.bar(df_top["gene"], df_top["count"])

plt.xticks(rotation=90)
plt.xlabel("Gene")
plt.ylabel("Frequency in corpus")
plt.title(f"Top {top_n} Most Frequent Genes (umbrella term: {umbrella_term}). Total genes: {len(df)}")

plt.tight_layout()
plt.show()
plt.savefig(f"../data/hist_top{top_n}_genes_{umbrella_term}.png", dpi=300)

print(f"Plot saved successfully: ../data/hist_top{top_n}_genes_{umbrella_term}.png")

import pandas as pd
import matplotlib.pyplot as plt


csv_path = "../data/genes_extracted_neurodegenerative_disease_with_freq.csv"
top_n = 30   # quantos genes mostrar


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
plt.title(f"Top {top_n} Most Frequent Genes")

plt.tight_layout()
plt.show()

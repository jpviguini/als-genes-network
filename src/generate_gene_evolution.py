import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import numpy as np
from adjustText import adjust_text

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "lines.linewidth": 2,
    "lines.markersize": 6
})

MODELS = ["w2v"]  # word2vec
BASE_DIR = "./validation/per_gene/"
OUTPUT_DIR = "./validation/plots/"
COLUMN = "dot_minmax"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GENE_DISCOVERY_YEAR = {  # source: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1170996/full
    'SPTLC1': 2021, 'WDR7': 2020, 'CAV1': 2020,
    'GLT8D1': 2019, 'ARPP21': 2019, 'DNAJC7': 2019,
    'KIF5A': 2018, 'TIA1': 2017, 'ANXA11': 2017,
    'CCNF': 2016, 'NEK1': 2016, 'C21ORF2': 2016,
    'TBK1': 2015, 'CHCHD10': 2014, 'MATR3': 2014,
    'TUBA4A': 2014, 'ERBB4': 2013, 'HNRNPA2B1': 2013,
    'HNRNPA1': 2013, 'ATXN1': 2012, 'EPHA4': 2012,
    'PFN1': 2012, 'C9ORF72': 2011, 'SQSTM1': 2011,
    'UBQLN2': 2011, 'SIGMAR1': 2010, 'ATXN2': 2010,
    'OPTN': 2010, 'SPG11': 2010, 'VCP': 2010,
    'ANG': 2006, 'FIG4': 2009, 'UNC13A': 2009,
    'ELP3': 2009, 'FUS': 2009, 'TARDBP': 2008,
    'CHMP2B': 2006, 'HFE': 2004, 'VAPB': 2004,
    'DCTN1': 2003, 'ALS2': 2001, 'SETX': 1998,
    'NEFH': 1994, 'SOD1': 1993
}

WINDOW_SIZE = 3
X_THRESHOLD = 0.02
T_THRESHOLD = 0.04


def detect_latent_knowledge(df):
    # detects year in which the notification could be released
    years = df["year"].values
    values = df[COLUMN].values
    n = len(values)

    for i in range(n - WINDOW_SIZE + 1):
        window_years = years[i:i + WINDOW_SIZE]
        window_values = values[i:i + WINDOW_SIZE]

        derivadas = np.diff(window_values)
        derivada_media = np.mean(derivadas)
        crit1 = derivada_media >= X_THRESHOLD

        crit2 = (window_values[-1] - window_values[0]) >= T_THRESHOLD

        if crit1 or crit2:
            return int(window_years[-1] + 1)

    return None


def _prepare_df_for_detection(df, gene):
    # keeps the same filtering logic used in per-gene plots
    df = df.copy()
    if "dot" in df.columns:
        df = df[df["dot"] > 0]
    discovery_year = GENE_DISCOVERY_YEAR.get(gene.upper())
    if discovery_year is not None:
        df = df[df["year"] <= discovery_year]
    return df

def plot_gene(df, gene, model_name, column):
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(7, 3.5))

    df = _prepare_df_for_detection(df, gene)
    if df.empty:
        return None

    discovery_year = GENE_DISCOVERY_YEAR.get(gene.upper())
    notification_year = detect_latent_knowledge(df)

    color_line = "#2b4b7c"
    color_discovery = "#a83232"
    color_notif = "#2ca02c"

    # plot line
    ax.plot(df["year"], df[column], marker="o", markersize=5,
            linewidth=2, color=color_line, label="semantic similarity", zorder=2)

    # vertical line of discovery year
    if discovery_year:
        ax.axvline(x=discovery_year, color=color_discovery, linestyle="--",
                   linewidth=1.5, alpha=0.8, zorder=1)

    # notification star
    if notification_year:
        notif_vals = df[df["year"] == notification_year][column].values
        if len(notif_vals) > 0:
            val = float(notif_vals[0])
            ax.scatter(notification_year, val, color="white", edgecolors=color_notif,
                       s=150, marker='*', linewidth=1.5, zorder=5)
            ax.scatter(notification_year, val, color=color_notif,
                       s=60, marker='*', zorder=6, label=f"signal ({notification_year})")

    ax.set_title(f"Discovery Trajectory: {gene.upper()}", fontweight='bold', pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("similarity score")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    sns.despine(trim=True, offset=5)

    # caption config
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    
    
    if discovery_year:
        line_disc = Line2D([0], [0], color=color_discovery, linestyle='--',
                           label=f"official discovery ({discovery_year})")
        handles.append(line_disc)


    ax.legend(handles=handles, loc='best', 
              frameon=True, framealpha=0.9, edgecolor='white', 
              fontsize=9)
    
    plt.tight_layout()

    filename_png = f"{gene}_{model_name}_{column}.png"
    filename_pdf = f"{gene}_{model_name}_{column}.pdf"

    # saves in png and pdf
    plt.savefig(os.path.join(OUTPUT_DIR, filename_png), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, filename_pdf), format='pdf', bbox_inches='tight')
    plt.close()
    return filename_png

def build_summary_rows_for_model(model_name):
    folder = os.path.join(BASE_DIR, model_name)
    if not os.path.exists(folder):
        print(f"folder {folder} not found.")
        return []

    rows = []
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    for csv_file in csv_files:
        gene = csv_file.replace(".csv", "").upper()
        if gene not in GENE_DISCOVERY_YEAR:
            continue

        file_path = os.path.join(folder, csv_file)
        try:
            df = pd.read_csv(file_path)
            if COLUMN not in df.columns or "year" not in df.columns:
                continue

            df2 = _prepare_df_for_detection(df, gene)
            if df2.empty:
                rows.append({
                    "model": model_name.upper(),
                    "gene": gene,
                    "discovery_year": int(GENE_DISCOVERY_YEAR[gene]),
                    "signal_year": np.nan,
                    "delta_years": np.nan
                })
                continue

            signal_year = detect_latent_knowledge(df2)
            discovery_year = int(GENE_DISCOVERY_YEAR[gene])
            delta = (discovery_year - signal_year) if signal_year is not None else np.nan

            rows.append({
                "model": model_name.upper(),
                "gene": gene,
                "discovery_year": discovery_year,
                "signal_year": float(signal_year) if signal_year is not None else np.nan,
                "delta_years": float(delta) if signal_year is not None else np.nan
            })
        except Exception as e:
            print(f"error when processing {gene} ({model_name}): {e}")

    return rows

def plot_summary(df_summary, annotate="all"): 
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(9, 6)) 

    df_plot = df_summary.dropna(subset=["signal_year"]).copy()
    if df_plot.empty:
        print("no signal years found. summary plot skipped.")
        return

    df_plot["discovery_year"] = df_plot["discovery_year"].astype(int)
    df_plot["signal_year"] = df_plot["signal_year"].astype(int)
    
    
    min_year = int(min(df_plot["discovery_year"].min(), df_plot["signal_year"].min())) - 1
    max_year = int(max(df_plot["discovery_year"].max(), df_plot["signal_year"].max())) + 2

    ax.plot([min_year, max_year], [min_year, max_year],
            linestyle="--", linewidth=1.5, color="gray", alpha=0.5, zorder=1)

    
    sns.scatterplot(
        data=df_plot,
        x="discovery_year",
        y="signal_year",
        hue="model",
        palette={"W2V": "#2b4b7c", "FT": "#6a3d9a"},
        s=100, 
        alpha=0.8,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
        zorder=2,
        legend=False
    )
    

    df_grouped = df_plot.groupby(["discovery_year", "signal_year"])["gene"].apply(
        lambda genes: "\n".join(genes)
    ).reset_index()


    texts = []
    for _, r in df_grouped.iterrows():
 
        texts.append(ax.text(r["discovery_year"], r["signal_year"], r["gene"], 
                             fontsize=8, color='#333333', 
                             linespacing=1.1)) 

  
    ax.set_xlim(min_year, max_year)
    ax.set_ylim(min_year, max_year)
    ax.set_xlabel("Official Discovery Year")
    ax.set_ylabel("Model Notification Signal")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='both', linestyle=':', alpha=0.3)
    sns.despine(trim=True, offset=5)
    # ax.legend(title="", frameon=False, loc="upper left")


    from adjustText import adjust_text
    adjust_text(texts, 
                x=df_grouped["discovery_year"].values, 
                y=df_grouped["signal_year"].values,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                ax=ax)

    plt.tight_layout()

    out_pdf = os.path.join(OUTPUT_DIR, "temporal_summary_stacked.pdf")
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[info] summary plot saved (stacked): {out_pdf}")



def process_model(model_name):
    folder = os.path.join(BASE_DIR, model_name)
    if not os.path.exists(folder):
        print(f"folder {folder} not found.")
        return

    print(f"generating per-gene plots for {model_name.upper()}...")
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    for csv_file in tqdm(csv_files, desc=f"{model_name} per-gene"):
        gene = csv_file.replace(".csv", "")


        if gene.upper() not in GENE_DISCOVERY_YEAR:
            continue

        file_path = os.path.join(folder, csv_file)
        try:
            df = pd.read_csv(file_path)
            if COLUMN not in df.columns or "year" not in df.columns:
                continue
            plot_gene(df, gene, model_name.upper(), COLUMN)
        except Exception as e:
            print(f"error when processing {gene}: {e}")


def main():
    # per-gene plots
    for model in MODELS:
        process_model(model)

    # summary plot
    all_rows = []
    for model in MODELS:
        all_rows.extend(build_summary_rows_for_model(model))

    df_summary = pd.DataFrame(all_rows)
    df_summary_path = os.path.join(OUTPUT_DIR, "temporal_summary_df.csv")
    df_summary.to_csv(df_summary_path, index=False)
    print(f"[info] summary dataframe saved: {df_summary_path}")

    plot_summary(df_summary, annotate="all")

    print(f"\nvisualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

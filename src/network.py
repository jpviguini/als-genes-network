import os
import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Supondo que você tenha isso no seu config
from config import VALIDATION_GENES 

umbrella_term = "neurodegenerative_disease"
MODEL_NAME = "pubmedbert"
FEATURES_PATH = f"./features_{MODEL_NAME}_{umbrella_term}/features_ALS_{MODEL_NAME}.pkl"
GENES_CSV_PATH = "../data/genes_extracted_neurodegenerative_disease.csv" 

# --- CAMINHOS DOS ARQUIVOS LOCAIS DO STRING DB ---
STRING_LINKS_PATH = "../data/9606.protein.links.v12.0.txt"     
STRING_ALIASES_PATH = "../data/9606.protein.aliases.v12.0.txt" 

OUT_DIR = f"./pageranked_{MODEL_NAME}_{umbrella_term}/"
CV_METRICS_JSON = os.path.join(OUT_DIR, "cv_metrics.json")
CV_OOF_GOLD_NPZ = os.path.join(OUT_DIR, "scores_oof_gold_only.npz")
FINAL_ALLGENES_NPZ = os.path.join(OUT_DIR, "scores_final_allgenes.npz")

N_FOLDS = 5
SEED = 42
NEG_RATIO = 4
RELIABLE_NEG_MAX_BAG = 10
MAX_INST_TRAIN = 500

def set_seed(seed: int):
    np.random.seed(seed)

def subsample(vectors: List[np.ndarray], max_inst: int) -> List[np.ndarray]:
    if len(vectors) <= max_inst:
        return vectors
    idx = np.random.choice(len(vectors), max_inst, replace=False)
    return [vectors[i] for i in idx]

def get_reliable_negatives(gene_dict, exclude, n_needed, max_bag=5):
    low = [g for g, v in gene_dict.items() if (g not in exclude and 0 < len(v) < max_bag)]
    if len(low) >= n_needed:
        return list(np.random.choice(low, size=n_needed, replace=False))
    remaining = [g for g, v in gene_dict.items() if (g not in exclude and g not in set(low) and len(v) > 0)]
    need = n_needed - len(low)
    if need > 0:
        if need >= len(remaining):
            low.extend(remaining)
        else:
            low.extend(list(np.random.choice(remaining, size=need, replace=False)))
    return low

# --- FUNÇÃO ATUALIZADA: Mapeamento Seguro ---
def load_local_string_graph(links_path: str, aliases_path: str, valid_genes: Set[str], min_score: int = 700) -> nx.Graph:
    print(f"\n[info] Carregando rede STRING local (score mínimo: {min_score})...")
    
    df_links = pd.read_csv(links_path, sep=" ")
    df_filtered = df_links[df_links["combined_score"] >= min_score].copy()
    print(f"[info] Interações de alta confiança mantidas: {len(df_filtered)}")

    print("[info] Mapeando IDs para Gene Symbols (Modo Seguro/Forçado)...")
    df_aliases = pd.read_csv(aliases_path, sep='\t')

    # Padroniza nomes
    df_aliases['alias'] = df_aliases['alias'].astype(str).str.strip().str.upper()

    # O PULO DO GATO: Filtra os aliases para manter APENAS os que nos interessam
    df_symbols = df_aliases[df_aliases['alias'].isin(valid_genes)].copy()

    col_id = '#string_protein_id' if '#string_protein_id' in df_symbols.columns else 'string_protein_id'
    
    # Remove duplicatas mantendo a primeira ocorrencia para evitar bugs no dicionario
    df_symbols = df_symbols.drop_duplicates(subset=[col_id])
    
    id_to_gene = dict(zip(df_symbols[col_id], df_symbols['alias']))

    # Aplica o dicionario mapeando de ENSP para SOD1, FUS, etc...
    df_filtered['gene1'] = df_filtered['protein1'].map(id_to_gene)
    df_filtered['gene2'] = df_filtered['protein2'].map(id_to_gene)
    
    # Remove qualquer linha onde os dois genes não foram mapeados com sucesso
    df_filtered = df_filtered.dropna(subset=['gene1', 'gene2'])

    G = nx.from_pandas_edgelist(
        df_filtered, source='gene1', target='gene2', edge_attr='combined_score'
    )

    print(f"[info] Grafo Global criado: {G.number_of_nodes()} nós mapeados e {G.number_of_edges()} arestas.")
    return G

# --- FUNÇÃO 2: Personalized PageRank à prova de Leakage ---
def compute_personalized_pagerank(G: nx.Graph, seed_genes: List[str]) -> Dict[str, float]:
    valid_seeds = [g for g in seed_genes if g in G.nodes()]
    
    if not valid_seeds:
        print("  [aviso] Nenhum gene seed encontrado no grafo para PPR. Usando PR normal.")
        return nx.pagerank(G, alpha=0.85, weight='combined_score')
    
    personalization = {n: 0.0 for n in G.nodes()}
    for g in valid_seeds:
        personalization[g] = 1.0
        
    return nx.pagerank(G, alpha=0.85, personalization=personalization, weight='combined_score')

def build_training_data(gene_vectors, pos_genes, neg_genes, pr_dict, default_pr=0.0):
    X, y = [], []
    for g in pos_genes:
        vecs = subsample(gene_vectors[g], MAX_INST_TRAIN)
        mean_vec = np.mean(vecs, axis=0)
        pr_score = pr_dict.get(g, default_pr)
        X.append(np.append(mean_vec, pr_score))
        y.append(1)
        
    for g in neg_genes:
        vecs = subsample(gene_vectors[g], MAX_INST_TRAIN)
        mean_vec = np.mean(vecs, axis=0)
        pr_score = pr_dict.get(g, default_pr)
        X.append(np.append(mean_vec, pr_score))
        y.append(0)
        
    return np.array(X), np.array(y)

def compute_fold_metrics(ranked_genes, val_gold):
    if len(val_gold) == 0: return {}
    ranks = {g: i + 1 for i, g in enumerate(ranked_genes)} 
    found_ranks = [ranks[g] for g in val_gold if g in ranks]
    def recall_at(k): return float(len(val_gold & set(ranked_genes[:k])) / len(val_gold))
    mrr = sum((1.0 / ranks[g]) for g in val_gold if g in ranks) / len(val_gold)
    mrr20 = sum((1.0 / ranks[g]) for g in val_gold if g in ranks and ranks[g] <= 20) / len(val_gold)
    return {
        "n_val_gold": float(len(val_gold)), "auc": float("nan"),
        "recall@10": recall_at(10), "recall@50": recall_at(50), "recall@100": recall_at(100),
        "mrr": float(mrr), "mrr@20": float(mrr20),
        "mean_rank": float(np.mean(found_ranks)) if found_ranks else float("nan"),
        "median_rank": float(np.median(found_ranks)) if found_ranks else float("nan"),
    }

def fold_mean_std(cv_folds, key):
    vals = [f[key] for f in cv_folds if key in f and not np.isnan(f[key])]
    if len(vals) == 0: return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

def fmt_pm(mean, std, decimals=3):
    return "nan" if np.isnan(mean) or np.isnan(std) else f"{mean:.{decimals}f} ± {std:.{decimals}f}"

# --- MAIN EXECUTIONS ---
def main():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features not found: {FEATURES_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("[info] Lendo arquivo CSV de genes...")
    df_genes = pd.read_csv(GENES_CSV_PATH)
    lista_genes_csv = df_genes['gene'].astype(str).str.upper().tolist()

    print("[info] Loading features...")
    with open(FEATURES_PATH, "rb") as f:
        gene_vectors = pickle.load(f)

    all_genes = [g for g in lista_genes_csv if g in gene_vectors and len(gene_vectors[g]) > 0]
    all_genes_set = set(all_genes)

    gold_all = sorted({str(g).strip().upper() for g in VALIDATION_GENES})
    gold_available = [g for g in gold_all if g in all_genes_set]
    
    # Criamos o set com TUDO o que nos interessa para passar pro mapeador
    valid_genes_set = all_genes_set | set(gold_all)

    # 1. CARREGA O GRAFO GLOBAL
    if os.path.exists(STRING_LINKS_PATH) and os.path.exists(STRING_ALIASES_PATH):
        G_full = load_local_string_graph(STRING_LINKS_PATH, STRING_ALIASES_PATH, valid_genes_set, min_score=700)
        G = G_full.copy()
    else:
        raise FileNotFoundError("\n[ERRO] Baixe os arquivos da STRING e coloque na pasta correta!")

    print(f"SOD1 está no grafo? {'SOD1' in G.nodes()}")

    gold_arr = np.array(gold_available)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    gold_oof_probs: Dict[str, float] = {}
    cv_folds: List[Dict[str, float]] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(gold_arr)):
        set_seed(SEED + fold)
        gold_train = gold_arr[tr_idx].tolist()
        gold_val = gold_arr[va_idx].tolist()
        gold_val_set = set(gold_val)

        print(f"\n" + "="*40)
        print(f"CV FOLD {fold+1}/{N_FOLDS}")
        
        pos_train = [g for g in gold_train if len(gene_vectors[g]) > 0]
        
        # 2. CALCULA O PERSONALIZED PAGERANK
        print(f"  -> Computando PPR usando {len(pos_train)} genes de treino como semente...")
        fold_pr_dict = compute_personalized_pagerank(G, pos_train)
        default_pr = 0.0

        exclude_train = set(gold_train) | gold_val_set
        neg_train = get_reliable_negatives(gene_vectors, exclude_train, len(pos_train) * NEG_RATIO, RELIABLE_NEG_MAX_BAG)

        exclude_val = exclude_train | set(neg_train)
        neg_val = get_reliable_negatives(gene_vectors, exclude_val, len(gold_val) * NEG_RATIO, RELIABLE_NEG_MAX_BAG)

        X_train, y_train = build_training_data(gene_vectors, pos_train, neg_train, fold_pr_dict, default_pr)
        X_val, y_val = build_training_data(gene_vectors, gold_val, neg_val, fold_pr_dict, default_pr)

        X_all_genes = np.array([
            np.append(np.mean(gene_vectors[g], axis=0), fold_pr_dict.get(g, default_pr)) 
            for g in all_genes
        ])

        # 4. TREINAMENTO
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty='l1', solver='liblinear', C=0.1, class_weight='balanced', max_iter=1000, random_state=SEED + fold)
        )
        model.fit(X_train, y_train)

        # 5. EXTRAÇÃO DE COEFICIENTES DA REDE
        coeficientes = model.named_steps['logisticregression'].coef_[0]
        peso_ppr = coeficientes[-1] 
        
        print(f"  [resultado] Peso aprendido para o PPR (Feature Importance): {peso_ppr:.4f}")
        if peso_ppr == 0:
            print("  [aviso] A feature de PPR foi zerada pelo L1 neste fold (sem impacto).")

        probs = model.predict_proba(X_all_genes)[:, 1]
        scores = {g: prob for g, prob in zip(all_genes, probs)}

        for g in gold_val:
            gold_oof_probs[g] = scores[g]

        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fold_metrics = compute_fold_metrics(ranked, gold_val_set)

        y_true_val, y_score_val = [], []
        for g in gold_val:
            y_true_val.append(1)
            y_score_val.append(scores[g])
        for g in neg_val:
            if g in scores:
                y_true_val.append(0)
                y_score_val.append(scores[g])

        if len(set(y_true_val)) == 2:
            fold_metrics["auc"] = float(average_precision_score(y_true_val, y_score_val))
        else:
            fold_metrics["auc"] = float("nan")

        fold_metrics["fold"] = float(fold + 1)
        cv_folds.append(fold_metrics)
        
    metrics_keys = ["auc", "recall@10", "recall@50", "recall@100", "mrr", "mrr@20", "mean_rank", "median_rank"]
    summary_stats, summary_fmt = {}, {}

    print("\n" + "="*40)
    for k in metrics_keys:
        mean, std = fold_mean_std(cv_folds, k)
        summary_stats[k] = {"mean": mean, "std": std}
        summary_fmt[k] = fmt_pm(mean, std, decimals=1 if k == "mean_rank" else 3)

    cv_summary = {"folds": cv_folds, "summary_mean_std": summary_stats, "summary_fmt": summary_fmt, "n_folds": N_FOLDS, "seed": SEED}

    with open(CV_METRICS_JSON, "w") as f:
        json.dump(cv_summary, f, indent=2)

    print(f"[info] Saved CV metrics to: {CV_METRICS_JSON}")
    for k in metrics_keys:
        print(f"  {k:10s}: {summary_fmt[k]}")

    oof_genes = sorted(gold_oof_probs.keys())
    oof_scores = np.array([gold_oof_probs[g] for g in oof_genes], dtype=np.float32)
    oof_counts = np.array([len(gene_vectors[g]) for g in oof_genes], dtype=np.int32)

    np.savez_compressed(
        CV_OOF_GOLD_NPZ, genes=np.array(oof_genes, dtype=np.str_), scores_topm=oof_scores,
        ctx_counts=oof_counts, meta=np.array([json.dumps({"method": "LogReg_l1_Pipeline_w_PPR", "folds": N_FOLDS})], dtype=np.str_),
    )

    print("\n" + "="*40)
    print("FINAL MODEL TRAINING (ALL GOLD AVAILABLE)")
    set_seed(SEED + 999)
    pos_final = [g for g in gold_available if len(gene_vectors[g]) > 0]
    neg_final = get_reliable_negatives(gene_vectors, set(gold_available), len(pos_final) * NEG_RATIO, RELIABLE_NEG_MAX_BAG)

    print(f"  -> Computando PPR Global usando {len(pos_final)} genes finais de ouro como semente...")
    final_pr_dict = compute_personalized_pagerank(G, pos_final)
    
    X_train_final, y_train_final = build_training_data(gene_vectors, pos_final, neg_final, final_pr_dict, default_pr)
    
    X_all_genes_final = np.array([
        np.append(np.mean(gene_vectors[g], axis=0), final_pr_dict.get(g, default_pr)) 
        for g in all_genes
    ])

    model_final = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty='l1', solver='liblinear', C=0.1, class_weight='balanced', max_iter=1000, random_state=SEED + 999)
    )
    model_final.fit(X_train_final, y_train_final)
    
    peso_final_ppr = model_final.named_steps['logisticregression'].coef_[0][-1]
    print(f"  [resultado] Peso final aprendido para o PPR: {peso_final_ppr:.4f}")

    final_probs = model_final.predict_proba(X_all_genes_final)[:, 1]
    final_counts = [len(gene_vectors[g]) for g in all_genes]

    np.savez_compressed(
        FINAL_ALLGENES_NPZ, genes=np.array(all_genes, dtype=np.str_), scores_topm=np.array(final_probs, dtype=np.float32),
        ctx_counts=np.array(final_counts, dtype=np.int32), meta=np.array([json.dumps({"method": "LogReg_l1_Pipeline_w_PPR", "seed": SEED})], dtype=np.str_),
    )
    print(f"[info] Saved FINAL all-genes NPZ to: {FINAL_ALLGENES_NPZ}")

if __name__ == "__main__":
    main()
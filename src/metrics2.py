import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.utils import resample 
from gensim.models import Word2Vec, FastText
from gensim.models import KeyedVectors
from tabulate import tabulate
from config import VALIDATION_GENES



SIMILARITY_FOR_EMBEDDINGS = "cosine" # or anything for dot product

EPS = 1e-12
N_BOOTSTRAPS = 1000  
CONFIDENCE_LEVEL = 0.95


def l2_normalize_rows(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def l2_normalize_vec(v: np.ndarray, eps: float = EPS) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


class GeneModelEvaluator:
    def __init__(self, gold_standard_list, all_candidate_genes=None, target_terms=["als_disease_token"]):
        self.gold_genes = set([str(g).strip().upper() for g in gold_standard_list])
        self.target_terms = [t.strip() for t in target_terms]

        if all_candidate_genes:
            self.valid_candidates = set([str(g).strip().upper() for g in all_candidate_genes])
        else:
            self.valid_candidates = None

        print(f"Evaluator initialized with {len(self.gold_genes)} gold standard genes.")
        if self.valid_candidates:
            print(f"Ranking Filter ACTIVE: Restricted to {len(self.valid_candidates)} valid gene candidates.")
        else:
            print("Ranking Filter INACTIVE: Calculating ranking against ALL vocabulary words (Noisy!).")

    def _get_target_vector(self, model_kv):
        """
        Search for als_disease_token in KeyedVectors
        """
        for term in self.target_terms:
            if term in model_kv:
                return model_kv[term], term
            if term.lower() in model_kv:
                return model_kv[term.lower()], term.lower()
        return None, None

    def _calculate_bootstrap_ci(self, y_true, y_scores, k_values):
            """
            Bootstrapping to calculate Std Dev (for consistency with CV results)
            """
            boot_metrics = {
                "AUC": [],
                "MRR": [],
                "MRR@20": [],
                "Mean Rank": [],
                "Median Rank": []
            }
            for k in k_values:
                boot_metrics[f"Recall@{k}"] = []

            rng = np.random.RandomState(42)

            for _ in range(N_BOOTSTRAPS):
                # resampling
                indices = rng.randint(0, len(y_true), len(y_true))
                y_true_boot = y_true[indices]
                y_scores_boot = y_scores[indices]

                if np.sum(y_true_boot) == 0:
                    continue

                sorted_idx = np.argsort(y_scores_boot)[::-1]
                y_true_sorted = y_true_boot[sorted_idx]
             

                # AUC
                try:
                    if len(np.unique(y_true_boot)) > 1:
                        # auc = roc_auc_score(y_true_boot, y_scores_boot)
                        auc = average_precision_score(y_true_boot, y_scores_boot)
                        boot_metrics["AUC"].append(auc)
                except: pass

                gold_positions = np.where(y_true_sorted == 1)[0] + 1
                
                if len(gold_positions) > 0:
                    boot_metrics["MRR"].append(np.mean(1.0 / gold_positions))
                    gold_positions_20 = gold_positions[gold_positions <= 20]
                    mrr20 = np.mean(1.0 / gold_positions_20) if len(gold_positions_20) > 0 else 0.0
                    boot_metrics["MRR@20"].append(mrr20)
                    boot_metrics["Mean Rank"].append(np.mean(gold_positions))
                    boot_metrics["Median Rank"].append(np.median(gold_positions))
                else:
                    boot_metrics["MRR"].append(0.0)
                    boot_metrics["MRR@20"].append(0.0)

                total_gold_in_sample = np.sum(y_true_boot)
                for k in k_values:
                    hits = np.sum(y_true_sorted[:k])
                    rec = hits / total_gold_in_sample if total_gold_in_sample > 0 else 0.0
                    boot_metrics[f"Recall@{k}"].append(rec)

            
            results = {}
            for metric, values in boot_metrics.items():
                if len(values) > 0:
                    std_dev = np.std(values) 
                    results[f"{metric}_STD"] = std_dev
                else:
                    results[f"{metric}_STD"] = 0.0
            
            return results

    def evaluate_disease_association(self, model_kv, k_values=[10, 50, 100, 200]):
        """
        Embeddings mode with bootstrapping
        """
        target_vec, target_term = self._get_target_vector(model_kv)

        if target_vec is None:
            print(f"WARNING: No target terms found in model: {self.target_terms}")
            return {f"Recall@{k}": 0.0 for k in k_values} | {
                "MRR": 0.0, "AUC": 0.0, "Mean Rank": 0.0, "Median Rank": 0.0,
                "Target_Found": False, "Target_Used": "None", "Top_Genes": [],
            }

        vocab_list = model_kv.index_to_key
        valid_indices = []
        valid_words = []

        if self.valid_candidates:
            for idx, word in enumerate(vocab_list):
                if word == target_term: continue
                if word.upper() in self.valid_candidates:
                    valid_indices.append(idx)
                    valid_words.append(word)

            if not valid_indices:
                print("CRITICAL WARNING: No valid genes found in model vocabulary after filtering!")
                return {f"Recall@{k}": 0.0 for k in k_values} | {
                    "MRR": 0.0, "AUC": 0.0, "Mean Rank": 0.0, "Median Rank": 0.0, 
                    "Target_Found": True, "Target_Used": target_term, "Top_Genes": [],
                }
            vectors = model_kv.vectors[valid_indices]
            vocab_to_rank = valid_words
        else:
            vectors = model_kv.vectors
            vocab_to_rank = vocab_list

        # cosine similarity
        vectors = vectors.astype(np.float32, copy=False)
        target_vec = np.array(target_vec, dtype=np.float32)

        if SIMILARITY_FOR_EMBEDDINGS.lower() == "cosine":
            vectors_n = l2_normalize_rows(vectors, eps=EPS)
            target_n = l2_normalize_vec(target_vec, eps=EPS)
            scores = np.dot(vectors_n, target_n)
        else:
            scores = np.dot(vectors, target_vec)

       
        sorted_indices = np.argsort(scores)[::-1]
        ranked_genes = [vocab_to_rank[i].upper() for i in sorted_indices]
        
        ranked_scores = scores[sorted_indices]

        y_scores_raw = scores 
        y_true_raw = np.array([1 if vocab_to_rank[i].upper() in self.gold_genes else 0 for i in range(len(vocab_to_rank))], dtype=np.int32)

        metrics = {"Target_Used": target_term, "Target_Found": True, "Top_Genes": ranked_genes, "Top_Scores": ranked_scores.tolist()}
        
     
        try:
            metrics["AUC"] = float(average_precision_score(y_true_raw, y_scores_raw)) if len(np.unique(y_true_raw)) > 1 else 0.0
        except: metrics["AUC"] = 0.0

 
        y_true_sorted = np.array([1 if g in self.gold_genes else 0 for g in ranked_genes], dtype=np.int32)
        
        gold_positions = np.where(y_true_sorted == 1)[0] + 1
        
        if len(gold_positions) > 0:
            metrics["MRR"] = float(np.mean(1.0 / gold_positions))
            metrics["Mean Rank"] = float(np.mean(gold_positions))    
            metrics["Median Rank"] = float(np.median(gold_positions)) 
        else:
            metrics["MRR"] = 0.0
            metrics["Mean Rank"] = 0.0
            metrics["Median Rank"] = 0.0
        
        gold_positions_20 = gold_positions[gold_positions <= 20]
        metrics["MRR@20"] = float(np.mean(1.0 / gold_positions_20)) if len(gold_positions_20) > 0 else 0.0

        # total_gold = len(set(ranked_genes) & self.gold_genes)
        total_gold = int(np.sum(y_true_raw))
        for k in k_values:
            hits = int(np.sum(y_true_sorted[:k]))
            metrics[f"Recall@{k}"] = hits / total_gold if total_gold > 0 else 0.0

        # bootstrap
        print(f"   Computing Bootstrap CIs ({N_BOOTSTRAPS} iterations)...")
        cis = self._calculate_bootstrap_ci(y_true_raw, y_scores_raw, k_values)
        metrics.update(cis)

        return metrics

    def evaluate_from_scores(self, gene_scores: dict, k_values=[10, 50, 100, 200]):
        """
        Pre-computed scores mode with Bootstrapping
        """
        if self.valid_candidates is not None:
            gene_scores = {g.upper(): float(s) for g, s in gene_scores.items() if g.upper() in self.valid_candidates}
        else:
            gene_scores = {g.upper(): float(s) for g, s in gene_scores.items()}

        if not gene_scores:
            print("CRITICAL WARNING: gene_scores empty. Nothing to evaluate.")
            return {f"Recall@{k}": 0.0 for k in k_values} | {
                "MRR": 0.0, "AUC": 0.0, "Mean Rank": 0.0, "Median Rank": 0.0, 
                "Target_Found": True, "Target_Used": "precomputed", "Top_Genes": []
            }

        # arrays raw
        genes_list = list(gene_scores.keys())
        y_scores_raw = np.array(list(gene_scores.values()), dtype=np.float32)
        y_true_raw = np.array([1 if g in self.gold_genes else 0 for g in genes_list], dtype=np.int32)

        ranked = sorted(gene_scores.items(), key=lambda x: float(x[1]), reverse=True)
        ranked_genes = [g for g, _ in ranked]
        ranked_scores = np.array([float(s) for _, s in ranked], dtype=np.float32)
        y_true_sorted = np.array([1 if g in self.gold_genes else 0 for g in ranked_genes], dtype=np.int32)

        metrics = {"Target_Used": "precomputed_scores", "Target_Found": True, "Top_Genes": ranked_genes, "Top_Scores": ranked_scores.tolist()}

        try:
            metrics["AUC"] = float(average_precision_score(y_true_raw, y_scores_raw)) if len(np.unique(y_true_raw)) > 1 else 0.0
        except: metrics["AUC"] = 0.0

        gold_positions = np.where(y_true_sorted == 1)[0] + 1
        
      
        if len(gold_positions) > 0:
            metrics["MRR"] = float(np.mean(1.0 / gold_positions))
            metrics["Mean Rank"] = float(np.mean(gold_positions))    
            metrics["Median Rank"] = float(np.median(gold_positions))
        else:
            metrics["MRR"] = 0.0
            metrics["Mean Rank"] = 0.0
            metrics["Median Rank"] = 0.0
        
        gold_positions_20 = gold_positions[gold_positions <= 20]
        metrics["MRR@20"] = float(np.mean(1.0 / gold_positions_20)) if len(gold_positions_20) > 0 else 0.0

        # total_gold = len(set(ranked_genes) & self.gold_genes)
        total_gold = int(np.sum(y_true_raw))
        for k in k_values:
            hits = int(np.sum(y_true_sorted[:k]))
            metrics[f"Recall@{k}"] = hits / total_gold if total_gold > 0 else 0.0

        # bootstrap
        print(f"   Computing Bootstrap CIs ({N_BOOTSTRAPS} iterations)...")
        cis = self._calculate_bootstrap_ci(y_true_raw, y_scores_raw, k_values)
        metrics.update(cis)

        return metrics


def load_unified_model(filepath):
   
    print(f"\nLoading model: {os.path.basename(filepath)}...")

    if filepath.endswith(".npz"):
        data = np.load(filepath, allow_pickle=True)
        files = set(data.files)
        if ("scores_topm" in files) and ("genes" in files):
            genes = [str(x).strip().upper() for x in data["genes"]]
            scores = data["scores_topm"].astype(np.float32)
            gene_scores = {g: float(s) for g, s in zip(genes, scores)}
            return {"__type__": "scores", "gene_scores": gene_scores}

        if ("embeddings" not in files) or ("words" not in files):
            raise KeyError(f"NPZ missing keys. Found: {sorted(list(files))}")

        words = data["words"]
        emb = data["embeddings"]
        counts = data["counts"] if "counts" in files else None
        kv = KeyedVectors(vector_size=emb.shape[1])
        kv.add_vectors(words, emb)

        if counts is not None:
            for w, c in zip(words, counts):
                kv.set_vecattr(w, "count", int(c))
        return kv

    elif filepath.endswith(".model"):
        try: return Word2Vec.load(filepath).wv
        except:
            try: return FastText.load(filepath).wv
            except: return KeyedVectors.load(filepath)
    elif filepath.endswith(".txt"):
        return KeyedVectors.load_word2vec_format(filepath, binary=False)
    else:
        raise ValueError(f"Unknown format: {filepath}")


if __name__ == "__main__":

    umbrella_term = "motor_neuron_disease"

    #SINGLE_MODEL_PATH = "./word2vec_models/model_ALS_v200_a0p05_n15.model"
    #SINGLE_MODEL_PATH = "../fasttext_models/model_ALS_v200_a0p0025_n5.model"
    #SINGLE_MODEL_PATH = "./scores_scibert_top0/scores_top0_ALS_1970_2026.npz"
    SINGLE_MODEL_PATH = f"./logits_MIL_t_pubmedbert_{umbrella_term}/scores_final_allgenes.npz"
    #SINGLE_MODEL_PATH = "./scores_f_baseline/scores_frequency_cooc_ALS_1970_2026.npz"
    

    MY_GOLD_GENES = VALIDATION_GENES
    GENES_CSV_PATH = f"../data/genes_extracted_{umbrella_term}_with_freq.csv"
    GENE_COLUMN_NAME = "gene"
    ALL_CANDIDATE_GENES = None

    if os.path.exists(GENES_CSV_PATH):
        print(f"Loading gene universe from CSV: {GENES_CSV_PATH}")
        df_genes = pd.read_csv(GENES_CSV_PATH)
        if GENE_COLUMN_NAME not in df_genes.columns:
            raise ValueError(f"Column '{GENE_COLUMN_NAME}' not found.")
        ALL_CANDIDATE_GENES = df_genes[GENE_COLUMN_NAME].dropna().astype(str).tolist()
        print(f"Loaded {len(set([g.strip().upper() for g in ALL_CANDIDATE_GENES]))} unique gene candidates.")
    else:
        print(f"WARNING: CSV not found at {GENES_CSV_PATH}. Metrics will be noisy.")

    evaluator = GeneModelEvaluator(
        gold_standard_list=MY_GOLD_GENES,
        all_candidate_genes=ALL_CANDIDATE_GENES,
        target_terms=["als_disease_token"],
    )

    if not os.path.exists(SINGLE_MODEL_PATH):
        raise FileNotFoundError(f"ERROR: {SINGLE_MODEL_PATH}")

    model_obj = load_unified_model(SINGLE_MODEL_PATH)

    if isinstance(model_obj, dict) and model_obj.get("__type__") == "scores":
        print("Calculating Ranking Metrics (precomputed scores)...")
        rank_metrics = evaluator.evaluate_from_scores(model_obj["gene_scores"])
    else:
        print(f"Calculating Ranking Metrics (embeddings, similarity={SIMILARITY_FOR_EMBEDDINGS})...")
        rank_metrics = evaluator.evaluate_disease_association(model_obj)


    def fmt_ci(metrics_dict, key, is_percent=False):
        val = metrics_dict.get(key, 0.0)
        std = metrics_dict.get(f"{key}_STD", 0.0)
     
        if key in ["Mean Rank", "Median Rank"]:
            return f"{val:.1f} ± {std:.1f}"
        
        return f"{val:.3f} ± {std:.3f}"

    rank_data = [
        ["AUC", fmt_ci(rank_metrics, "AUC")],
        ["MRR", fmt_ci(rank_metrics, "MRR")],
        ["MRR@20", fmt_ci(rank_metrics, "MRR@20")],
        ["Mean Rank", fmt_ci(rank_metrics, "Mean Rank")],    
        ["Median Rank", fmt_ci(rank_metrics, "Median Rank")], 
        ["Recall@10", fmt_ci(rank_metrics, "Recall@10", is_percent=True)],
        ["Recall@50", fmt_ci(rank_metrics, "Recall@50", is_percent=True)],
        ["Recall@100", fmt_ci(rank_metrics, "Recall@100", is_percent=True)],
        ["Target Used", rank_metrics["Target_Used"]],
    ]

    # top_genes = rank_metrics.get("Top_Genes", [])[:20]


    # ranking_data = [[i + 1, gene, "YES" if gene in evaluator.gold_genes else ""] for i, gene in enumerate(top_genes)]

    # print("\n" + "=" * 50)
    # print(f" RESULTS FOR: {os.path.basename(SINGLE_MODEL_PATH)}")
    # print("=" * 50)

    # print("\n Gene Prioritization Performance (with 95% Bootstrap CI)")
    # print(tabulate(rank_data, headers=["Metric", "Score (95% CI)"], tablefmt="fancy_grid"))

    # print("\n Top 20 Genes in Ranking")
    # print(tabulate(ranking_data, headers=["Rank", "Gene", "In Gold Set?"], tablefmt="fancy_grid"))

    # print("\n" + "=" * 50)

    top_genes = rank_metrics.get("Top_Genes", [])[:50]
    top_scores = rank_metrics.get("Top_Scores", [])[:50] # Recupera os scores
    
    # Se por algum motivo não tiver scores (ex: erro anterior), preenche com 0.0
    if not top_scores: 
        top_scores = [0.0] * len(top_genes)

    ranking_data = []
    for i, (gene, score) in enumerate(zip(top_genes, top_scores)):
        is_gold = "YES" if gene in evaluator.gold_genes else ""
        ranking_data.append([i + 1, gene, f"{score:.6f}", is_gold])

    print("\n Top 20 Genes in Ranking")
    print(tabulate(ranking_data, headers=["Rank", "Gene", "Score", "In Gold Set?"], tablefmt="fancy_grid"))
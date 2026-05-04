import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import VALIDATION_GENES


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

UMBRELLA_TERM = "neurodegenerative_disease"
REGULARIZATION = "l1"
DEFAULT_C = 0.1
SEED = 42
THRESHOLD = 0.5

DEFAULT_MODEL_TAG = f"all_scores_LR_{REGULARIZATION}_word2vec_{UMBRELLA_TERM}"
DEFAULT_OT_JSON = SRC_DIR / "external/OT-MONDO_0004976-associated-targets-2_12_2026-v25_12.json"
DEFAULT_SCORES_NPZ = SRC_DIR / f"scores/{UMBRELLA_TERM}/{DEFAULT_MODEL_TAG}/scores_final_allgenes.npz"
DEFAULT_W2V_MODEL = SRC_DIR / f"./scores/neurodegenerative_disease/all_scores_LR_l1_word2vec_neurodegenerative_disease/word2vec_neurodegenerative_disease.model"
DEFAULT_GENE_UNIVERSE_CSV = PROJECT_ROOT / f"data/corpus/extracted_genes/genes_extracted_{UMBRELLA_TERM}.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/metrics_plots"

GT_SOURCES = [
    "eva",
    "chembl",
    "clingen",
    "crispr",
    "crisprScreen",
    "expressionAtlas",
    "geneBurden",
    "gene2Phenotype",
    "genomicsEngland",
    "impc",
    "orphanet",
    "gwasCredibleSets",
    "reactome",
    "uniprotLiterature",
    "uniprotVariants",
]


def get_plot_libs():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        return plt, sns
    except ImportError:
        return None, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia Word2Vec nas fontes do Open Targets (estilo eval.py)."
    )
    parser.add_argument("--ot-json", default=str(DEFAULT_OT_JSON), help="JSON do Open Targets.")
    parser.add_argument(
        "--scores-npz",
        default=str(DEFAULT_SCORES_NPZ),
        help="NPZ com arrays `genes` e `scores_topm`.",
    )
    parser.add_argument(
        "--w2v-model",
        default="",
        help=(
            "Arquivo .model/.kv do Word2Vec. Quando informado, o script gera scores por gene "
            "a partir dos embeddings e ignora --scores-npz."
        ),
    )
    parser.add_argument(
        "--gene-universe-csv",
        default=str(DEFAULT_GENE_UNIVERSE_CSV),
        help="CSV de universo de genes (coluna --gene-col).",
    )
    parser.add_argument("--gene-col", default="gene", help="Coluna de gene no CSV de universo.")
    parser.add_argument(
        "--regularization",
        default=REGULARIZATION,
        choices=["l1", "l2"],
        help="Regularizacao usada para gerar score a partir do .model.",
    )
    parser.add_argument("--c-value", type=float, default=DEFAULT_C, help="C da regressao logistica.")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed do ajuste da regressao logistica.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Threshold para classificar positivo em cada source do Open Targets.",
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=5,
        help="Numero minimo de positivos para calcular metricas por source.",
    )
    parser.add_argument(
        "--predictor-name",
        default="Word2Vec-LR",
        help="Nome exibido nos graficos/tabela para o modelo avaliado.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Diretorio para CSVs e graficos.",
    )
    parser.add_argument(
        "--model-tag",
        default=DEFAULT_MODEL_TAG,
        help="Prefixo de nome dos arquivos de saida.",
    )
    parser.add_argument(
        "--save-generated-scores-npz",
        default="",
        help="Se usar --w2v-model, salva o score gerado neste NPZ.",
    )
    return parser.parse_args()


def to_abs(path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def load_ot_wide(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if "symbol" not in df.columns:
        raise ValueError(f"Campo 'symbol' nao encontrado em {json_path}")

    df["gene"] = df["symbol"].astype(str).str.strip().str.upper()
    existing_cols = [c for c in df.columns if c in GT_SOURCES + ["europepmc"]]
    cols_to_keep = ["gene"] + existing_cols
    df = df[cols_to_keep]

    for c in df.columns:
        if c == "gene":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df.groupby("gene", as_index=False).max()


def load_scores_npz(npz_path: Path, score_col: str = "word2vec_score") -> pd.DataFrame:
    d = np.load(npz_path, allow_pickle=True)
    if "genes" not in d or "scores_topm" not in d:
        raise ValueError(f"NPZ invalido: {npz_path}. Esperado arrays 'genes' e 'scores_topm'.")

    df = pd.DataFrame(
        {
            "gene": d["genes"].astype(str),
            score_col: d["scores_topm"].astype(float),
        }
    )
    df["gene"] = df["gene"].str.strip().str.upper()
    return df.groupby("gene", as_index=False)[score_col].max()


def load_gene_universe(path: Path, gene_col: str) -> Set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if gene_col not in df.columns:
        raise ValueError(f"Coluna '{gene_col}' nao encontrada em {path}")
    return {str(g).strip().upper() for g in df[gene_col].dropna().astype(str) if str(g).strip()}


def load_keyed_vectors(model_path: Path):
    try:
        from gensim.models import KeyedVectors, Word2Vec
    except ImportError as exc:
        raise ImportError(
            "Dependencia ausente: gensim. Instale com `pip install gensim` para usar --w2v-model."
        ) from exc

    try:
        model = Word2Vec.load(str(model_path))
        return model.wv
    except Exception:
        pass

    try:
        return KeyedVectors.load(str(model_path))
    except Exception as exc:
        raise ValueError(f"Nao foi possivel carregar Word2Vec/KeyedVectors de {model_path}") from exc


def resolve_vocab_key(wv, gene: str) -> Optional[str]:
    candidates = [gene.lower(), gene, gene.upper()]
    seen = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        if key in wv:
            return key
    return None


def get_token_count(wv, key: str) -> int:
    try:
        return max(1, int(wv.get_vecattr(key, "count")))
    except Exception:
        pass

    try:
        return max(1, int(wv.vocab[key].count))
    except Exception:
        return 1


def build_lr_scores_from_w2v_model(
    model_path: Path,
    candidate_genes: Sequence[str],
    regularization: str,
    c_value: float,
    seed: int,
    score_col: str = "word2vec_score",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    wv = load_keyed_vectors(model_path)

    genes: list[str] = []
    vectors: list[np.ndarray] = []
    counts: Dict[str, int] = {}

    for gene in sorted(set(g.upper() for g in candidate_genes)):
        key = resolve_vocab_key(wv, gene)
        if key is None:
            continue
        genes.append(gene)
        vectors.append(np.asarray(wv[key], dtype=np.float32))
        counts[gene] = get_token_count(wv, key)

    if not genes:
        raise ValueError("Nenhum gene candidato foi encontrado no vocabulario do Word2Vec.")

    X = np.vstack(vectors).astype(np.float32)
    pos_set = {str(g).strip().upper() for g in VALIDATION_GENES}
    y = np.array([1 if g in pos_set else 0 for g in genes], dtype=np.int32)

    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("Nenhum gene de VALIDATION_GENES foi encontrado entre os candidatos com embedding.")
    if n_pos == len(y):
        raise ValueError("Todos os genes candidatos sao positivos; nao ha negativos para treinar LR.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=regularization,
            solver="liblinear",
            C=float(c_value),
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
    )
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]

    df_scores = pd.DataFrame({"gene": genes, score_col: probs.astype(np.float64)})
    return df_scores.groupby("gene", as_index=False)[score_col].max(), counts


def save_scores_npz(df_scores: pd.DataFrame, counts: Dict[str, int], out_path: Path, meta: Dict[str, str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    genes = df_scores["gene"].astype(str).to_numpy(dtype=np.str_)
    scores = df_scores["word2vec_score"].astype(float).to_numpy(dtype=np.float32)
    ctx_counts = np.array([counts.get(g, 1) for g in genes], dtype=np.int32)

    np.savez_compressed(
        out_path,
        genes=genes,
        scores_topm=scores,
        ctx_counts=ctx_counts,
        meta=np.array([json.dumps(meta)], dtype=np.str_),
    )


def compute_metrics(
    df_eval: pd.DataFrame,
    predictor_col: str,
    predictor_name: str,
    threshold: float,
    min_positives: int,
) -> list[Dict[str, float]]:
    metrics: list[Dict[str, float]] = []
    predictors = [(predictor_col, predictor_name), ("europepmc", "EuropePMC")]

    for gt_src in GT_SOURCES:
        if gt_src not in df_eval.columns:
            print(f"[Aviso] Source {gt_src} nao encontrada no JSON.")
            continue

        y_true = (df_eval[gt_src] >= threshold).astype(int)
        n_pos = int(y_true.sum())
        if n_pos < min_positives:
            print(f"PULANDO {gt_src}: apenas {n_pos} positivos no hold-out.")
            continue

        for pred_col, pred_name in predictors:
            if pred_col not in df_eval.columns:
                continue
            y_score = pd.to_numeric(df_eval[pred_col], errors="coerce").fillna(0.0)

            try:
                roc = roc_auc_score(y_true, y_score)
                pr = average_precision_score(y_true, y_score)
            except ValueError as exc:
                print(f"[Aviso] Erro ao calcular metricas para {gt_src} x {pred_name}: {exc}")
                continue

            metrics.append(
                {
                    "gt_source": gt_src,
                    "predictor": pred_name,
                    "roc_auc": float(roc),
                    "pr_auc": float(pr),
                    "n_positives": float(n_pos),
                }
            )
    return metrics


def plot_auc_comparison(
    results: Iterable[Dict[str, float]],
    metric_name: str,
    out_path: Path,
    umbrella_term: str,
    threshold: float,
) -> None:
    plt, sns = get_plot_libs()
    if plt is None or sns is None:
        print("[Aviso] matplotlib/seaborn nao instalados. Pulando geracao de grafico.")
        return

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_res, x="gt_source", y=metric_name, hue="predictor", palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"[{umbrella_term}] Hold-out validation | source score >= {threshold}")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_top_genes_heatmap(
    df_eval: pd.DataFrame,
    score_col: str,
    out_path: Path,
    umbrella_term: str,
    top_n: int = 50,
) -> None:
    plt, sns = get_plot_libs()
    if plt is None or sns is None:
        print("[Aviso] matplotlib/seaborn nao instalados. Pulando heatmap.")
        return

    if score_col not in df_eval.columns:
        return

    top_genes = df_eval.sort_values(score_col, ascending=False).head(top_n)
    cols_to_plot = [c for c in GT_SOURCES + ["europepmc", score_col] if c in df_eval.columns]
    if not cols_to_plot:
        return

    heatmap_data = top_genes.set_index("gene")[cols_to_plot]
    plt.figure(figsize=(15, 12))
    sns.heatmap(heatmap_data, annot=False, cmap="YlGnBu", cbar_kws={"label": "Score"})
    plt.title(f"[{umbrella_term}] Top {top_n} genes (hold-out)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    ot_json_path = to_abs(args.ot_json)
    out_dir = to_abs(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] Carregando Open Targets...")
    ot = load_ot_wide(ot_json_path)
    print(f"[info] Genes no OT (apos agrupamento): {len(ot)}")

    score_col = "word2vec_score"
    score_label = args.predictor_name

    if args.w2v_model.strip():
        w2v_model_path = to_abs(args.w2v_model)
        if not w2v_model_path.exists():
            raise FileNotFoundError(f"Arquivo .model nao encontrado: {w2v_model_path}")

        candidates = set(ot["gene"].astype(str).str.upper().tolist())
        candidates |= {str(g).strip().upper() for g in VALIDATION_GENES}

        gene_universe_path = to_abs(args.gene_universe_csv)
        extra_genes = load_gene_universe(gene_universe_path, args.gene_col)
        if extra_genes:
            candidates |= extra_genes
            print(f"[info] Universo de genes carregado: +{len(extra_genes)} genes.")
        else:
            print("[info] Universo de genes nao encontrado/nao usado; avaliando com genes do OT + VALIDATION_GENES.")

        print(f"[info] Gerando score a partir do .model para {len(candidates)} genes candidatos...")
        scores_df, counts = build_lr_scores_from_w2v_model(
            model_path=w2v_model_path,
            candidate_genes=sorted(candidates),
            regularization=args.regularization,
            c_value=args.c_value,
            seed=args.seed,
            score_col=score_col,
        )
        print(f"[info] Genes com score vindo do .model: {len(scores_df)}")

        if args.save_generated_scores_npz.strip():
            npz_out = to_abs(args.save_generated_scores_npz)
            meta = {
                "method": f"LogReg_{args.regularization}_from_word2vec_model",
                "source_model": str(w2v_model_path),
                "seed": str(args.seed),
            }
            save_scores_npz(scores_df, counts, npz_out, meta)
            print(f"[info] Scores gerados salvos em: {npz_out}")
    else:
        npz_path = to_abs(args.scores_npz)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"NPZ nao encontrado: {npz_path}\nUse --w2v-model para gerar score direto do .model."
            )
        print(f"[info] Carregando scores do NPZ: {npz_path}")
        scores_df = load_scores_npz(npz_path, score_col=score_col)
        print(f"[info] Genes com score no NPZ: {len(scores_df)}")

    df_full = ot.merge(scores_df, on="gene", how="inner")
    print(f"[info] Intersecao OT + score: {len(df_full)} genes")

    train_genes = {str(g).strip().upper() for g in VALIDATION_GENES}
    df_eval = df_full[~df_full["gene"].isin(train_genes)].copy()
    print(f"[info] Genes de treino removidos: {len(df_full) - len(df_eval)}")
    print(f"[info] Hold-out final: {len(df_eval)} genes")

    print(f"[info] Calculando metricas (threshold={args.threshold})...")
    all_metrics = compute_metrics(
        df_eval=df_eval,
        predictor_col=score_col,
        predictor_name=score_label,
        threshold=args.threshold,
        min_positives=args.min_positives,
    )

    holdout_table = out_dir / f"{args.model_tag}_external_eval_holdout_table.csv"
    df_eval.to_csv(holdout_table, index=False)
    print(f"[info] Tabela hold-out salva em: {holdout_table}")

    if all_metrics:
        res_df = pd.DataFrame(all_metrics)
        metrics_csv = out_dir / f"{args.model_tag}_metrics_holdout.csv"
        res_df.to_csv(metrics_csv, index=False)

        roc_plot = out_dir / f"{args.model_tag}_roc_auc_holdout.png"
        pr_plot = out_dir / f"{args.model_tag}_pr_auc_holdout.png"
        heatmap_plot = out_dir / f"{args.model_tag}_heatmap_top50_holdout.png"

        plot_auc_comparison(all_metrics, "roc_auc", roc_plot, UMBRELLA_TERM, args.threshold)
        plot_auc_comparison(all_metrics, "pr_auc", pr_plot, UMBRELLA_TERM, args.threshold)
        plot_top_genes_heatmap(df_eval, score_col, heatmap_plot, UMBRELLA_TERM, top_n=50)

        summary = res_df.pivot(index="gt_source", columns="predictor", values="roc_auc")
        print("\n=== ROC-AUC (hold-out) ===")
        print(summary.to_string())
        print(f"\n[info] Metricas salvas em: {metrics_csv}")
        print(f"[info] Grafico ROC-AUC: {roc_plot}")
        print(f"[info] Grafico PR-AUC: {pr_plot}")
        print(f"[info] Heatmap: {heatmap_plot}")
    else:
        print("[Aviso] Nenhuma metrica foi calculada. Verifique positivos restantes no hold-out.")

    print("\n=== Gold genes por source (hold-out) ===")
    for gt_src in GT_SOURCES:
        if gt_src not in df_eval.columns:
            print(f"{gt_src}: nao encontrada")
            continue
        n_gold = int((df_eval[gt_src] >= args.threshold).sum())
        print(f"{gt_src}: {n_gold}")


if __name__ == "__main__":
    main()

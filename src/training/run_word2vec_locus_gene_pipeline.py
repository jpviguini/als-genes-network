#!/usr/bin/env python3
"""Run clean Word2Vec ALS locus-to-gene pipeline up to ranking/plots stage."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path("/home/viguinijpv/200.18.99.75:8000/IC")
SRC_DIR = PROJECT_ROOT / "src"
TRAINING_DIR = SRC_DIR / "training"
PYTHON_BIN = Path(sys.executable)

DEFAULT_CORPUS_CSV = PROJECT_ROOT / "data/corpus/preprocessed/corpus_neurodegenerative_disease_preprocessed.csv"
DEFAULT_GENE_UNIVERSE_CSV = PROJECT_ROOT / "data/corpus/extracted_genes/genes_extracted_neurodegenerative_disease.csv"
DEFAULT_BASE_FEATURE_TABLE = SRC_DIR / "data/als_cs_gene_tables/GCST90027164_cs_gene_candidate_feature_table_neurodegenerative_disease.csv"
DEFAULT_OUTPUT_DIR = SRC_DIR / "data/als_cs_gene_tables" / f"word2vec_locus_gene_pipeline_{date.today().strftime('%Y%m%d')}"
DEFAULT_PRETRAINED_W2V_MODEL = SRC_DIR / "data/als_cs_gene_tables/word2vec_tmp/models/word2vec/word2vec_neurodegenerative_disease.model"
DEFAULT_PRETRAINED_W2V_SUMMARY = SRC_DIR / "data/als_cs_gene_tables/word2vec_tmp/models/word2vec/word2vec_training_summary.json"


@dataclass
class StagePaths:
    root: Path
    models_word2vec: Path
    embeddings: Path
    pca: Path
    feature_tables: Path
    training_outputs: Path
    plots: Path
    reports: Path
    logs: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Word2Vec locus-gene pipeline in separate output directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--corpus-csv", type=Path, default=DEFAULT_CORPUS_CSV)
    parser.add_argument("--gene-universe-csv", type=Path, default=DEFAULT_GENE_UNIVERSE_CSV)
    parser.add_argument("--base-feature-table", type=Path, default=DEFAULT_BASE_FEATURE_TABLE)
    parser.add_argument("--study-id", type=str, default="GCST90027164")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--regularization-strength", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument(
        "--skip-word2vec-train",
        action="store_true",
        help="Skip Word2Vec training stage and reuse an already trained model.",
    )
    parser.add_argument(
        "--pretrained-word2vec-model",
        type=Path,
        default=DEFAULT_PRETRAINED_W2V_MODEL,
        help="Path to pre-trained Word2Vec .model file (used with --skip-word2vec-train).",
    )
    parser.add_argument(
        "--pretrained-word2vec-summary",
        type=Path,
        default=DEFAULT_PRETRAINED_W2V_SUMMARY,
        help="Path to pre-trained Word2Vec training summary JSON (optional, used with --skip-word2vec-train).",
    )
    parser.add_argument(
        "--penalties",
        type=str,
        default="l2",
        help="Comma-separated penalties for train_locus_gene_ranker.py (e.g., l2 or none,l1,l2).",
    )
    parser.add_argument(
        "--run-l1-benchmark",
        action="store_true",
        help="Enable --run-l1-benchmark when l1 is included in --penalties.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete existing output directory before running.",
    )
    return parser.parse_args()


def normalize_gene_symbol(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().upper()
    return s if s else None


def parse_penalties(text: str) -> List[str]:
    allowed = {"none", "l1", "l2", "elasticnet"}
    vals = [p.strip().lower() for p in str(text).split(",") if p.strip()]
    if not vals:
        raise ValueError("--penalties cannot be empty")
    bad = [p for p in vals if p not in allowed]
    if bad:
        raise ValueError(f"Unsupported penalties: {bad}. Allowed: {sorted(allowed)}")
    return vals


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def init_stage_paths(root: Path) -> StagePaths:
    return StagePaths(
        root=root,
        models_word2vec=root / "models" / "word2vec",
        embeddings=root / "embeddings",
        pca=root / "pca",
        feature_tables=root / "feature_tables",
        training_outputs=root / "training_outputs",
        plots=root / "plots",
        reports=root / "reports",
        logs=root / "logs",
    )


def ensure_dirs(paths: StagePaths) -> None:
    for p in [
        paths.root,
        paths.models_word2vec,
        paths.embeddings,
        paths.pca,
        paths.feature_tables,
        paths.training_outputs,
        paths.plots,
        paths.reports,
        paths.logs,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(x) for x in cmd) + "\n\n")
        proc = subprocess.run(
            [str(x) for x in cmd],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-60:])
        except Exception:
            tail = "(could not read log tail)"
        raise RuntimeError(
            f"Command failed (exit={proc.returncode}): {' '.join(str(x) for x in cmd)}\n"
            f"See log: {log_path}\n\nLast log lines:\n{tail}"
        )


def build_word2vec_feature_table(
    base_feature_table: Path,
    embedding_pickle: Path,
    out_csv: Path,
    out_parquet: Path,
) -> Dict[str, object]:
    df = load_table(base_feature_table)

    with embedding_pickle.open("rb") as f:
        embed_map_raw = pickle.load(f)

    embed_map: Dict[str, np.ndarray] = {}
    for gene_raw, vec_raw in dict(embed_map_raw).items():
        gene = normalize_gene_symbol(gene_raw)
        if gene is None:
            continue
        vec = np.asarray(vec_raw, dtype=np.float32)
        if vec.ndim != 1 or vec.size == 0:
            continue
        embed_map[gene] = vec

    if not embed_map:
        raise ValueError("No usable embeddings found in pickle.")

    emb_dim = int(len(next(iter(embed_map.values()))))

    old_emb_cols = sorted([c for c in df.columns if str(c).startswith("gene_emb_")])
    drop_cols = list(old_emb_cols)
    if "has_gene_embedding" in df.columns:
        drop_cols.append("has_gene_embedding")

    base_df = df.drop(columns=drop_cols, errors="ignore").copy()

    mat = np.zeros((len(base_df), emb_dim), dtype=np.float32)
    has = np.zeros(len(base_df), dtype=np.int32)
    missing: set[str] = set()

    gene_col = "gene_symbol"
    if gene_col not in base_df.columns:
        raise ValueError("Expected column 'gene_symbol' in base feature table.")

    for i, raw_gene in enumerate(base_df[gene_col].tolist()):
        g = normalize_gene_symbol(raw_gene)
        if g is None:
            continue
        vec = embed_map.get(g)
        if vec is None or len(vec) != emb_dim:
            missing.add(g)
            continue
        mat[i, :] = vec
        has[i] = 1

    emb_cols = [f"gene_emb_{i:04d}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(mat, index=base_df.index, columns=emb_cols)
    out_df = pd.concat([base_df, emb_df], axis=1)
    out_df["has_gene_embedding"] = has.astype(int)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    out_df.to_parquet(out_parquet, index=False)

    stats = {
        "rows": int(len(out_df)),
        "embedding_dim": int(emb_dim),
        "n_old_embedding_columns_dropped": int(len(old_emb_cols)),
        "rows_with_word2vec_embedding": int(has.sum()),
        "rows_missing_word2vec_embedding": int((has == 0).sum()),
        "unique_genes": int(out_df[gene_col].astype(str).nunique()),
        "unique_genes_missing_word2vec_embedding": int(len(missing)),
        "candidate_genes_with_word2vec_embedding": int(
            out_df.loc[out_df["has_gene_embedding"] == 1, gene_col].astype(str).str.upper().nunique()
        ),
    }
    return stats


def run_pca_stage(
    embeddings_npz: Path,
    out_dir: Path,
    pca_dim: int,
    random_state: int,
) -> Dict[str, object]:
    with np.load(embeddings_npz, allow_pickle=True) as data:
        genes = np.asarray(data["genes"]).astype(str)
        emb = np.asarray(data["embeddings"], dtype=np.float32)

    if emb.ndim != 2 or emb.shape[0] == 0 or emb.shape[1] == 0:
        raise ValueError("Embeddings NPZ has empty or invalid matrix.")

    scaler = StandardScaler(with_mean=True, with_std=True)
    emb_scaled = scaler.fit_transform(emb)

    n_comp = int(min(int(pca_dim), emb_scaled.shape[0], emb_scaled.shape[1]))
    if n_comp <= 0:
        raise ValueError("PCA produced n_components <= 0")

    pca = PCA(n_components=n_comp, random_state=int(random_state))
    emb_pca = pca.fit_transform(emb_scaled)

    out_dir.mkdir(parents=True, exist_ok=True)

    pca_cols = [f"word2vec_pca_{i:03d}" for i in range(n_comp)]
    pca_df = pd.DataFrame(emb_pca, columns=pca_cols)
    pca_df.insert(0, "gene_symbol", genes)

    pca_table_path = out_dir / "word2vec_gene_embeddings_pca.csv"
    pca_df.to_csv(pca_table_path, index=False)

    evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    evr_df = pd.DataFrame(
        {
            "component": [f"emb_pca_{i:03d}" for i in range(n_comp)],
            "explained_variance_ratio": evr,
            "cumulative_explained_variance_ratio": np.cumsum(evr),
        }
    )
    evr_path = out_dir / "word2vec_pca_explained_variance.csv"
    evr_df.to_csv(evr_path, index=False)

    emb_feature_names = [f"gene_emb_{i:04d}" for i in range(emb.shape[1])]
    pca_feature_names = [f"emb_pca_{i:03d}" for i in range(n_comp)]
    artifacts_path = out_dir / "word2vec_pca_artifacts.npz"
    np.savez(
        artifacts_path,
        genes=np.asarray(genes, dtype=object),
        embedding_feature_names=np.asarray(emb_feature_names, dtype=object),
        pca_feature_names=np.asarray(pca_feature_names, dtype=object),
        emb_scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        emb_scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        pca_components=np.asarray(pca.components_, dtype=np.float64),
        pca_mean=np.asarray(pca.mean_, dtype=np.float64),
        explained_variance_ratio=evr,
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float64),
    )

    summary = {
        "n_genes_with_embeddings": int(emb.shape[0]),
        "embedding_dim_before_pca": int(emb.shape[1]),
        "pca_dim_after_reduction": int(n_comp),
        "total_explained_variance_ratio": float(evr.sum()),
        "outputs": {
            "pca_table_csv": str(pca_table_path),
            "pca_explained_variance_csv": str(evr_path),
            "pca_artifacts_npz": str(artifacts_path),
        },
    }

    summary_path = out_dir / "word2vec_pca_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_mode_pca_metrics(run_dir: Path) -> Dict[str, object]:
    p = run_dir / "cv_lolo_gene_exclusion" / "mode_pca" / "summary_metrics.json"
    if not p.exists():
        return {}
    return load_json(p)


def collect_plot_paths(plot_dir: Path, top_n: int = 25) -> List[str]:
    files = sorted([p for p in plot_dir.rglob("*.png") if p.is_file()])
    return [str(p) for p in files[:top_n]]


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def stage1_use_pretrained_word2vec(
    *,
    pretrained_model: Path,
    pretrained_summary: Optional[Path],
    output_model_dir: Path,
) -> Dict[str, object]:
    if not pretrained_model.exists():
        raise FileNotFoundError(f"Pre-trained Word2Vec model not found: {pretrained_model}")

    output_model_dir.mkdir(parents=True, exist_ok=True)
    model_name = pretrained_model.name
    copied_paths: Dict[str, str] = {}

    # Copy the full gensim Word2Vec model bundle (main .model + sidecar .npy files).
    model_bundle_files = [
        p
        for p in sorted(pretrained_model.parent.glob(f"{model_name}*"))
        if p.is_file()
    ]
    if not model_bundle_files:
        raise FileNotFoundError(f"No files found for model bundle prefix: {pretrained_model}")
    for src in model_bundle_files:
        dst = output_model_dir / src.name
        shutil.copy2(src, dst)
    copied_paths["model_path"] = str(output_model_dir / model_name)
    copied_paths["model_bundle_files"] = [str(output_model_dir / p.name) for p in model_bundle_files]

    base_stem = pretrained_model.stem
    parent = pretrained_model.parent
    sibling_candidates = {
        "keyed_vectors_path": parent / f"{base_stem}.kv",
        "vocab_top500_tsv": parent / "word2vec_vocab_top500.tsv",
    }
    for key, src in sibling_candidates.items():
        dst = output_model_dir / src.name
        if _copy_if_exists(src, dst):
            copied_paths[key] = str(dst)

    loaded_summary: Dict[str, object] = {}
    summary_src = pretrained_summary if pretrained_summary is not None else Path("")
    if summary_src and summary_src.exists():
        loaded_summary = load_json(summary_src)
        _copy_if_exists(summary_src, output_model_dir / summary_src.name)

    summary = dict(loaded_summary) if loaded_summary else {}
    summary["pretrained_model_reused"] = True
    summary["pretrained_model_source"] = str(pretrained_model)
    if summary_src and summary_src.exists():
        summary["pretrained_summary_source"] = str(summary_src)

    outputs = dict(summary.get("outputs", {}))
    outputs.update(copied_paths)
    summary["outputs"] = outputs

    summary_path = output_model_dir / "word2vec_training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def write_report(
    report_path: Path,
    *,
    output_dir: Path,
    corpus_path: Path,
    scripts_created: Sequence[Path],
    reused_scripts: Sequence[Path],
    train_summary: Dict[str, object],
    embedding_summary: Dict[str, object],
    pca_summary: Dict[str, object],
    feature_stats: Dict[str, object],
    feature_table_csv: Path,
    feature_table_parquet: Path,
    penalty_runs: List[Dict[str, object]],
    plots_by_penalty: Dict[str, List[str]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    train_params = train_summary.get("word2vec_parameters", {})
    corpus_stats = train_summary.get("corpus_stats", {})
    model_stats = train_summary.get("model_stats", {})

    lines: List[str] = []
    lines.append("# Word2Vec Locus-to-Gene Pipeline Report")
    lines.append("")
    lines.append(f"Run date: {date.today().isoformat()}")
    lines.append(f"Output root: `{output_dir}`")
    lines.append("")

    lines.append("## 1) Scripts Created")
    lines.append("")
    for p in scripts_created:
        lines.append(f"- `{p}`")
    lines.append("")

    lines.append("## 2) Existing Scripts Reused")
    lines.append("")
    for p in reused_scripts:
        lines.append(f"- `{p}`")
    lines.append("")

    lines.append("## 3) Neurodegenerative Corpus Location")
    lines.append("")
    lines.append(f"- `{corpus_path}`")
    lines.append("")

    lines.append("## 4) How Word2Vec Was Trained")
    lines.append("")
    lines.append("- Preprocessing: regex tokenization `[A-Za-z0-9_]+`, lowercase, year-window filter.")
    lines.append(f"- Corpus rows (total): `{corpus_stats.get('total_rows')}`")
    lines.append(f"- Corpus rows in year window: `{corpus_stats.get('rows_in_year_window')}`")
    lines.append(f"- Corpus rows with tokens: `{corpus_stats.get('rows_with_tokens')}`")
    lines.append(f"- Total tokens used: `{corpus_stats.get('total_tokens')}`")
    lines.append(f"- Year range observed: `{corpus_stats.get('min_year_observed')} .. {corpus_stats.get('max_year_observed')}`")
    lines.append(f"- Vocabulary size: `{model_stats.get('vocabulary_size')}`")
    lines.append(f"- Word2Vec params: `{json.dumps(train_params, ensure_ascii=True)}`")
    lines.append("")

    lines.append("## 5) How One Embedding Per Gene Was Constructed")
    lines.append("")
    lines.append(
        "- Gene symbols were normalized to uppercase and mapped to Word2Vec vocabulary keys via case variants "
        "(lower/original/upper)."
    )
    lines.append(
        "- Each mapped gene receives one static Word2Vec vector (single token embedding), i.e., one vector per gene "
        "without per-document averaging."
    )
    lines.append("")

    lines.append("## 6) Embedding Dimensionality Before PCA")
    lines.append("")
    lines.append(f"- `{pca_summary.get('embedding_dim_before_pca')}`")
    lines.append("")

    lines.append("## 7) PCA Dimension After Reduction")
    lines.append("")
    lines.append(f"- `{pca_summary.get('pca_dim_after_reduction')}`")
    lines.append("")

    lines.append("## 8) Explained Variance")
    lines.append("")
    lines.append(f"- Total explained variance ratio: `{pca_summary.get('total_explained_variance_ratio')}`")
    lines.append(f"- PCA summary: `{output_dir / 'pca' / 'word2vec_pca_summary.json'}`")
    lines.append(f"- PCA EVR table: `{output_dir / 'pca' / 'word2vec_pca_explained_variance.csv'}`")
    lines.append("")

    lines.append("## 9) Number of Genes with Word2Vec Embeddings")
    lines.append("")
    gene_counts = embedding_summary.get("gene_counts", {})
    lines.append(f"- Universe+candidate genes evaluated: `{gene_counts.get('combined')}`")
    lines.append(f"- Genes with embeddings: `{gene_counts.get('with_word2vec_embedding')}`")
    lines.append(f"- Candidate-table genes with embeddings: `{gene_counts.get('candidate_table_with_word2vec_embedding')}`")
    lines.append("")

    lines.append("## 10) Number of Candidate Genes in Final Locus-Ranking Setup")
    lines.append("")
    lines.append(f"- Feature-table rows: `{feature_stats.get('rows')}`")
    lines.append(f"- Unique candidate genes: `{feature_stats.get('unique_genes')}`")
    lines.append(f"- Candidate genes with Word2Vec embedding: `{feature_stats.get('candidate_genes_with_word2vec_embedding')}`")
    lines.append("")

    lines.append("## 11) Main Training/Evaluation Metrics")
    lines.append("")
    if not penalty_runs:
        lines.append("- No completed training runs found.")
    else:
        lines.append("| penalty | PR-AUC | ROC-AUC | Recall@1 | Recall@3 | MRR |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for run in penalty_runs:
            m = run.get("metrics", {})
            lines.append(
                "| {penalty} | {pr:.4f} | {roc:.4f} | {r1:.4f} | {r3:.4f} | {mrr:.4f} |".format(
                    penalty=run.get("penalty", "?"),
                    pr=float(m.get("mean_fold_pr_auc", float("nan"))),
                    roc=float(m.get("mean_fold_roc_auc", float("nan"))),
                    r1=float(m.get("mean_recall_at_1", float("nan"))),
                    r3=float(m.get("mean_recall_at_3", float("nan"))),
                    mrr=float(m.get("mean_mrr", float("nan")),),
                )
            )
    lines.append("")

    lines.append("## 12) Paths to Key Outputs and Plots")
    lines.append("")
    lines.append(f"- Word2Vec model dir: `{output_dir / 'models' / 'word2vec'}`")
    lines.append(f"- Gene embedding artifacts: `{output_dir / 'embeddings'}`")
    lines.append(f"- PCA artifacts: `{output_dir / 'pca'}`")
    lines.append(f"- Word2Vec locus feature table CSV: `{feature_table_csv}`")
    lines.append(f"- Word2Vec locus feature table Parquet: `{feature_table_parquet}`")
    lines.append(f"- Training outputs: `{output_dir / 'training_outputs'}`")
    lines.append(f"- Plots root: `{output_dir / 'plots'}`")

    for run in penalty_runs:
        pen = str(run.get("penalty"))
        run_dir = run.get("run_dir")
        lines.append(f"- Penalty `{pen}` run dir: `{run_dir}`")
        paths = plots_by_penalty.get(pen, [])
        if paths:
            lines.append(f"  - Example plots ({min(5, len(paths))} shown):")
            for p in paths[:5]:
                lines.append(f"    - `{p}`")

    lines.append("")
    lines.append("## Additional Notes")
    lines.append("")
    lines.append(
        "- Non-text locus features and candidate rows were preserved from the existing neurodegenerative candidate table; "
        "only the embedding block (`gene_emb_*`, `has_gene_embedding`) was replaced with Word2Vec outputs."
    )
    lines.append("- Network analysis was not run in this pipeline.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    penalties = parse_penalties(args.penalties)

    out_dir = args.output_dir.resolve()
    if out_dir.exists() and args.overwrite_output:
        shutil.rmtree(out_dir)

    stage_paths = init_stage_paths(out_dir)
    ensure_dirs(stage_paths)

    # Stage 1: train Word2Vec model
    if args.skip_word2vec_train:
        train_summary = stage1_use_pretrained_word2vec(
            pretrained_model=args.pretrained_word2vec_model.resolve(),
            pretrained_summary=(args.pretrained_word2vec_summary.resolve() if args.pretrained_word2vec_summary else None),
            output_model_dir=stage_paths.models_word2vec,
        )
        skip_log = stage_paths.logs / "stage1_train_word2vec.log"
        skip_log.write_text(
            "[info] Stage 1 skipped.\n"
            f"[info] Reused pre-trained model: {args.pretrained_word2vec_model.resolve()}\n"
            f"[info] Output model dir: {stage_paths.models_word2vec}\n",
            encoding="utf-8",
        )
    else:
        train_script = TRAINING_DIR / "train_word2vec_gene_embeddings.py"
        train_log = stage_paths.logs / "stage1_train_word2vec.log"
        run_cmd(
            [
                str(PYTHON_BIN),
                str(train_script),
                "--corpus-csv",
                str(args.corpus_csv),
                "--out-dir",
                str(stage_paths.models_word2vec),
            ],
            train_log,
        )
        train_summary = load_json(stage_paths.models_word2vec / "word2vec_training_summary.json")

    # Stage 2: build one embedding per gene
    build_script = TRAINING_DIR / "build_word2vec_gene_embedding_table.py"
    build_log = stage_paths.logs / "stage2_build_gene_embeddings.log"
    run_cmd(
        [
            str(PYTHON_BIN),
            str(build_script),
            "--word2vec-model",
            str(stage_paths.models_word2vec / "word2vec_neurodegenerative_disease.model"),
            "--gene-universe-csv",
            str(args.gene_universe_csv),
            "--candidate-table",
            str(args.base_feature_table),
            "--out-dir",
            str(stage_paths.embeddings),
        ],
        build_log,
    )
    embedding_summary = load_json(stage_paths.embeddings / "word2vec_gene_embedding_summary.json")

    # Stage 3: PCA on gene embeddings
    pca_summary = run_pca_stage(
        embeddings_npz=stage_paths.embeddings / "word2vec_gene_embeddings.npz",
        out_dir=stage_paths.pca,
        pca_dim=int(args.pca_dim),
        random_state=int(args.random_state),
    )

    # Stage 4: feature table (replace embedding block only)
    feature_table_csv = stage_paths.feature_tables / f"{args.study_id}_cs_gene_candidate_feature_table_word2vec.csv"
    feature_table_parquet = stage_paths.feature_tables / f"{args.study_id}_cs_gene_candidate_feature_table_word2vec.parquet"
    feature_stats = build_word2vec_feature_table(
        base_feature_table=args.base_feature_table,
        embedding_pickle=stage_paths.embeddings / "word2vec_gene_embeddings.pkl",
        out_csv=feature_table_csv,
        out_parquet=feature_table_parquet,
    )

    feature_stats_path = stage_paths.feature_tables / "word2vec_feature_table_summary.json"
    with feature_stats_path.open("w", encoding="utf-8") as f:
        json.dump(feature_stats, f, indent=2)

    # Stage 5 + 6: train ranker + generate plots per penalty
    ranker_script = TRAINING_DIR / "train_locus_gene_ranker.py"
    plot_script = TRAINING_DIR / "plot_locus_gene_ranker_figures.py"

    penalty_runs: List[Dict[str, object]] = []
    plots_by_penalty: Dict[str, List[str]] = {}

    for penalty in penalties:
        pen_out = stage_paths.training_outputs / "runs" / penalty
        train_log = stage_paths.logs / f"stage5_train_ranker_{penalty}.log"

        cmd = [
            str(PYTHON_BIN),
            str(ranker_script),
            "--input-table",
            str(feature_table_csv),
            "--out-dir",
            str(pen_out),
            "--embedding-mode",
            "all",
            "--cv-mode",
            "lolo_gene_exclusion",
            "--baseline-profile",
            "quantitative",
            "--pca-dim",
            str(int(args.pca_dim)),
            "--penalty",
            penalty,
            "--regularization-strength",
            str(float(args.regularization_strength)),
            "--max-iter",
            str(int(args.max_iter)),
            "--random-state",
            str(int(args.random_state)),
        ]
        if penalty == "l1" and args.run_l1_benchmark:
            cmd.append("--run-l1-benchmark")

        run_cmd(cmd, train_log)

        cv_dir = pen_out / "cv_lolo_gene_exclusion"
        plot_dir = stage_paths.plots / penalty
        plot_log = stage_paths.logs / f"stage6_plot_{penalty}.log"
        run_cmd(
            [
                str(PYTHON_BIN),
                str(plot_script),
                "--results-dir",
                str(cv_dir),
                "--figures-dir",
                str(plot_dir),
                "--ranking-mode",
                "pca",
            ],
            plot_log,
        )

        metrics = load_mode_pca_metrics(pen_out)
        penalty_runs.append(
            {
                "penalty": penalty,
                "run_dir": str(pen_out),
                "metrics": metrics,
            }
        )
        plots_by_penalty[penalty] = collect_plot_paths(plot_dir, top_n=100)

    # Write report
    report_path = out_dir / "word2vec_locus_gene_pipeline_report.md"
    write_report(
        report_path=report_path,
        output_dir=out_dir,
        corpus_path=args.corpus_csv,
        scripts_created=[
            TRAINING_DIR / "train_word2vec_gene_embeddings.py",
            TRAINING_DIR / "build_word2vec_gene_embedding_table.py",
            TRAINING_DIR / "run_word2vec_locus_gene_pipeline.py",
        ],
        reused_scripts=[
            TRAINING_DIR / "train_locus_gene_ranker.py",
            TRAINING_DIR / "plot_locus_gene_ranker_figures.py",
        ],
        train_summary=train_summary,
        embedding_summary=embedding_summary,
        pca_summary=pca_summary,
        feature_stats=feature_stats,
        feature_table_csv=feature_table_csv,
        feature_table_parquet=feature_table_parquet,
        penalty_runs=penalty_runs,
        plots_by_penalty=plots_by_penalty,
    )

    run_summary = {
        "output_dir": str(out_dir),
        "train_summary_json": str(stage_paths.models_word2vec / "word2vec_training_summary.json"),
        "embedding_summary_json": str(stage_paths.embeddings / "word2vec_gene_embedding_summary.json"),
        "pca_summary_json": str(stage_paths.pca / "word2vec_pca_summary.json"),
        "feature_table_summary_json": str(feature_stats_path),
        "report_path": str(report_path),
        "penalty_runs": penalty_runs,
        "plots_by_penalty": plots_by_penalty,
    }

    run_summary_path = stage_paths.reports / "word2vec_pipeline_run_summary.json"
    with run_summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("[done] Word2Vec locus-gene pipeline finished.")
    print(f"[done] Output directory: {out_dir}")
    print(f"[done] Report: {report_path}")


if __name__ == "__main__":
    main()

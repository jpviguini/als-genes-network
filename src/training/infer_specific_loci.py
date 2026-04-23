#!/usr/bin/env python3
"""Inference utility for applying trained locus-to-gene models to chosen loci.

This script keeps training and inference separated. It:
1) Loads (or rebuilds) the candidate-gene feature table using the same builder.
2) Filters to requested lead-variant IDs.
3) Loads trained model artifacts for both baseline and target/full modes.
4) Applies saved preprocessing + linear model parameters for each mode.
5) Compares within-locus rankings (baseline vs target/full) and writes inspection-friendly outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

import build_cs_gene_candidate_feature_table as feature_builder
from train_locus_gene_ranker import (
    GENE_ID_COL,
    GENE_SYMBOL_COL,
    LOCUS_COL,
    as_numeric_matrix,
    impute_distance_features_worst_case,
)


DEFAULT_MODEL_DIR = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/"
    "locus_gene_ranker_l1_benchmark_20260330/cv_lolo_gene_exclusion/mode_pca"
)
DEFAULT_FEATURE_TABLE = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/"
    "GCST90027164_cs_gene_candidate_feature_table.csv"
)
DEFAULT_OUT_DIR = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/locus_gene_inference_specific_loci"
)
DEFAULT_VARIANT_IDS = ["5_172927728_T_C", "7_131765333_G_A"]
LEAD_VARIANT_COL = "gwas_lead_variant_id"
LEAD_RSIDS_COL = "gwas_lead_variant_rsids"
STUDY_COL = "gwas_study_id"
LEAD_CHROM_COL = "gwas_lead_variant_chromosome"
LEAD_POS_COL = "gwas_lead_variant_position"
RSID_PATTERN = re.compile(r"rs\d+", flags=re.IGNORECASE)
VARIANT_COMPONENT_PATTERN = re.compile(
    r"^(?:chr)?([0-9]+|x|y|mt|m)[_: -]?([0-9]+)[_: /-]+([acgtn]+)[_: /->]+([acgtn]+)$",
    flags=re.IGNORECASE,
)
CHR_POS_PATTERN = re.compile(r"^(?:chr)?([0-9]+|x|y|mt|m)[_: -]?([0-9]+)$", flags=re.IGNORECASE)
SIGMOID_FLOOR = float(1.0 / (1.0 + np.exp(50.0)))


@dataclass(frozen=True)
class InferenceArtifacts:
    model_dir: Path
    mode: str
    feature_order: List[str]
    coefficients: np.ndarray
    intercept: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    baseline_features: List[str]
    embedding_feature_names: List[str]
    pca_feature_names: List[str]
    emb_scaler_mean: np.ndarray | None
    emb_scaler_scale: np.ndarray | None
    pca_components: np.ndarray | None
    pca_mean: np.ndarray | None
    residual_coefficients: np.ndarray | None
    residual_missing_embedding_zero: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply trained locus-to-gene model to specific lead variants and/or rsIDs."
    )
    parser.add_argument(
        "--variant-ids",
        nargs="+",
        default=None,
        help="Lead variant IDs (e.g., 5_172927728_T_C 7_131765333_G_A).",
    )
    parser.add_argument(
        "--rsids",
        nargs="+",
        default=None,
        help="rsIDs to match against gwas_lead_variant_rsids (e.g., rs12608932 rs9275477).",
    )
    parser.add_argument(
        "--study-locus-ids",
        nargs="+",
        default=None,
        help="Optional Open Targets study-locus IDs to resolve directly.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory of target/full trained model artifacts (e.g., .../mode_pca).",
    )
    parser.add_argument(
        "--baseline-model-dir",
        type=Path,
        default=None,
        help=(
            "Directory of baseline trained model artifacts (e.g., .../mode_none). "
            "If omitted, inferred as sibling mode_none near --model-dir."
        ),
    )
    parser.add_argument(
        "--feature-table",
        type=Path,
        default=DEFAULT_FEATURE_TABLE,
        help="Existing candidate-gene feature table (CSV/Parquet).",
    )
    parser.add_argument(
        "--rebuild-feature-table",
        action="store_true",
        help="Rebuild feature table via build_cs_gene_candidate_feature_table.py logic before inference.",
    )
    parser.add_argument(
        "--study-id",
        type=str,
        default=feature_builder.DEFAULT_STUDY_ID,
        help="Study ID used when rebuilding the feature table.",
    )
    parser.add_argument(
        "--feature-out-dir",
        type=Path,
        default=feature_builder.DEFAULT_OUT_DIR,
        help="Output directory for rebuilt feature tables.",
    )
    parser.add_argument(
        "--hpa-path",
        type=Path,
        default=None,
        help="Optional HPA RNA tissue table path used when rebuilding.",
    )
    parser.add_argument("--enable-hpa", action="store_true", help="Enable HPA during rebuild.")
    parser.add_argument("--disable-hpa", action="store_true", help="Disable HPA during rebuild.")
    parser.add_argument(
        "--hpa-expression-threshold",
        type=float,
        default=feature_builder.DEFAULT_HPA_EXPRESSION_THRESHOLD,
        help="HPA threshold for binary features during rebuild.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for inference CSV/markdown files.",
    )
    parser.add_argument(
        "--max-model-input-features",
        type=int,
        default=10,
        help="Max number of model-space features to append in output tables.",
    )
    return parser.parse_args()


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported feature table format: {path}")


def _resolve_feature_table_path(args: argparse.Namespace) -> Path:
    if not bool(args.rebuild_feature_table):
        return args.feature_table

    enable_hpa = bool(feature_builder.CONFIG.enable_hpa)
    if bool(args.enable_hpa):
        enable_hpa = True
    if bool(args.disable_hpa):
        enable_hpa = False

    config = replace(
        feature_builder.CONFIG,
        study_id=str(args.study_id),
        out_dir=Path(args.feature_out_dir),
        hpa_path=args.hpa_path,
        enable_hpa=enable_hpa,
        hpa_expression_threshold=float(args.hpa_expression_threshold),
    )
    feature_builder.main(config)
    return Path(config.out_dir) / f"{config.study_id}_cs_gene_candidate_feature_table.csv"


def _safe_scale(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale_safe = np.asarray(scale, dtype=np.float64).copy()
    zero_mask = ~np.isfinite(scale_safe) | (np.abs(scale_safe) <= 1e-12)
    scale_safe[zero_mask] = 1.0
    out = values / scale_safe
    out[:, zero_mask] = 0.0
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _infer_has_embedding_mask(df: pd.DataFrame, embedding_cols: Sequence[str]) -> np.ndarray:
    if "has_gene_embedding" in df.columns:
        raw = pd.to_numeric(df["has_gene_embedding"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return raw > 0.5
    if not embedding_cols:
        return np.zeros(len(df), dtype=bool)
    emb = as_numeric_matrix(df, embedding_cols, fill_value=0.0)
    return np.any(np.abs(emb) > 1e-12, axis=1)


def _normalize_rsid_token(text: str) -> str:
    token = str(text).strip().lower()
    if not token:
        return ""
    matches = RSID_PATTERN.findall(token)
    if len(matches) == 1 and matches[0] == token:
        return token
    return ""


def _extract_rsids_from_value(value: object) -> List[str]:
    if value is None:
        return []
    tokens = RSID_PATTERN.findall(str(value).lower())
    if not tokens:
        return []
    seen = set()
    out: List[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _resolve_requested_inputs(args: argparse.Namespace) -> Tuple[List[str], List[str], List[str]]:
    raw_variant_ids = args.variant_ids if args.variant_ids is not None else None
    raw_rsids = args.rsids if args.rsids is not None else None
    raw_study_locus_ids = args.study_locus_ids if args.study_locus_ids is not None else None

    if raw_variant_ids is None and raw_rsids is None and raw_study_locus_ids is None:
        variant_ids = list(DEFAULT_VARIANT_IDS)
        rsids: List[str] = []
        study_locus_ids: List[str] = []
        return variant_ids, rsids, study_locus_ids

    variant_ids = [str(v).strip() for v in (raw_variant_ids or []) if str(v).strip()]
    variant_ids = list(dict.fromkeys(variant_ids))

    rsids = [str(v).strip() for v in (raw_rsids or []) if str(v).strip()]
    rsids = list(dict.fromkeys(rsids))

    study_locus_ids = [str(v).strip() for v in (raw_study_locus_ids or []) if str(v).strip()]
    study_locus_ids = list(dict.fromkeys(study_locus_ids))
    return variant_ids, rsids, study_locus_ids


def _canonical_chromosome_token(text: object) -> str:
    token = str(text).strip().lower()
    if token.startswith("chr"):
        token = token[3:]
    if token == "m":
        token = "mt"
    if not token:
        return ""
    return token.upper()


def _normalize_variant_id_token(text: str) -> str:
    token = str(text).strip()
    if not token:
        return ""
    m = VARIANT_COMPONENT_PATTERN.match(token)
    if m is None:
        return ""
    chrom = _canonical_chromosome_token(m.group(1))
    pos = str(int(m.group(2)))
    ref = m.group(3).upper()
    alt = m.group(4).upper()
    return f"{chrom}_{pos}_{ref}_{alt}"


def _normalize_chr_pos_token(text: str) -> str:
    token = str(text).strip()
    if not token:
        return ""
    m = CHR_POS_PATTERN.match(token)
    if m is None:
        return ""
    chrom = _canonical_chromosome_token(m.group(1))
    pos = str(int(m.group(2)))
    return f"{chrom}_{pos}"


def _normalize_chr_pos_from_row(chrom_value: object, pos_value: object) -> str:
    chrom = _canonical_chromosome_token(chrom_value)
    if not chrom:
        return ""
    try:
        pos_float = float(pos_value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(pos_float):
        return ""
    pos = str(int(round(pos_float)))
    return f"{chrom}_{pos}"


def _first_non_empty(series: pd.Series) -> str:
    for value in series.tolist():
        token = str(value).strip()
        if token and token.lower() != "nan":
            return token
    return ""


def _unique_tokens_join(tokens: List[str]) -> str:
    seen: Set[str] = set()
    out: List[str] = []
    for token in tokens:
        t = str(token).strip()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return "|".join(out)


def _set_index_add(mapping: Dict[str, Set[str]], key: str, locus_id: str) -> None:
    if not key:
        return
    bucket = mapping.get(key)
    if bucket is None:
        mapping[key] = {locus_id}
    else:
        bucket.add(locus_id)


def _build_locus_reference(feature_df: pd.DataFrame) -> pd.DataFrame:
    required = [LOCUS_COL, LEAD_VARIANT_COL]
    for col in required:
        if col not in feature_df.columns:
            raise ValueError(f"Feature table does not contain required column '{col}'.")

    rows: List[Dict[str, str]] = []
    for locus_id, chunk in feature_df.groupby(LOCUS_COL, sort=True, dropna=False):
        locus_token = str(locus_id).strip()
        lead_variant = _first_non_empty(chunk[LEAD_VARIANT_COL].fillna("").astype(str))
        lead_rsids_tokens: List[str] = []
        if LEAD_RSIDS_COL in chunk.columns:
            for value in chunk[LEAD_RSIDS_COL].tolist():
                lead_rsids_tokens.extend(_extract_rsids_from_value(value))
        lead_rsids_joined = _unique_tokens_join(lead_rsids_tokens)
        study_id = _first_non_empty(chunk[STUDY_COL].fillna("").astype(str)) if STUDY_COL in chunk.columns else ""
        chrom_value = _first_non_empty(chunk[LEAD_CHROM_COL].fillna("").astype(str)) if LEAD_CHROM_COL in chunk.columns else ""
        pos_value: object = ""
        if LEAD_POS_COL in chunk.columns:
            pos_series = pd.to_numeric(chunk[LEAD_POS_COL], errors="coerce")
            finite = pos_series[np.isfinite(pos_series.to_numpy(dtype=np.float64))]
            if not finite.empty:
                pos_value = str(int(round(float(finite.iloc[0]))))
        rows.append(
            {
                LOCUS_COL: locus_token,
                LEAD_VARIANT_COL: lead_variant,
                LEAD_RSIDS_COL: lead_rsids_joined,
                STUDY_COL: study_id,
                LEAD_CHROM_COL: str(chrom_value).strip(),
                LEAD_POS_COL: str(pos_value).strip(),
                "canonical_lead_variant_id": _normalize_variant_id_token(lead_variant),
                "canonical_lead_chr_pos": _normalize_chr_pos_from_row(chrom_value, pos_value),
            }
        )

    locus_ref = pd.DataFrame(rows)
    locus_ref = locus_ref.sort_values([LEAD_VARIANT_COL, LOCUS_COL], kind="stable").reset_index(drop=True)
    return locus_ref


def _single_locus_match(candidates: Set[str]) -> Tuple[bool, str, str]:
    if not candidates:
        return False, "", ""
    if len(candidates) == 1:
        only = next(iter(candidates))
        return True, only, ""
    ordered = sorted(candidates)
    return False, "", f"ambiguous_match_multiple_loci:{'|'.join(ordered)}"


def resolve_requested_loci(
    feature_df: pd.DataFrame,
    *,
    requested_variant_ids: List[str],
    requested_rsids: List[str],
    requested_study_locus_ids: List[str],
) -> pd.DataFrame:
    locus_ref = _build_locus_reference(feature_df)
    locus_by_id = {str(row[LOCUS_COL]): row for _, row in locus_ref.iterrows()}

    lead_exact_map: Dict[str, Set[str]] = {}
    lead_norm_map: Dict[str, Set[str]] = {}
    rsid_map: Dict[str, Set[str]] = {}
    chr_pos_map: Dict[str, Set[str]] = {}
    study_locus_map: Dict[str, Set[str]] = {}

    for row in locus_ref.itertuples(index=False):
        locus_id = str(getattr(row, LOCUS_COL))
        lead_variant = str(getattr(row, LEAD_VARIANT_COL)).strip()
        lead_variant_norm = str(getattr(row, "canonical_lead_variant_id")).strip()
        lead_chr_pos = str(getattr(row, "canonical_lead_chr_pos")).strip()
        study_locus = str(getattr(row, LOCUS_COL)).strip()
        lead_rsids = _extract_rsids_from_value(getattr(row, LEAD_RSIDS_COL))

        _set_index_add(lead_exact_map, lead_variant, locus_id)
        _set_index_add(lead_norm_map, lead_variant_norm, locus_id)
        _set_index_add(chr_pos_map, lead_chr_pos, locus_id)
        _set_index_add(study_locus_map, study_locus, locus_id)
        for rsid in lead_rsids:
            _set_index_add(rsid_map, rsid, locus_id)

    queries: List[Tuple[str, str]] = []
    queries.extend([(str(v), "variant_id") for v in requested_variant_ids])
    queries.extend([(str(r), "rsid") for r in requested_rsids])
    queries.extend([(str(s), "study_locus_id") for s in requested_study_locus_ids])

    audit_rows: List[Dict[str, object]] = []
    for requested_query, query_type in queries:
        q = str(requested_query).strip()
        resolved_locus = ""
        resolution_method = "unresolved"
        notes = ""

        exact_hit, exact_locus, exact_note = _single_locus_match(lead_exact_map.get(q, set()))
        if exact_hit:
            resolved_locus = exact_locus
            resolution_method = "exact_gwas_lead_variant_id"
        elif exact_note:
            notes = exact_note
        else:
            rsid_token = _normalize_rsid_token(q)
            rsid_hit, rsid_locus, rsid_note = _single_locus_match(rsid_map.get(rsid_token, set()))
            if rsid_hit:
                resolved_locus = rsid_locus
                resolution_method = "exact_gwas_lead_variant_rsids"
            elif rsid_note:
                notes = rsid_note
            else:
                study_hit, study_locus, study_note = _single_locus_match(study_locus_map.get(q, set()))
                if study_hit:
                    resolved_locus = study_locus
                    resolution_method = "robust_gwas_study_locus_id"
                elif study_note:
                    notes = study_note
                else:
                    q_norm_variant = _normalize_variant_id_token(q)
                    norm_hit, norm_locus, norm_note = _single_locus_match(
                        lead_norm_map.get(q_norm_variant, set())
                    )
                    if norm_hit:
                        resolved_locus = norm_locus
                        resolution_method = "robust_canonical_variant_id"
                    elif norm_note:
                        notes = norm_note
                    else:
                        q_chr_pos = _normalize_chr_pos_token(q)
                        chr_pos_hit, chr_pos_locus, chr_pos_note = _single_locus_match(
                            chr_pos_map.get(q_chr_pos, set())
                        )
                        if chr_pos_hit:
                            resolved_locus = chr_pos_locus
                            resolution_method = "robust_lead_chr_pos_unique"
                        elif chr_pos_note:
                            notes = chr_pos_note

        resolved_flag = int(bool(resolved_locus))
        matched_lead_variant = ""
        matched_lead_rsids = ""
        matched_study_locus = ""
        if resolved_flag:
            row = locus_by_id.get(resolved_locus)
            if row is not None:
                matched_lead_variant = str(row.get(LEAD_VARIANT_COL, ""))
                matched_lead_rsids = str(row.get(LEAD_RSIDS_COL, ""))
                matched_study_locus = str(row.get(LOCUS_COL, ""))
            else:
                matched_study_locus = resolved_locus
        else:
            if not notes:
                notes = "not_found_in_current_feature_table_for_study"
            if query_type == "rsid" and not _normalize_rsid_token(q):
                notes = "identifier_mismatch_invalid_rsid_format"

        audit_rows.append(
            {
                "requested_query": q,
                "query_type": query_type,
                "resolved_flag": resolved_flag,
                "resolution_method": resolution_method,
                "matched_gwas_lead_variant_id": matched_lead_variant,
                "matched_gwas_lead_variant_rsids": matched_lead_rsids,
                "matched_gwas_study_locus_id": matched_study_locus,
                "notes": notes,
            }
        )

    audit_df = pd.DataFrame(
        audit_rows,
        columns=[
            "requested_query",
            "query_type",
            "resolved_flag",
            "resolution_method",
            "matched_gwas_lead_variant_id",
            "matched_gwas_lead_variant_rsids",
            "matched_gwas_study_locus_id",
            "notes",
        ],
    )
    return audit_df


def write_resolution_audit_markdown(audit_df: pd.DataFrame, output_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Locus Resolution Audit")
    lines.append("")
    lines.append(f"- Total queries: `{len(audit_df)}`")
    lines.append(f"- Resolved queries: `{int(audit_df['resolved_flag'].sum())}`")
    lines.append(f"- Unresolved queries: `{int((audit_df['resolved_flag'] == 0).sum())}`")
    lines.append("")
    lines.append(
        "| requested_query | query_type | resolved_flag | resolution_method | "
        "matched_gwas_lead_variant_id | matched_gwas_lead_variant_rsids | matched_gwas_study_locus_id | notes |"
    )
    lines.append("|---|---|---:|---|---|---|---|---|")
    for row in audit_df.itertuples(index=False):
        lines.append(
            f"| {row.requested_query} | {row.query_type} | {int(row.resolved_flag)} | "
            f"{row.resolution_method} | {row.matched_gwas_lead_variant_id} | "
            f"{row.matched_gwas_lead_variant_rsids} | {row.matched_gwas_study_locus_id} | {row.notes} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_baseline_model_dir(target_model_dir: Path, explicit_baseline_dir: Path | None) -> Path:
    if explicit_baseline_dir is not None:
        return Path(explicit_baseline_dir)

    inferred = Path(target_model_dir).parent / "mode_none"
    if inferred.exists():
        return inferred

    if target_model_dir.name == "mode_none":
        return target_model_dir

    raise FileNotFoundError(
        "Could not infer baseline mode_none directory. "
        "Provide it explicitly via --baseline-model-dir."
    )


def load_inference_artifacts(model_dir: Path) -> InferenceArtifacts:
    coef_path = model_dir / "coefficient_table.csv"
    scaler_path = model_dir / "scaler_feature_stats.csv"
    params_path = model_dir / "model_parameters.json"
    feature_lists_path = model_dir / "feature_lists.json"

    for p in [coef_path, scaler_path, params_path, feature_lists_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required model artifact: {p}")

    coef_df = pd.read_csv(coef_path)
    scaler_df = pd.read_csv(scaler_path)
    with open(params_path, "r", encoding="utf-8") as f:
        model_params = json.load(f)
    with open(feature_lists_path, "r", encoding="utf-8") as f:
        feature_lists = json.load(f)

    mode = str(model_params.get("mode", "")).strip().lower()
    if mode not in {"none", "pca", "full", "residual_pca"}:
        raise ValueError(f"Unsupported model mode in {model_dir}: {mode!r}")

    feature_order = scaler_df["feature"].astype(str).tolist()
    coef_map = dict(zip(coef_df["feature"].astype(str), pd.to_numeric(coef_df["coefficient"], errors="coerce")))
    coefficients = np.asarray([float(coef_map.get(f, 0.0)) for f in feature_order], dtype=np.float64)
    intercept = float(model_params.get("model_intercept", 0.0))
    scaler_mean = pd.to_numeric(scaler_df["scaler_mean"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    scaler_scale = pd.to_numeric(scaler_df["scaler_scale"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)

    baseline_features = [str(x) for x in feature_lists.get("baseline_features_used_in_mode", [])]
    embedding_feature_names = [str(x) for x in feature_lists.get("embedding_features_available", [])]
    pca_feature_names: List[str] = []
    emb_scaler_mean: np.ndarray | None = None
    emb_scaler_scale: np.ndarray | None = None
    pca_components: np.ndarray | None = None
    pca_mean: np.ndarray | None = None
    residual_coefficients: np.ndarray | None = None
    residual_missing_embedding_zero = False

    if mode == "none":
        embedding_feature_names = []
    elif mode == "pca":
        pca_path = model_dir / "pca_transformer_artifacts.npz"
        if not pca_path.exists():
            raise FileNotFoundError(f"Missing required PCA artifact: {pca_path}")
        pca_npz = np.load(pca_path, allow_pickle=True)
        embedding_feature_names = [str(x) for x in pca_npz["embedding_feature_names"].tolist()]
        pca_feature_names = [str(x) for x in pca_npz["pca_feature_names"].tolist()]
        emb_scaler_mean = np.asarray(pca_npz["emb_scaler_mean"], dtype=np.float64)
        emb_scaler_scale = np.asarray(pca_npz["emb_scaler_scale"], dtype=np.float64)
        pca_components = np.asarray(pca_npz["pca_components"], dtype=np.float64)
        pca_mean = np.asarray(pca_npz["pca_mean"], dtype=np.float64)
    elif mode == "full":
        embedding_feature_names = [str(x) for x in feature_lists.get("embedding_features_available", [])]
    elif mode == "residual_pca":
        residual_path = model_dir / "residual_pca_artifacts.npz"
        if not residual_path.exists():
            raise FileNotFoundError(f"Missing required residual artifacts: {residual_path}")
        residual_npz = np.load(residual_path, allow_pickle=True)
        baseline_features = [str(x) for x in residual_npz["baseline_feature_names"].tolist()]
        feature_order = list(baseline_features)
        coefficients = np.asarray(residual_npz["baseline_coefficients"], dtype=np.float64)
        intercept = float(np.asarray(residual_npz["baseline_intercept"], dtype=np.float64).reshape(-1)[0])
        scaler_mean = np.asarray(residual_npz["baseline_scaler_mean"], dtype=np.float64)
        scaler_scale = np.asarray(residual_npz["baseline_scaler_scale"], dtype=np.float64)
        embedding_feature_names = [str(x) for x in residual_npz["embedding_feature_names"].tolist()]
        pca_feature_names = [str(x) for x in residual_npz["residual_pca_feature_names"].tolist()]
        emb_scaler_mean = np.asarray(residual_npz["emb_scaler_mean"], dtype=np.float64)
        emb_scaler_scale = np.asarray(residual_npz["emb_scaler_scale"], dtype=np.float64)
        pca_components = np.asarray(residual_npz["pca_components"], dtype=np.float64)
        pca_mean = np.asarray(residual_npz["pca_mean"], dtype=np.float64)
        residual_coefficients = np.asarray(residual_npz["residual_coefficients"], dtype=np.float64)
        residual_missing_embedding_zero = bool(
            int(np.asarray(residual_npz["missing_embedding_residual_zero"], dtype=np.int64).reshape(-1)[0])
        )

    return InferenceArtifacts(
        model_dir=model_dir,
        mode=mode,
        feature_order=feature_order,
        coefficients=coefficients,
        intercept=intercept,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        baseline_features=baseline_features,
        embedding_feature_names=embedding_feature_names,
        pca_feature_names=pca_feature_names,
        emb_scaler_mean=emb_scaler_mean,
        emb_scaler_scale=emb_scaler_scale,
        pca_components=pca_components,
        pca_mean=pca_mean,
        residual_coefficients=residual_coefficients,
        residual_missing_embedding_zero=residual_missing_embedding_zero,
    )


def build_model_input_frame(df: pd.DataFrame, art: InferenceArtifacts) -> pd.DataFrame:
    working = df.copy()
    if art.mode == "pca":
        needed_cols = list(art.baseline_features) + list(art.embedding_feature_names)
        for col in needed_cols:
            if col not in working.columns:
                working[col] = 0.0

        if (
            art.emb_scaler_mean is None
            or art.emb_scaler_scale is None
            or art.pca_components is None
            or art.pca_mean is None
        ):
            raise ValueError(f"PCA artifacts are missing for mode={art.mode} in {art.model_dir}")

        x_base = as_numeric_matrix(working, art.baseline_features, fill_value=0.0)
        x_emb = as_numeric_matrix(working, art.embedding_feature_names, fill_value=0.0)
        x_emb_scaled = _safe_scale(x_emb - art.emb_scaler_mean.reshape(1, -1), art.emb_scaler_scale)
        x_emb_pca = (x_emb_scaled - art.pca_mean.reshape(1, -1)) @ art.pca_components.T

        base_df = pd.DataFrame(x_base, columns=art.baseline_features, index=working.index)
        pca_df = pd.DataFrame(x_emb_pca, columns=art.pca_feature_names, index=working.index)
        model_input = pd.concat([base_df, pca_df], axis=1)
    elif art.mode == "residual_pca":
        needed_cols = list(art.feature_order) + list(art.embedding_feature_names)
        for col in needed_cols:
            if col not in working.columns:
                working[col] = 0.0
        if (
            art.emb_scaler_mean is None
            or art.emb_scaler_scale is None
            or art.pca_components is None
            or art.pca_mean is None
        ):
            raise ValueError(f"Residual PCA artifacts are missing for mode={art.mode} in {art.model_dir}")
        x_base = as_numeric_matrix(working, art.feature_order, fill_value=0.0)
        x_emb = as_numeric_matrix(working, art.embedding_feature_names, fill_value=0.0)
        x_emb_scaled = _safe_scale(x_emb - art.emb_scaler_mean.reshape(1, -1), art.emb_scaler_scale)
        x_emb_pca = (x_emb_scaled - art.pca_mean.reshape(1, -1)) @ art.pca_components.T
        has_emb = _infer_has_embedding_mask(working, art.embedding_feature_names)
        if bool(art.residual_missing_embedding_zero):
            x_emb_pca = np.asarray(x_emb_pca, dtype=np.float64)
            x_emb_pca[~has_emb, :] = 0.0
        base_df = pd.DataFrame(x_base, columns=art.feature_order, index=working.index)
        pca_df = pd.DataFrame(x_emb_pca, columns=art.pca_feature_names, index=working.index)
        model_input = pd.concat([base_df, pca_df], axis=1)
        model_input["__has_embedding_for_residual"] = has_emb.astype(int)
    else:
        for col in art.feature_order:
            if col not in working.columns:
                working[col] = 0.0
        x = as_numeric_matrix(working, art.feature_order, fill_value=0.0)
        model_input = pd.DataFrame(x, columns=art.feature_order, index=working.index)

    if art.mode == "residual_pca":
        residual_cols = list(art.pca_feature_names)
        for col in list(art.feature_order) + residual_cols:
            if col not in model_input.columns:
                model_input[col] = 0.0
        if "__has_embedding_for_residual" not in model_input.columns:
            model_input["__has_embedding_for_residual"] = 0
        ordered = list(art.feature_order) + residual_cols + ["__has_embedding_for_residual"]
        return model_input.loc[:, ordered].copy()

    for col in art.feature_order:
        if col not in model_input.columns:
            model_input[col] = 0.0
    return model_input.loc[:, art.feature_order].copy()


def rank_within_locus(scored_df: pd.DataFrame, score_col: str) -> pd.Series:
    return scored_df.groupby([LOCUS_COL])[score_col].rank(method="first", ascending=False).astype(int)


def score_rows(model_input: pd.DataFrame, art: InferenceArtifacts) -> pd.DataFrame:
    if art.mode == "residual_pca":
        if art.residual_coefficients is None:
            raise ValueError(f"Residual coefficients missing for mode={art.mode} in {art.model_dir}")
        x_base = model_input.loc[:, art.feature_order].to_numpy(dtype=np.float64)
        z_base = _safe_scale(x_base - art.scaler_mean.reshape(1, -1), art.scaler_scale)
        baseline_linear = art.intercept + (z_base @ art.coefficients.reshape(-1, 1)).ravel()

        residual_cols = list(art.pca_feature_names)
        x_resid = (
            model_input.loc[:, residual_cols].to_numpy(dtype=np.float64)
            if residual_cols
            else np.zeros((len(model_input), 0), dtype=np.float64)
        )
        residual_linear = (
            (x_resid @ art.residual_coefficients.reshape(-1, 1)).ravel()
            if art.residual_coefficients.size > 0
            else np.zeros(len(model_input), dtype=np.float64)
        )
        if bool(art.residual_missing_embedding_zero) and "__has_embedding_for_residual" in model_input.columns:
            has_emb = pd.to_numeric(model_input["__has_embedding_for_residual"], errors="coerce").fillna(0).to_numpy(dtype=float)
            residual_linear[has_emb < 0.5] = 0.0
        final_linear = baseline_linear + residual_linear
        final_score = _sigmoid(final_linear)
        return pd.DataFrame(
            {
                "baseline_predicted_linear_score": baseline_linear,
                "embedding_residual_linear_score": residual_linear,
                "final_predicted_linear_score": final_linear,
                "final_predicted_score": final_score,
                "predicted_linear_score": final_linear,
                "predicted_score": final_score,
                "has_embedding_for_residual": (
                    pd.to_numeric(model_input["__has_embedding_for_residual"], errors="coerce").fillna(0).astype(int)
                    if "__has_embedding_for_residual" in model_input.columns
                    else pd.Series(np.zeros(len(model_input), dtype=int), index=model_input.index)
                ),
            },
            index=model_input.index,
        )

    x = model_input.to_numpy(dtype=np.float64)
    z = _safe_scale(x - art.scaler_mean.reshape(1, -1), art.scaler_scale)
    linear = art.intercept + (z @ art.coefficients.reshape(-1, 1)).ravel()
    score = _sigmoid(linear)
    return pd.DataFrame(
        {
            "predicted_linear_score": linear,
            "predicted_score": score,
        },
        index=model_input.index,
    )


def build_saturation_diagnostics(scored_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    model_specs = [
        ("baseline", "baseline_predicted_linear_score", "baseline_predicted_score"),
        ("full", "full_predicted_linear_score", "full_predicted_score"),
    ]

    def add_row(
        *,
        model_label: str,
        linear_col: str,
        score_col: str,
        scope: str,
        variant_id: str,
        locus_id: str,
        chunk: pd.DataFrame,
    ) -> None:
        linear = pd.to_numeric(chunk[linear_col], errors="coerce").to_numpy(dtype=np.float64)
        score = pd.to_numeric(chunk[score_col], errors="coerce").to_numpy(dtype=np.float64)
        valid_linear = np.isfinite(linear)
        valid_score = np.isfinite(score)
        n_rows = int(len(chunk))
        n_linear_lt_minus50 = int(np.sum(valid_linear & (linear < -50.0)))
        n_score_floor = int(
            np.sum(
                valid_score
                & np.isclose(
                    score,
                    SIGMOID_FLOOR,
                    rtol=0.0,
                    atol=max(1e-30, float(SIGMOID_FLOOR) * 1e-9),
                )
            )
        )
        frac_linear_lt_minus50 = float(n_linear_lt_minus50 / n_rows) if n_rows > 0 else float("nan")
        frac_score_floor = float(n_score_floor / n_rows) if n_rows > 0 else float("nan")
        n_unique_score = int(pd.Series(score[valid_score]).nunique(dropna=True)) if np.any(valid_score) else 0
        n_unique_linear = int(pd.Series(linear[valid_linear]).nunique(dropna=True)) if np.any(valid_linear) else 0

        rows.append(
            {
                "scope": scope,
                "variant_id": variant_id,
                LOCUS_COL: locus_id,
                "model_label": model_label,
                "n_rows": n_rows,
                "predicted_linear_min": float(np.nanmin(linear)) if np.any(valid_linear) else float("nan"),
                "predicted_linear_max": float(np.nanmax(linear)) if np.any(valid_linear) else float("nan"),
                "n_linear_lt_minus50": n_linear_lt_minus50,
                "frac_linear_lt_minus50": frac_linear_lt_minus50,
                "sigmoid_floor_value": float(SIGMOID_FLOOR),
                "n_score_at_sigmoid_floor": n_score_floor,
                "frac_score_at_sigmoid_floor": frac_score_floor,
                "n_unique_predicted_score": n_unique_score,
                "n_unique_predicted_linear_score": n_unique_linear,
            }
        )

    for model_label, linear_col, score_col in model_specs:
        add_row(
            model_label=model_label,
            linear_col=linear_col,
            score_col=score_col,
            scope="overall",
            variant_id="",
            locus_id="",
            chunk=scored_df,
        )
        for (variant_id, locus_id), chunk in scored_df.groupby(["variant_id", LOCUS_COL], sort=True):
            add_row(
                model_label=model_label,
                linear_col=linear_col,
                score_col=score_col,
                scope="per_locus",
                variant_id=str(variant_id),
                locus_id=str(locus_id),
                chunk=chunk,
            )

    return pd.DataFrame(rows)


def sanitize_name(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(text))
    return safe[:200]


def _to_has_embedding_flag(value: object) -> int:
    try:
        return 1 if float(value) > 0.5 else 0
    except (TypeError, ValueError):
        return 0


def _row_has_embedding_flag(row: object) -> int:
    return _to_has_embedding_flag(getattr(row, "has_gene_embedding", 0))


def summarize_locus_markdown(
    locus_df: pd.DataFrame,
    *,
    label_id: str,
    variant_id: str,
    locus_id: str,
    baseline_mode: str,
    full_mode: str,
    output_path: Path,
) -> None:
    lines: List[str] = []
    lines.append(f"# Locus Inference Summary: {label_id}")
    lines.append("")
    lines.append(f"- Query label: `{label_id}`")
    lines.append(f"- Variant ID: `{variant_id}`")
    lines.append(f"- Locus ID: `{locus_id}`")
    lines.append(f"- Candidate genes: `{len(locus_df)}`")
    lines.append(f"- Baseline mode: `{baseline_mode}`")
    lines.append(f"- Full/target mode: `{full_mode}`")
    lines.append("")

    lines.append("## Top 5 Genes by Baseline Model")
    lines.append("")
    lines.append("| baseline_rank | gene_symbol | has_embedding | gene_id | baseline_linear | baseline_predicted_score |")
    lines.append("|---:|---|---:|---|---:|---:|")
    top_baseline = locus_df.sort_values(
        ["baseline_rank_within_locus", "baseline_predicted_linear_score"],
        ascending=[True, False],
        kind="stable",
    ).head(5)
    for row in top_baseline.itertuples(index=False):
        has_embedding = _row_has_embedding_flag(row)
        lines.append(
            f"| {int(row.baseline_rank_within_locus)} | {row.gene_symbol} | {has_embedding} | {row.gene_id} | "
            f"{float(row.baseline_predicted_linear_score):.6f} | {float(row.baseline_predicted_score):.6f} |"
        )
    lines.append("")

    lines.append("## Top 5 Genes by Full Model")
    lines.append("")
    lines.append("| full_rank | gene_symbol | has_embedding | gene_id | full_linear | full_predicted_score |")
    lines.append("|---:|---|---:|---|---:|---:|")
    top_full = locus_df.sort_values(
        ["full_rank_within_locus", "full_predicted_linear_score"],
        ascending=[True, False],
        kind="stable",
    ).head(5)
    for row in top_full.itertuples(index=False):
        has_embedding = _row_has_embedding_flag(row)
        lines.append(
            f"| {int(row.full_rank_within_locus)} | {row.gene_symbol} | {has_embedding} | {row.gene_id} | "
            f"{float(row.full_predicted_linear_score):.6f} | {float(row.full_predicted_score):.6f} |"
        )
    lines.append("")

    lines.append("## Largest Rank Improvements (baseline -> full)")
    lines.append("")
    lines.append("| rank_delta | baseline_rank | full_rank | gene_symbol | has_embedding | gene_id |")
    lines.append("|---:|---:|---:|---|---:|---|")
    improved = locus_df.loc[locus_df["rank_delta"] > 0].copy()
    if improved.empty:
        lines.append("| 0 | - | - | (none) | - | (none) |")
    else:
        improved = improved.sort_values(
            ["rank_delta", "full_rank_within_locus", "full_predicted_linear_score"],
            ascending=[False, True, False],
            kind="stable",
        ).head(5)
        for row in improved.itertuples(index=False):
            has_embedding = _row_has_embedding_flag(row)
            lines.append(
                f"| {int(row.rank_delta)} | {int(row.baseline_rank_within_locus)} | "
                f"{int(row.full_rank_within_locus)} | {row.gene_symbol} | {has_embedding} | {row.gene_id} |"
            )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    variant_ids, rsids, study_locus_ids = _resolve_requested_inputs(args)
    if not variant_ids and not rsids and not study_locus_ids:
        raise ValueError("No valid --variant-ids, --rsids, or --study-locus-ids were provided.")

    feature_table_path = _resolve_feature_table_path(args)
    feature_df = _load_table(feature_table_path)
    feature_df = impute_distance_features_worst_case(feature_df)

    if LEAD_VARIANT_COL not in feature_df.columns:
        raise ValueError(
            f"Feature table does not contain required column '{LEAD_VARIANT_COL}'."
        )
    feature_df[LEAD_VARIANT_COL] = feature_df[LEAD_VARIANT_COL].fillna("").astype(str)
    if LOCUS_COL not in feature_df.columns:
        raise ValueError(f"Feature table does not contain required column '{LOCUS_COL}'.")
    feature_df[LOCUS_COL] = feature_df[LOCUS_COL].fillna("").astype(str)
    if LEAD_RSIDS_COL not in feature_df.columns:
        feature_df[LEAD_RSIDS_COL] = ""
    feature_df[LEAD_RSIDS_COL] = feature_df[LEAD_RSIDS_COL].fillna("").astype(str)

    resolution_audit_df = resolve_requested_loci(
        feature_df,
        requested_variant_ids=variant_ids,
        requested_rsids=rsids,
        requested_study_locus_ids=study_locus_ids,
    )
    resolution_audit_csv_path = args.out_dir / "locus_resolution_audit.csv"
    resolution_audit_md_path = args.out_dir / "locus_resolution_audit.md"
    resolution_audit_df.to_csv(resolution_audit_csv_path, index=False)
    write_resolution_audit_markdown(resolution_audit_df, resolution_audit_md_path)

    resolved_audit_df = resolution_audit_df.loc[resolution_audit_df["resolved_flag"] == 1].copy()
    unresolved_audit_df = resolution_audit_df.loc[resolution_audit_df["resolved_flag"] == 0].copy()

    resolved_locus_ids = sorted(
        resolved_audit_df["matched_gwas_study_locus_id"].fillna("").astype(str).unique().tolist()
    )
    resolved_locus_ids = [x for x in resolved_locus_ids if x]

    resolved_variant_ids = sorted(
        resolved_audit_df.loc[
            resolved_audit_df["query_type"] == "variant_id", "requested_query"
        ].astype(str).tolist()
    )
    missing_variants = sorted(
        unresolved_audit_df.loc[
            unresolved_audit_df["query_type"] == "variant_id", "requested_query"
        ].astype(str).tolist()
    )
    resolved_rsids = sorted(
        resolved_audit_df.loc[
            resolved_audit_df["query_type"] == "rsid", "requested_query"
        ].astype(str).tolist()
    )
    missing_rsids = sorted(
        unresolved_audit_df.loc[
            unresolved_audit_df["query_type"] == "rsid", "requested_query"
        ].astype(str).tolist()
    )
    resolved_study_locus_ids = sorted(
        resolved_audit_df.loc[
            resolved_audit_df["query_type"] == "study_locus_id", "requested_query"
        ].astype(str).tolist()
    )
    missing_study_locus_ids = sorted(
        unresolved_audit_df.loc[
            unresolved_audit_df["query_type"] == "study_locus_id", "requested_query"
        ].astype(str).tolist()
    )

    subset = feature_df.loc[feature_df[LOCUS_COL].isin(resolved_locus_ids)].copy()

    locus_to_queries = (
        resolved_audit_df.loc[
            resolved_audit_df["matched_gwas_study_locus_id"].astype(str).str.len() > 0,
            ["matched_gwas_study_locus_id", "requested_query"],
        ]
        .drop_duplicates()
        .groupby("matched_gwas_study_locus_id", sort=True)["requested_query"]
        .apply(lambda s: "|".join(s.astype(str).tolist()))
        .to_dict()
    )
    locus_to_rsids = (
        resolved_audit_df.loc[
            (resolved_audit_df["query_type"] == "rsid")
            & (resolved_audit_df["matched_gwas_study_locus_id"].astype(str).str.len() > 0),
            ["matched_gwas_study_locus_id", "requested_query"],
        ]
        .drop_duplicates()
        .groupby("matched_gwas_study_locus_id", sort=True)["requested_query"]
        .apply(lambda s: "|".join(s.astype(str).tolist()))
        .to_dict()
    )
    if not subset.empty:
        subset["matched_requested_queries"] = subset[LOCUS_COL].map(locus_to_queries).fillna("")
        subset["matched_requested_rsids"] = subset[LOCUS_COL].map(locus_to_rsids).fillna("")
    else:
        subset["matched_requested_queries"] = []
        subset["matched_requested_rsids"] = []

    target_model_dir = Path(args.model_dir)
    baseline_model_dir = _resolve_baseline_model_dir(target_model_dir, args.baseline_model_dir)
    baseline_artifacts = load_inference_artifacts(baseline_model_dir)
    full_artifacts = load_inference_artifacts(target_model_dir)

    if baseline_artifacts.mode != "none":
        raise ValueError(
            f"Baseline model must be mode='none'. Got mode={baseline_artifacts.mode!r} in {baseline_model_dir}."
        )

    report_lines: List[str] = []
    report_lines.append("# Specific-Loci Inference Report")
    report_lines.append("")
    report_lines.append(f"- Baseline model directory: `{baseline_model_dir}` (mode=`{baseline_artifacts.mode}`)")
    report_lines.append(f"- Full/target model directory: `{target_model_dir}` (mode=`{full_artifacts.mode}`)")
    report_lines.append(f"- Feature table used: `{feature_table_path}`")
    report_lines.append(f"- Requested variants: `{', '.join(variant_ids) if variant_ids else 'none'}`")
    report_lines.append(f"- Resolved variants: `{', '.join(resolved_variant_ids) if resolved_variant_ids else 'none'}`")
    report_lines.append(f"- Unresolved variants: `{', '.join(missing_variants) if missing_variants else 'none'}`")
    report_lines.append(f"- Requested rsIDs: `{', '.join(rsids) if rsids else 'none'}`")
    report_lines.append(f"- Resolved rsIDs: `{', '.join(resolved_rsids) if resolved_rsids else 'none'}`")
    report_lines.append(f"- Unresolved rsIDs: `{', '.join(missing_rsids) if missing_rsids else 'none'}`")
    report_lines.append(
        f"- Requested study locus IDs: `{', '.join(study_locus_ids) if study_locus_ids else 'none'}`"
    )
    report_lines.append(
        f"- Resolved study locus IDs: `{', '.join(resolved_study_locus_ids) if resolved_study_locus_ids else 'none'}`"
    )
    report_lines.append(
        f"- Unresolved study locus IDs: `{', '.join(missing_study_locus_ids) if missing_study_locus_ids else 'none'}`"
    )
    report_lines.append(
        f"- Resolved loci in GCST90027164 feature table: `{', '.join(resolved_locus_ids) if resolved_locus_ids else 'none'}`"
    )
    report_lines.append(f"- Resolution audit CSV: `{resolution_audit_csv_path}`")
    report_lines.append(f"- Resolution audit Markdown: `{resolution_audit_md_path}`")
    report_lines.append("")

    if not unresolved_audit_df.empty:
        report_lines.append("## Unresolved Queries")
        report_lines.append("")
        report_lines.append("| requested_query | query_type | notes |")
        report_lines.append("|---|---|---|")
        unresolved_short = unresolved_audit_df.loc[:, ["requested_query", "query_type", "notes"]]
        unresolved_short = unresolved_short.sort_values(["query_type", "requested_query"], kind="stable")
        for row in unresolved_short.itertuples(index=False):
            report_lines.append(f"| {row.requested_query} | {row.query_type} | {row.notes} |")
        report_lines.append("")

    if subset.empty:
        report_lines.append("No queries resolved to GCST90027164 loci; no ranking files were produced.")
        (args.out_dir / "inference_summary.md").write_text("\n".join(report_lines), encoding="utf-8")
        print(f"[warn] No resolved loci found. Summary: {args.out_dir / 'inference_summary.md'}")
        return

    baseline_model_input = build_model_input_frame(subset, baseline_artifacts)
    baseline_score_df = score_rows(baseline_model_input, baseline_artifacts)
    full_model_input = build_model_input_frame(subset, full_artifacts)
    full_score_df = score_rows(full_model_input, full_artifacts)

    scored = subset.copy()
    scored["variant_id"] = scored[LEAD_VARIANT_COL].astype(str)
    if LEAD_RSIDS_COL in scored.columns:
        scored[LEAD_RSIDS_COL] = scored[LEAD_RSIDS_COL].fillna("").astype(str)
    else:
        scored[LEAD_RSIDS_COL] = ""
    scored["baseline_predicted_linear_score"] = baseline_score_df["predicted_linear_score"]
    scored["baseline_predicted_score"] = baseline_score_df["predicted_score"]
    scored["baseline_rank_within_locus"] = rank_within_locus(scored, "baseline_predicted_linear_score")
    scored["full_predicted_linear_score"] = full_score_df["predicted_linear_score"]
    scored["full_predicted_score"] = full_score_df["predicted_score"]
    if full_artifacts.mode == "residual_pca":
        if "baseline_predicted_linear_score" in full_score_df.columns:
            scored["baseline_predicted_linear_score"] = full_score_df["baseline_predicted_linear_score"]
            scored["baseline_predicted_score"] = _sigmoid(
                pd.to_numeric(scored["baseline_predicted_linear_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            )
        residual_series = (
            full_score_df["embedding_residual_linear_score"]
            if "embedding_residual_linear_score" in full_score_df.columns
            else pd.Series(np.zeros(len(full_score_df), dtype=float), index=full_score_df.index)
        )
        final_linear_series = (
            full_score_df["final_predicted_linear_score"]
            if "final_predicted_linear_score" in full_score_df.columns
            else full_score_df["predicted_linear_score"]
        )
        final_score_series = (
            full_score_df["final_predicted_score"]
            if "final_predicted_score" in full_score_df.columns
            else full_score_df["predicted_score"]
        )
        scored["embedding_residual_linear_score"] = pd.to_numeric(
            residual_series, errors="coerce"
        ).fillna(0.0)
        scored["final_predicted_linear_score"] = pd.to_numeric(
            final_linear_series,
            errors="coerce",
        ).fillna(0.0)
        scored["final_predicted_score"] = pd.to_numeric(
            final_score_series,
            errors="coerce",
        ).fillna(0.0)
        scored["full_predicted_linear_score"] = scored["final_predicted_linear_score"]
        scored["full_predicted_score"] = scored["final_predicted_score"]
        if "has_embedding_for_residual" in full_score_df.columns:
            scored["has_embedding_for_residual"] = pd.to_numeric(
                full_score_df["has_embedding_for_residual"], errors="coerce"
            ).fillna(0).astype(int)
    else:
        scored["embedding_residual_linear_score"] = 0.0
        scored["final_predicted_linear_score"] = scored["full_predicted_linear_score"]
        scored["final_predicted_score"] = scored["full_predicted_score"]
    scored["full_rank_within_locus"] = rank_within_locus(scored, "full_predicted_linear_score")
    scored["rank_delta"] = scored["baseline_rank_within_locus"] - scored["full_rank_within_locus"]
    scored["moved_up_flag"] = (scored["rank_delta"] > 0).astype(int)

    # Backward-compatible aliases: keep previous single-model columns mapped to full/target mode.
    scored["predicted_linear_score"] = scored["final_predicted_linear_score"]
    scored["predicted_score"] = scored["final_predicted_score"]
    scored["rank_within_locus"] = scored["full_rank_within_locus"]

    top_model_features: List[str] = []
    if full_artifacts.mode == "residual_pca" and full_artifacts.residual_coefficients is not None:
        combined_names = list(full_artifacts.feature_order) + list(full_artifacts.pca_feature_names)
        combined_coef = np.concatenate(
            [
                np.asarray(full_artifacts.coefficients, dtype=np.float64).reshape(-1),
                np.asarray(full_artifacts.residual_coefficients, dtype=np.float64).reshape(-1),
            ]
        )
        coef_abs_order = np.argsort(-np.abs(combined_coef))
        for idx in coef_abs_order.tolist():
            name = combined_names[idx]
            if len(top_model_features) >= int(args.max_model_input_features):
                break
            if name not in top_model_features:
                top_model_features.append(name)
    else:
        coef_abs_order = np.argsort(-np.abs(full_artifacts.coefficients))
        for idx in coef_abs_order.tolist():
            name = full_artifacts.feature_order[idx]
            if len(top_model_features) >= int(args.max_model_input_features):
                break
            if name not in top_model_features:
                top_model_features.append(name)
    for feat in top_model_features:
        scored[f"model_input_{feat}"] = full_model_input[feat].to_numpy(dtype=np.float64)

    keep_cols = [
        "variant_id",
        LOCUS_COL,
        GENE_SYMBOL_COL,
        GENE_ID_COL,
        LEAD_RSIDS_COL,
        "matched_requested_queries",
        "matched_requested_rsids",
        "baseline_predicted_linear_score",
        "baseline_predicted_score",
        "baseline_rank_within_locus",
        "embedding_residual_linear_score",
        "final_predicted_linear_score",
        "final_predicted_score",
        "full_predicted_linear_score",
        "full_predicted_score",
        "full_rank_within_locus",
        "rank_delta",
        "moved_up_flag",
        "has_embedding_for_residual",
        "predicted_linear_score",
        "predicted_score",
        "rank_within_locus",
        "dist_variant_to_gene_kb",
        "dist_variant_to_tss_kb",
        "colocalisation_h4_max",
        "colocalisation_clpp_max",
        "hpa_brain_expression_value",
        "hpa_muscle_expression_value",
        "has_gene_embedding",
    ] + [f"model_input_{feat}" for feat in top_model_features]
    keep_cols = [c for c in keep_cols if c in scored.columns]
    scored_export = scored.loc[:, keep_cols].copy()
    scored_export = scored_export.sort_values(
        by=["variant_id", LOCUS_COL, "full_rank_within_locus", "full_predicted_linear_score"],
        ascending=[True, True, True, False],
        kind="stable",
    ).reset_index(drop=True)

    combined_path = args.out_dir / "all_requested_loci_ranked_genes.csv"
    comparison_combined_path = args.out_dir / "all_requested_loci_ranked_genes_baseline_vs_full.csv"
    scored_export.to_csv(combined_path, index=False)
    scored_export.to_csv(comparison_combined_path, index=False)

    per_locus_paths: List[Path] = []
    rsid_paths: List[Path] = []
    for (variant_id, locus_id), locus_df in scored_export.groupby(["variant_id", LOCUS_COL], sort=True):
        variant_safe = sanitize_name(variant_id)
        locus_safe = sanitize_name(locus_id)
        csv_path = args.out_dir / f"{variant_safe}__{locus_safe}_ranked_genes.csv"
        md_path = args.out_dir / f"{variant_safe}__{locus_safe}_summary.md"
        locus_df.sort_values(
            ["full_rank_within_locus", "full_predicted_linear_score"],
            ascending=[True, False],
            kind="stable",
        ).to_csv(csv_path, index=False)
        summarize_locus_markdown(
            locus_df=locus_df,
            label_id=variant_id,
            variant_id=variant_id,
            locus_id=locus_id,
            baseline_mode=baseline_artifacts.mode,
            full_mode=full_artifacts.mode,
            output_path=md_path,
        )
        per_locus_paths.extend([csv_path, md_path])

    resolved_rsid_queries = resolved_audit_df.loc[
        resolved_audit_df["query_type"] == "rsid",
        ["requested_query", "matched_gwas_lead_variant_id", "matched_gwas_study_locus_id"],
    ].drop_duplicates()
    for row in resolved_rsid_queries.itertuples(index=False):
        rsid = str(row.requested_query)
        variant_id = str(row.matched_gwas_lead_variant_id)
        locus_id = str(row.matched_gwas_study_locus_id)
        locus_df = scored_export.loc[
            (scored_export["variant_id"] == variant_id)
            & (scored_export[LOCUS_COL] == locus_id)
        ].copy()
        if locus_df.empty:
            continue
        rsid_safe = sanitize_name(rsid)
        variant_safe = sanitize_name(variant_id)
        locus_safe = sanitize_name(locus_id)
        csv_path = args.out_dir / f"rsid_{rsid_safe}__{variant_safe}__{locus_safe}_ranked_genes.csv"
        md_path = args.out_dir / f"rsid_{rsid_safe}__{variant_safe}__{locus_safe}_summary.md"
        locus_df.sort_values(
            ["full_rank_within_locus", "full_predicted_linear_score"],
            ascending=[True, False],
            kind="stable",
        ).to_csv(csv_path, index=False)
        summarize_locus_markdown(
            locus_df=locus_df,
            label_id=f"rsid:{rsid}",
            variant_id=variant_id,
            locus_id=locus_id,
            baseline_mode=baseline_artifacts.mode,
            full_mode=full_artifacts.mode,
            output_path=md_path,
        )
        rsid_paths.extend([csv_path, md_path])

    report_lines.append(f"- Combined output CSV: `{combined_path}`")
    report_lines.append(f"- Comparison combined CSV: `{comparison_combined_path}`")
    report_lines.append(f"- Per-locus files generated (variant/locus): `{len(per_locus_paths)}`")
    report_lines.append(f"- Per-rsID files generated: `{len(rsid_paths)}`")
    report_lines.append("")

    if full_artifacts.mode == "residual_pca" and {
        "has_embedding_for_residual",
        "embedding_residual_linear_score",
    }.issubset(set(scored_export.columns)):
        has_emb = pd.to_numeric(scored_export["has_embedding_for_residual"], errors="coerce").fillna(0).to_numpy(dtype=float)
        resid = pd.to_numeric(scored_export["embedding_residual_linear_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        missing_mask = has_emb < 0.5
        max_abs_missing = float(np.max(np.abs(resid[missing_mask]))) if np.any(missing_mask) else 0.0
        report_lines.append(
            "- Residual neutral-check on missing embeddings: "
            f"rows_missing_embedding={int(np.sum(missing_mask))}, "
            f"max_abs_embedding_residual_linear_score={max_abs_missing:.12f}"
        )
        report_lines.append("")

    saturation_df = build_saturation_diagnostics(scored_export)
    saturation_csv_path = args.out_dir / "saturation_diagnostics.csv"
    saturation_json_path = args.out_dir / "saturation_diagnostics.json"
    saturation_df.to_csv(saturation_csv_path, index=False)
    with open(saturation_json_path, "w", encoding="utf-8") as f:
        json.dump(saturation_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    report_lines.append(f"- Saturation diagnostics CSV: `{saturation_csv_path}`")
    report_lines.append(f"- Saturation diagnostics JSON: `{saturation_json_path}`")
    report_lines.append("")
    report_lines.append("## Saturation Diagnostics (Overall)")
    report_lines.append("")
    report_lines.append(
        "| model | n_rows | linear_min | linear_max | n_linear<-50 | frac_linear<-50 | "
        "n_score_at_floor | frac_score_at_floor | unique_linear | unique_score |"
    )
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    overall_diag = saturation_df.loc[saturation_df["scope"] == "overall"].copy()
    for row in overall_diag.itertuples(index=False):
        report_lines.append(
            f"| {row.model_label} | {int(row.n_rows)} | {float(row.predicted_linear_min):.6f} | "
            f"{float(row.predicted_linear_max):.6f} | {int(row.n_linear_lt_minus50)} | "
            f"{float(row.frac_linear_lt_minus50):.3f} | {int(row.n_score_at_sigmoid_floor)} | "
            f"{float(row.frac_score_at_sigmoid_floor):.3f} | "
            f"{int(row.n_unique_predicted_linear_score)} | {int(row.n_unique_predicted_score)} |"
        )
    report_lines.append("")

    report_lines.append("## Per-Locus Comparison Highlights")
    report_lines.append("")
    for (variant_id, locus_id), locus_df in scored_export.groupby(["variant_id", LOCUS_COL], sort=True):
        report_lines.append(f"### {variant_id} / {locus_id}")
        report_lines.append("- Top 5 baseline:")
        top5_baseline = locus_df.sort_values(
            ["baseline_rank_within_locus", "baseline_predicted_linear_score"],
            ascending=[True, False],
            kind="stable",
        ).head(5)
        for row in top5_baseline.itertuples(index=False):
            has_embedding = _row_has_embedding_flag(row)
            report_lines.append(
                f"  - rank={int(row.baseline_rank_within_locus)} gene={row.gene_symbol} "
                f"has_embedding={has_embedding} "
                f"linear={float(row.baseline_predicted_linear_score):.6f} "
                f"score={float(row.baseline_predicted_score):.6f}"
            )
        report_lines.append("- Top 5 full:")
        top5_full = locus_df.sort_values(
            ["full_rank_within_locus", "full_predicted_linear_score"],
            ascending=[True, False],
            kind="stable",
        ).head(5)
        for row in top5_full.itertuples(index=False):
            has_embedding = _row_has_embedding_flag(row)
            report_lines.append(
                f"  - rank={int(row.full_rank_within_locus)} gene={row.gene_symbol} "
                f"has_embedding={has_embedding} "
                f"linear={float(row.full_predicted_linear_score):.6f} "
                f"score={float(row.full_predicted_score):.6f}"
            )
        report_lines.append("- Largest rank improvements (baseline -> full):")
        top_improved = locus_df.loc[locus_df["rank_delta"] > 0].sort_values(
            ["rank_delta", "full_rank_within_locus", "full_predicted_linear_score"],
            ascending=[False, True, False],
            kind="stable",
        ).head(5)
        if top_improved.empty:
            report_lines.append("  - none")
        for row in top_improved.itertuples(index=False):
            has_embedding = _row_has_embedding_flag(row)
            report_lines.append(
                f"  - delta={int(row.rank_delta)} gene={row.gene_symbol} "
                f"has_embedding={has_embedding} "
                f"(baseline={int(row.baseline_rank_within_locus)} -> full={int(row.full_rank_within_locus)})"
            )
        report_lines.append("")

    summary_path = args.out_dir / "inference_summary.md"
    summary_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[done] Wrote combined CSV: {combined_path}")
    print(f"[done] Wrote comparison CSV: {comparison_combined_path}")
    for p in per_locus_paths:
        print(f"[done] Wrote: {p}")
    for p in rsid_paths:
        print(f"[done] Wrote: {p}")
    print(f"[done] Wrote summary: {summary_path}")
    if not unresolved_audit_df.empty:
        unresolved_queries = unresolved_audit_df["requested_query"].astype(str).tolist()
        print(f"[warn] Unresolved queries: {', '.join(unresolved_queries)}")


if __name__ == "__main__":
    main()

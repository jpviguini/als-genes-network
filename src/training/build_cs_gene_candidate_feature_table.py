#!/usr/bin/env python3
"""
Build gene-candidate feature table for one GWAS study (Open Targets), using a
500kb window around each GWAS lead variant.

Pipeline (gene-centric):
1. Fetch GWAS credible sets for one study.
2. Fetch molQTL colocalisation records for each GWAS credible set.
3. Build candidate genes from a local protein-coding GTF within +/-500kb.
4. Aggregate QTL/colocalisation evidence to (gwas_cs_id, gene).
5. Compute distance features and distScore.
6. Attach gene embeddings.
7. Attach ClinVar/EVA label for ALS (positive if EVA score >= 0.5).

Open Targets fields used:
- study(studyId).credibleSets rows (GWAS CS metadata + lead variant)
- credibleSet(studyLocusId).colocalisation rows (h4, clpp, tissue, QTL locus)
- target(ensemblId) (only to resolve missing gene symbols in colocalisation)
- disease(efoId).associatedTargets rows with datasourceScores (ClinVar/EVA labels)

Features computed directly:
- dist features from lead variant to gene body/TSS
- dist_score_500kb_log
- coloc features (h4/clpp summaries, qtl/tissue counts)
- coloc_score (thresholded from max H4)
- gene embeddings and has_gene_embedding
- ClinVar/EVA score and binary label (label_positive)

Features not computed directly (set to default 0 for now):
- coding_score_sum_pip
- coding_variant_count
- expression_score

No interaction-network feature is implemented in this script.
"""

from __future__ import annotations

import gzip
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests


DEFAULT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_STUDY_ID = "GCST90027164"
DEFAULT_DISEASE_ID = "MONDO_0004976"
DEFAULT_DATASOURCE_ID = "eva"
DEFAULT_EVA_THRESHOLD = 0.5
DEFAULT_GTF_PATH = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/reference/gencode.v38.annotation.gtf.gz"
)
DEFAULT_EMBEDDING_PATH = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/features/"
    "featuresUPPER_pubmedbert_neurodegenerative_disease/features_ALS_pubmedbert.pkl"
)
DEFAULT_OUT_DIR = Path("/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables")
QTL_STUDY_TYPES = ["eqtl", "pqtl", "sqtl", "sceqtl", "scpqtl", "scsqtl", "tuqtl", "sctuqtl"]


@dataclass(frozen=True)
class PipelineConfig:
    study_id: str = DEFAULT_STUDY_ID
    disease_id: str = DEFAULT_DISEASE_ID
    datasource_id: str = DEFAULT_DATASOURCE_ID
    graphql_url: str = DEFAULT_GRAPHQL_URL
    gtf_path: Path = DEFAULT_GTF_PATH
    embedding_path: Path = DEFAULT_EMBEDDING_PATH
    out_dir: Path = DEFAULT_OUT_DIR
    window_bp: int = 500_000
    h4_threshold: float = 0.90
    eva_positive_threshold: float = DEFAULT_EVA_THRESHOLD
    page_size: int = 200
    timeout_sec: float = 60.0
    max_retries: int = 3
    embedding_fill_value: float = float("nan")
    write_csv: bool = True
    write_parquet: bool = True
    write_raw_coloc: bool = True


CONFIG = PipelineConfig()


STUDY_CREDIBLE_SETS_QUERY = """
query StudyCredibleSets($studyId: String!, $page: Pagination) {
  study(studyId: $studyId) {
    id
    traitFromSource
    publicationTitle
    publicationFirstAuthor
    publicationDate
    hasSumstats
    nSamples
    studyType
    credibleSets(page: $page) {
      count
      rows {
        studyLocusId
        studyId
        studyType
        chromosome
        position
        region
        credibleSetIndex
        pValueMantissa
        pValueExponent
        beta
        zScore
        confidence
        locusStart
        locusEnd
        qualityControls
        variant {
          id
          rsIds
          chromosome
          position
        }
      }
    }
  }
}
"""


CREDIBLE_SET_COLOCALISATION_QUERY = """
query CredibleSetColocalisation($id: String!, $page: Pagination, $studyTypes: [StudyTypeEnum!]) {
  credibleSet(studyLocusId: $id) {
    studyLocusId
    colocalisation(studyTypes: $studyTypes, page: $page) {
      count
      rows {
        h3
        h4
        clpp
        betaRatioSignAverage
        colocalisationMethod
        numberColocalisingVariants
        rightStudyType
        chromosome
        otherStudyLocus {
          studyLocusId
          studyId
          studyType
          qtlGeneId
          isTransQtl
          subStudyDescription
          variant {
            id
            rsIds
            chromosome
            position
          }
          study {
            id
            projectId
            studyType
            traitFromSource
            condition
            biosample {
              biosampleId
              biosampleName
            }
            target {
              id
              approvedSymbol
            }
          }
        }
      }
    }
  }
}
"""


TARGET_COORD_QUERY = """
query TargetCoord($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    genomicLocation {
      chromosome
      start
      end
      strand
    }
  }
}
"""


DISEASE_ASSOC_QUERY = """
query DiseaseClinVarAssociations($diseaseId: String!, $pageIndex: Int!, $pageSize: Int!) {
  disease(efoId: $diseaseId) {
    id
    name
    associatedTargets(page: { index: $pageIndex, size: $pageSize }) {
      count
      rows {
        target {
          id
          approvedSymbol
        }
        score
        datasourceScores {
          id
          score
        }
      }
    }
  }
}
"""


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return None


def normalize_chrom(chrom: object) -> Optional[str]:
    if chrom is None:
        return None
    s = str(chrom).strip()
    if not s:
        return None
    if s.lower().startswith("chr"):
        s = s[3:]
    s_u = s.upper()
    alias = {"23": "X", "24": "Y", "25": "MT", "M": "MT"}
    return alias.get(s_u, s_u)


def normalize_gene_symbol(gene: object) -> Optional[str]:
    if gene is None:
        return None
    s = str(gene).strip().upper()
    return s if s else None


def normalize_gene_id(gene_id: object) -> Optional[str]:
    if gene_id is None:
        return None
    s = str(gene_id).strip()
    if not s:
        return None
    return s.split(".", 1)[0]


def unique_join(values: Iterable[object], sep: str = "|", max_items: Optional[int] = None) -> str:
    uniq: List[str] = []
    seen = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        uniq.append(s)
        seen.add(s)
    uniq.sort()
    if max_items is not None:
        uniq = uniq[:max_items]
    return sep.join(uniq)


def nunique_non_null(series: pd.Series) -> int:
    return int(series.dropna().astype(str).str.strip().replace("", np.nan).dropna().nunique())


def any_true(series: pd.Series) -> bool:
    vals = [to_bool(v) for v in series.tolist()]
    return any(v is True for v in vals)


def count_true(series: pd.Series) -> int:
    vals = [to_bool(v) for v in series.tolist()]
    return int(sum(v is True for v in vals))


def interval_distance(pos: int, start: int, end: int) -> int:
    if pos < start:
        return start - pos
    if pos > end:
        return pos - end
    return 0


def dist_score_500kb_log(distance_bp: int, window_bp: int = 500_000, min_bp: int = 1_000) -> float:
    d = max(int(distance_bp), 0)
    if d <= min_bp:
        return 1.0
    if d >= window_bp:
        return 0.0
    denom = np.log10(window_bp) - np.log10(min_bp)
    if denom <= 0:
        return 0.0
    score = 1.0 - ((np.log10(d) - np.log10(min_bp)) / denom)
    return float(np.clip(score, 0.0, 1.0))


class GraphQLClient:
    def __init__(
        self,
        url: str,
        timeout_sec: float = 60.0,
        max_retries: int = 3,
        backoff_sec: float = 1.0,
    ) -> None:
        self.url = url
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.session = requests.Session()

    def execute(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"query": query, "variables": variables or {}}
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(self.url, json=payload, timeout=self.timeout_sec)
                response.raise_for_status()
                body = response.json()
                if body.get("errors"):
                    raise RuntimeError(f"GraphQL errors: {body['errors']}")
                data = body.get("data")
                if not isinstance(data, dict):
                    raise RuntimeError(f"Unexpected GraphQL response shape: {body}")
                return data
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_sec * attempt)
        raise RuntimeError(f"GraphQL request failed after {self.max_retries} attempts") from last_error


def parse_credible_set_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    variant = row.get("variant") if isinstance(row.get("variant"), Mapping) else {}
    return {
        "study_locus_id": row.get("studyLocusId"),
        "study_id": row.get("studyId"),
        "study_type": row.get("studyType"),
        "chromosome": normalize_chrom(row.get("chromosome")),
        "position": to_int(row.get("position")),
        "region": row.get("region"),
        "credible_set_index": to_int(row.get("credibleSetIndex")),
        "p_value_mantissa": to_float(row.get("pValueMantissa")),
        "p_value_exponent": to_int(row.get("pValueExponent")),
        "beta": to_float(row.get("beta")),
        "z_score": to_float(row.get("zScore")),
        "confidence": row.get("confidence"),
        "locus_start": to_int(row.get("locusStart")),
        "locus_end": to_int(row.get("locusEnd")),
        "quality_controls": unique_join(row.get("qualityControls", [])),
        "lead_variant_id": variant.get("id"),
        "lead_variant_rsids": unique_join(variant.get("rsIds", [])),
        "lead_variant_chromosome": normalize_chrom(variant.get("chromosome")),
        "lead_variant_position": to_int(variant.get("position")),
    }


def fetch_study_credible_sets(
    client: GraphQLClient,
    study_id: str,
    page_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    index = 0
    rows_all: List[Dict[str, Any]] = []
    study_meta: Dict[str, Any] = {}

    while True:
        data = client.execute(
            STUDY_CREDIBLE_SETS_QUERY,
            {"studyId": study_id, "page": {"index": index, "size": page_size}},
        )
        study = data.get("study")
        if study is None:
            raise ValueError(f"Study not found: {study_id}")
        if not isinstance(study, Mapping):
            raise ValueError(f"Unexpected study payload for {study_id}")

        if not study_meta:
            study_meta = {
                "study_id": study.get("id"),
                "trait_from_source": study.get("traitFromSource"),
                "publication_title": study.get("publicationTitle"),
                "publication_first_author": study.get("publicationFirstAuthor"),
                "publication_date": study.get("publicationDate"),
                "has_sumstats": study.get("hasSumstats"),
                "n_samples": study.get("nSamples"),
                "study_type": study.get("studyType"),
            }

        cs_obj = study.get("credibleSets") if isinstance(study.get("credibleSets"), Mapping) else {}
        page_rows = cs_obj.get("rows", [])
        if not isinstance(page_rows, list):
            raise ValueError("Expected list in study.credibleSets.rows")

        for row in page_rows:
            if isinstance(row, Mapping):
                rows_all.append(parse_credible_set_row(row))

        count = to_int(cs_obj.get("count"))
        if count is None:
            if len(page_rows) < page_size:
                break
        else:
            if len(rows_all) >= count:
                break

        if not page_rows:
            break
        index += 1

    return study_meta, rows_all


def parse_colocalisation_row(gwas_study_locus_id: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    other = row.get("otherStudyLocus") if isinstance(row.get("otherStudyLocus"), Mapping) else {}
    other_study = other.get("study") if isinstance(other.get("study"), Mapping) else {}
    other_biosample = other_study.get("biosample") if isinstance(other_study.get("biosample"), Mapping) else {}
    other_target = other_study.get("target") if isinstance(other_study.get("target"), Mapping) else {}

    gene_id_raw = other_target.get("id") or other.get("qtlGeneId")
    gene_id = normalize_gene_id(gene_id_raw)
    gene_symbol = normalize_gene_symbol(other_target.get("approvedSymbol"))

    return {
        "gwas_study_locus_id": gwas_study_locus_id,
        "qtl_study_locus_id": other.get("studyLocusId"),
        "qtl_study_id": other.get("studyId") or other_study.get("id"),
        "qtl_study_type": other.get("studyType") or other_study.get("studyType"),
        "gene_id": gene_id,
        "gene_symbol": gene_symbol,
        "qtl_is_trans_qtl": to_bool(other.get("isTransQtl")),
        "qtl_sub_study_description": other.get("subStudyDescription"),
        "qtl_project_id": other_study.get("projectId"),
        "qtl_trait_from_source": other_study.get("traitFromSource"),
        "qtl_condition": other_study.get("condition"),
        "qtl_biosample_id": other_biosample.get("biosampleId"),
        "qtl_biosample_name": other_biosample.get("biosampleName"),
        "colocalisation_h3": to_float(row.get("h3")),
        "colocalisation_h4": to_float(row.get("h4")),
        "colocalisation_clpp": to_float(row.get("clpp")),
        "number_colocalising_variants": to_int(row.get("numberColocalisingVariants")),
        "colocalisation_method": row.get("colocalisationMethod"),
    }


def fetch_colocalisations_for_gwas_cs(
    client: GraphQLClient,
    gwas_study_locus_id: str,
    page_size: int,
    qtl_study_types: Sequence[str],
) -> List[Dict[str, Any]]:
    index = 0
    records: List[Dict[str, Any]] = []

    while True:
        data = client.execute(
            CREDIBLE_SET_COLOCALISATION_QUERY,
            {
                "id": gwas_study_locus_id,
                "page": {"index": index, "size": page_size},
                "studyTypes": list(qtl_study_types),
            },
        )
        cs = data.get("credibleSet")
        if cs is None:
            break
        if not isinstance(cs, Mapping):
            raise ValueError(f"Unexpected colocalisation payload for {gwas_study_locus_id}")

        coloc_obj = cs.get("colocalisation") if isinstance(cs.get("colocalisation"), Mapping) else {}
        rows = coloc_obj.get("rows", [])
        if not isinstance(rows, list):
            raise ValueError(f"Expected list in colocalisation rows for {gwas_study_locus_id}")

        for row in rows:
            if not isinstance(row, Mapping):
                continue
            rec = parse_colocalisation_row(gwas_study_locus_id, row)
            if rec.get("qtl_study_locus_id"):
                records.append(rec)

        count = to_int(coloc_obj.get("count"))
        if count is None:
            if len(rows) < page_size:
                break
        else:
            if len(records) >= count:
                break

        if not rows:
            break
        index += 1

    return records


def resolve_missing_gene_symbols(records: List[Dict[str, Any]], client: GraphQLClient) -> int:
    to_resolve = sorted(
        {
            str(rec.get("gene_id")).strip()
            for rec in records
            if not rec.get("gene_symbol") and rec.get("gene_id")
        }
    )
    if not to_resolve:
        return 0

    cache: Dict[str, Optional[str]] = {}
    for gene_id in to_resolve:
        try:
            data = client.execute(TARGET_COORD_QUERY, {"ensemblId": gene_id})
            tgt = data.get("target")
            if isinstance(tgt, Mapping):
                cache[gene_id] = normalize_gene_symbol(tgt.get("approvedSymbol"))
            else:
                cache[gene_id] = None
        except Exception:
            cache[gene_id] = None

    resolved = 0
    for rec in records:
        if rec.get("gene_symbol"):
            continue
        gid = str(rec.get("gene_id") or "").strip()
        if not gid:
            continue
        sym = cache.get(gid)
        if sym:
            rec["gene_symbol"] = sym
            resolved += 1
    return resolved


def fetch_eva_scores_for_disease(
    client: GraphQLClient,
    disease_id: str,
    datasource_id: str,
    page_size: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    index = 0
    total_count: Optional[int] = None
    ds_id_norm = datasource_id.strip().lower()

    while True:
        data = client.execute(
            DISEASE_ASSOC_QUERY,
            {"diseaseId": disease_id, "pageIndex": index, "pageSize": page_size},
        )
        disease = data.get("disease")
        if disease is None:
            raise ValueError(f"Disease not found in Open Targets: {disease_id}")
        if not isinstance(disease, Mapping):
            raise ValueError(f"Unexpected disease payload for: {disease_id}")

        assoc = disease.get("associatedTargets")
        if not isinstance(assoc, Mapping):
            raise ValueError("Missing disease.associatedTargets in GraphQL response")

        total_count = to_int(assoc.get("count"))
        page_rows = assoc.get("rows", [])
        if not isinstance(page_rows, list):
            raise ValueError("Expected list in disease.associatedTargets.rows")

        for row in page_rows:
            if not isinstance(row, Mapping):
                continue
            target = row.get("target") if isinstance(row.get("target"), Mapping) else {}
            ds_scores = row.get("datasourceScores")
            if not isinstance(ds_scores, list):
                ds_scores = []

            eva_score: Optional[float] = None
            for ds in ds_scores:
                if not isinstance(ds, Mapping):
                    continue
                ds_id = str(ds.get("id", "")).strip().lower()
                if ds_id != ds_id_norm:
                    continue
                eva_score = to_float(ds.get("score"))
                if eva_score is not None:
                    break

            if eva_score is None:
                continue

            rows.append(
                {
                    "gene_id": normalize_gene_id(target.get("id")),
                    "gene_symbol": normalize_gene_symbol(target.get("approvedSymbol")),
                    "clinvar_eva_score": float(eva_score),
                    "association_score_global": to_float(row.get("score")),
                }
            )

        if not page_rows:
            break
        if total_count is not None and (index + 1) * page_size >= int(total_count):
            break
        index += 1

    eva_df = pd.DataFrame(rows)
    if eva_df.empty:
        raise ValueError(
            f"No datasource scores found for datasource_id='{datasource_id}' in disease '{disease_id}'."
        )

    eva_df = eva_df.sort_values("clinvar_eva_score", ascending=False, kind="stable")
    eva_df = eva_df.drop_duplicates(subset=["gene_id", "gene_symbol"], keep="first").reset_index(drop=True)

    stats = {
        "disease_id": disease_id,
        "datasource_id": datasource_id,
        "associated_targets_total": int(total_count or 0),
        "associated_targets_with_eva": int(len(eva_df)),
    }
    return eva_df, stats


def attach_clinvar_labels(
    feature_df: pd.DataFrame,
    eva_df: pd.DataFrame,
    eva_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = feature_df.copy()
    out["gene_symbol"] = out["gene_symbol"].map(normalize_gene_symbol)
    out["gene_id"] = out["gene_id"].map(normalize_gene_id)

    eva_by_symbol = (
        eva_df.dropna(subset=["gene_symbol"])
        .groupby("gene_symbol", dropna=True)["clinvar_eva_score"]
        .max()
        .to_dict()
    )
    eva_by_gene_id = (
        eva_df.dropna(subset=["gene_id"])
        .groupby("gene_id", dropna=True)["clinvar_eva_score"]
        .max()
        .to_dict()
    )

    scores: List[float] = []
    for row in out.itertuples(index=False):
        symbol = normalize_gene_symbol(getattr(row, "gene_symbol", None))
        gene_id = normalize_gene_id(getattr(row, "gene_id", None))
        score = np.nan
        if symbol and symbol in eva_by_symbol:
            score = float(eva_by_symbol[symbol])
        elif gene_id and gene_id in eva_by_gene_id:
            score = float(eva_by_gene_id[gene_id])
        scores.append(score)

    out["clinvar_eva_score"] = np.asarray(scores, dtype=np.float64)
    out["label_positive"] = (out["clinvar_eva_score"].fillna(0.0) >= float(eva_threshold)).astype(int)

    unique_genes = int(out["gene_symbol"].dropna().nunique())
    positive_rows = int(out["label_positive"].sum())
    positive_unique_genes = int(out.loc[out["label_positive"] == 1, "gene_symbol"].dropna().nunique())

    stats = {
        "n_rows": int(len(out)),
        "n_unique_genes": unique_genes,
        "positive_rows": positive_rows,
        "negative_rows": int(len(out) - positive_rows),
        "positive_unique_genes": positive_unique_genes,
        "negative_unique_genes": int(unique_genes - positive_unique_genes),
        "rows_with_missing_eva_score": int(out["clinvar_eva_score"].isna().sum()),
    }
    return out, stats


def parse_gtf_attributes(attr: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for chunk in attr.strip().split(";"):
        part = chunk.strip()
        if not part or " " not in part:
            continue
        key, value = part.split(" ", 1)
        out[key] = value.strip().strip('"')
    return out


def load_protein_coding_genes_from_gtf(gtf_path: Path) -> pd.DataFrame:
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF not found: {gtf_path}")

    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    rows: List[Dict[str, Any]] = []

    with opener(gtf_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != "gene":
                continue

            attrs = parse_gtf_attributes(parts[8])
            gene_type = attrs.get("gene_type") or attrs.get("gene_biotype")
            if gene_type != "protein_coding":
                continue

            gene_id_version = attrs.get("gene_id")
            gene_id = normalize_gene_id(gene_id_version)
            gene_symbol = normalize_gene_symbol(attrs.get("gene_name"))
            chrom = normalize_chrom(parts[0])
            start = to_int(parts[3])
            end = to_int(parts[4])
            strand_s = parts[6].strip()
            strand = -1 if strand_s == "-" else 1

            if gene_id is None or gene_symbol is None or chrom is None or start is None or end is None:
                continue
            if end < start:
                start, end = end, start
            tss = end if strand == -1 else start

            rows.append(
                {
                    "gene_id": gene_id,
                    "gene_id_version": gene_id_version,
                    "gene_symbol": gene_symbol,
                    "gene_type": gene_type,
                    "gene_chromosome": chrom,
                    "gene_start": start,
                    "gene_end": end,
                    "gene_strand": strand,
                    "gene_tss": tss,
                }
            )

    gene_df = pd.DataFrame(rows)
    if gene_df.empty:
        raise ValueError(f"No protein-coding genes parsed from GTF: {gtf_path}")

    gene_df = gene_df.drop_duplicates(subset=["gene_id"], keep="first").reset_index(drop=True)
    return gene_df


def build_candidate_gene_rows(
    gwas_cs_rows: Sequence[Dict[str, Any]],
    gene_df: pd.DataFrame,
    window_bp: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    by_chr = {
        chrom: sub.reset_index(drop=True)
        for chrom, sub in gene_df.groupby("gene_chromosome", sort=False)
    }

    out_rows: List[Dict[str, Any]] = []
    stats = {
        "n_gwas_cs_input": len(gwas_cs_rows),
        "n_gwas_cs_with_lead_variant": 0,
        "n_gwas_cs_without_lead_variant": 0,
    }

    for row in gwas_cs_rows:
        cs_id = row.get("study_locus_id")
        chrom = normalize_chrom(row.get("lead_variant_chromosome") or row.get("chromosome"))
        lead_pos = to_int(row.get("lead_variant_position") or row.get("position"))
        lead_id = row.get("lead_variant_id")

        if cs_id is None or chrom is None or lead_pos is None:
            stats["n_gwas_cs_without_lead_variant"] += 1
            continue
        stats["n_gwas_cs_with_lead_variant"] += 1

        genes_chr = by_chr.get(chrom)
        if genes_chr is None or genes_chr.empty:
            continue

        window_start = max(1, int(lead_pos) - int(window_bp))
        window_end = int(lead_pos) + int(window_bp)

        candidate = genes_chr[
            (genes_chr["gene_end"] >= window_start) & (genes_chr["gene_start"] <= window_end)
        ]

        for g in candidate.itertuples(index=False):
            d_gene_bp = interval_distance(int(lead_pos), int(g.gene_start), int(g.gene_end))
            d_tss_bp = abs(int(lead_pos) - int(g.gene_tss))
            out_rows.append(
                {
                    "gwas_study_locus_id": cs_id,
                    "gwas_study_id": row.get("study_id"),
                    "gwas_study_type": row.get("study_type"),
                    "gwas_cs_chromosome": normalize_chrom(row.get("chromosome")),
                    "gwas_cs_position": to_int(row.get("position")),
                    "gwas_cs_region": row.get("region"),
                    "gwas_cs_index": to_int(row.get("credible_set_index")),
                    "gwas_cs_p_value_mantissa": to_float(row.get("p_value_mantissa")),
                    "gwas_cs_p_value_exponent": to_int(row.get("p_value_exponent")),
                    "gwas_cs_beta": to_float(row.get("beta")),
                    "gwas_cs_z_score": to_float(row.get("z_score")),
                    "gwas_cs_confidence": row.get("confidence"),
                    "gwas_cs_locus_start": to_int(row.get("locus_start")),
                    "gwas_cs_locus_end": to_int(row.get("locus_end")),
                    "gwas_cs_quality_controls": row.get("quality_controls"),
                    "gwas_lead_variant_id": lead_id,
                    "gwas_lead_variant_rsids": row.get("lead_variant_rsids"),
                    "gwas_lead_variant_chromosome": chrom,
                    "gwas_lead_variant_position": lead_pos,
                    "candidate_window_start": window_start,
                    "candidate_window_end": window_end,
                    "gene_id": g.gene_id,
                    "gene_id_version": g.gene_id_version,
                    "gene_symbol": g.gene_symbol,
                    "gene_chromosome": g.gene_chromosome,
                    "gene_start": int(g.gene_start),
                    "gene_end": int(g.gene_end),
                    "gene_tss": int(g.gene_tss),
                    "gene_strand": int(g.gene_strand),
                    "candidate_gene_in_window": 1,
                    "variant_inside_gene": 1 if d_gene_bp == 0 else 0,
                    "dist_variant_to_gene_bp": d_gene_bp,
                    "dist_variant_to_gene_kb": d_gene_bp / 1000.0,
                    "dist_variant_to_tss_bp": d_tss_bp,
                    "dist_variant_to_tss_kb": d_tss_bp / 1000.0,
                    "dist_score_500kb_log": dist_score_500kb_log(d_gene_bp, window_bp=window_bp),
                }
            )

    return pd.DataFrame(out_rows), stats


def aggregate_colocalisation_by_gene(
    coloc_records: Sequence[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols_id = [
        "gwas_study_locus_id",
        "gene_id",
        "coloc_record_count",
        "qtl_study_locus_count",
        "qtl_study_count",
        "tissue_count",
        "qtl_study_types",
        "qtl_projects",
        "qtl_biosample_ids",
        "qtl_tissues",
        "qtl_conditions",
        "qtl_sub_study_descriptions",
        "qtl_traits",
        "any_trans_qtl",
        "n_trans_qtl",
        "colocalisation_h4_max",
        "colocalisation_h4_mean",
        "colocalisation_h3_mean",
        "colocalisation_clpp_max",
        "colocalisation_clpp_mean",
    ]
    cols_symbol = [c if c != "gene_id" else "gene_symbol" for c in cols_id]

    if not coloc_records:
        return pd.DataFrame(columns=cols_id), pd.DataFrame(columns=cols_symbol)

    df = pd.DataFrame(coloc_records)
    if df.empty:
        return pd.DataFrame(columns=cols_id), pd.DataFrame(columns=cols_symbol)

    df["gwas_study_locus_id"] = df["gwas_study_locus_id"].astype(str)
    df["gene_id"] = df["gene_id"].map(normalize_gene_id)
    df["gene_symbol"] = df["gene_symbol"].map(normalize_gene_symbol)

    agg_spec = {
        "coloc_record_count": ("qtl_study_locus_id", "size"),
        "qtl_study_locus_count": ("qtl_study_locus_id", "nunique"),
        "qtl_study_count": ("qtl_study_id", "nunique"),
        "tissue_count": ("qtl_biosample_name", nunique_non_null),
        "qtl_study_types": ("qtl_study_type", lambda x: unique_join(x.tolist())),
        "qtl_projects": ("qtl_project_id", lambda x: unique_join(x.tolist())),
        "qtl_biosample_ids": ("qtl_biosample_id", lambda x: unique_join(x.tolist())),
        "qtl_tissues": ("qtl_biosample_name", lambda x: unique_join(x.tolist())),
        "qtl_conditions": ("qtl_condition", lambda x: unique_join(x.tolist())),
        "qtl_sub_study_descriptions": (
            "qtl_sub_study_description",
            lambda x: unique_join(x.tolist(), max_items=40),
        ),
        "qtl_traits": ("qtl_trait_from_source", lambda x: unique_join(x.tolist(), max_items=40)),
        "any_trans_qtl": ("qtl_is_trans_qtl", any_true),
        "n_trans_qtl": ("qtl_is_trans_qtl", count_true),
        "colocalisation_h4_max": ("colocalisation_h4", "max"),
        "colocalisation_h4_mean": ("colocalisation_h4", "mean"),
        "colocalisation_h3_mean": ("colocalisation_h3", "mean"),
        "colocalisation_clpp_max": ("colocalisation_clpp", "max"),
        "colocalisation_clpp_mean": ("colocalisation_clpp", "mean"),
    }

    agg_id = (
        df.dropna(subset=["gene_id"])
        .groupby(["gwas_study_locus_id", "gene_id"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )

    agg_symbol = (
        df.dropna(subset=["gene_symbol"])
        .groupby(["gwas_study_locus_id", "gene_symbol"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )

    return agg_id, agg_symbol


def merge_candidates_with_coloc(
    candidate_df: pd.DataFrame,
    agg_by_gene_id: pd.DataFrame,
    agg_by_symbol: pd.DataFrame,
    h4_threshold: float,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    out = candidate_df.copy()
    out["gwas_study_locus_id"] = out["gwas_study_locus_id"].astype(str)
    out["gene_id"] = out["gene_id"].map(normalize_gene_id)
    out["gene_symbol"] = out["gene_symbol"].map(normalize_gene_symbol)

    out = out.merge(
        agg_by_gene_id,
        on=["gwas_study_locus_id", "gene_id"],
        how="left",
    )

    if not agg_by_symbol.empty:
        symbol_idx = agg_by_symbol.set_index(["gwas_study_locus_id", "gene_symbol"])
        qtl_col = "qtl_study_locus_count"
        missing_mask = out[qtl_col].isna() & out["gene_symbol"].notna()
        if missing_mask.any():
            keys = list(
                zip(
                    out.loc[missing_mask, "gwas_study_locus_id"],
                    out.loc[missing_mask, "gene_symbol"],
                )
            )
            fill_cols = [
                c
                for c in agg_by_symbol.columns
                if c not in {"gwas_study_locus_id", "gene_symbol"}
            ]
            for col in fill_cols:
                mapping = symbol_idx[col].to_dict()
                out.loc[missing_mask, col] = [mapping.get(k, np.nan) for k in keys]

    numeric_zero_cols = [
        "coloc_record_count",
        "qtl_study_locus_count",
        "qtl_study_count",
        "tissue_count",
        "n_trans_qtl",
        "colocalisation_h4_max",
        "colocalisation_h4_mean",
        "colocalisation_h3_mean",
        "colocalisation_clpp_max",
        "colocalisation_clpp_mean",
    ]
    for col in numeric_zero_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    text_cols = [
        "qtl_study_types",
        "qtl_projects",
        "qtl_biosample_ids",
        "qtl_tissues",
        "qtl_conditions",
        "qtl_sub_study_descriptions",
        "qtl_traits",
    ]
    for col in text_cols:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    if "any_trans_qtl" not in out.columns:
        out["any_trans_qtl"] = False
    out["any_trans_qtl"] = out["any_trans_qtl"].map(to_bool)
    out["any_trans_qtl"] = out["any_trans_qtl"].where(out["any_trans_qtl"].notna(), False).astype(bool)

    out["coloc_score_raw_h4_max"] = out["colocalisation_h4_max"]
    out["coloc_score"] = out["colocalisation_h4_max"].map(
        lambda v: 0.0
        if (v is None or float(v) < h4_threshold)
        else float(np.clip((float(v) - h4_threshold) / (1.0 - h4_threshold), 0.0, 1.0))
    )
    out["has_qtl_evidence"] = (out["qtl_study_locus_count"] > 0).astype(int)
    out["has_strong_coloc_h4"] = (out["colocalisation_h4_max"] >= float(h4_threshold)).astype(int)

    # Placeholders for data sources not yet provided locally.
    out["coding_score_sum_pip"] = 0.0
    out["coding_variant_count"] = 0
    out["expression_score"] = 0.0

    return out


def reduce_embedding_value(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1 and arr.size > 0:
            return arr
        if arr.ndim == 2 and arr.shape[0] > 0:
            return arr.mean(axis=0)
        return None

    if isinstance(value, Mapping):
        for key in ("embedding", "vector", "mean_embedding", "features", "values"):
            if key in value:
                return reduce_embedding_value(value[key])
        return None

    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if all(np.isscalar(v) for v in value):
            arr = np.asarray(value, dtype=np.float32)
            return arr if arr.ndim == 1 and arr.size > 0 else None

        vectors: List[np.ndarray] = []
        for item in value:
            vec = reduce_embedding_value(item)
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            return None

        length_counts = Counter(len(v) for v in vectors if v.ndim == 1 and len(v) > 0)
        if not length_counts:
            return None
        common_len = length_counts.most_common(1)[0][0]
        vectors = [v for v in vectors if v.ndim == 1 and len(v) == common_len]
        if not vectors:
            return None
        return np.vstack(vectors).mean(axis=0).astype(np.float32)

    return None


def load_gene_embeddings(embedding_path: Path) -> Tuple[Dict[str, np.ndarray], int, Dict[str, int]]:
    with open(embedding_path, "rb") as f:
        obj = pickle.load(f)

    gene_to_vectors: Dict[str, List[np.ndarray]] = defaultdict(list)
    stats = {
        "records_seen": 0,
        "vectors_extracted": 0,
        "genes_with_vectors_raw": 0,
        "vectors_discarded_dim_mismatch": 0,
        "vectors_discarded_projection_error": 0,
        "raw_embedding_dim": None,
        "embedding_view": "gene",
        "base_embedding_dim": 768,
    }

    def add_vector(gene_raw: object, value: Any) -> None:
        stats["records_seen"] += 1
        gene = normalize_gene_symbol(gene_raw)
        if gene is None:
            return
        vec = reduce_embedding_value(value)
        if vec is None or vec.ndim != 1 or vec.size == 0:
            return
        gene_to_vectors[gene].append(vec.astype(np.float32))
        stats["vectors_extracted"] += 1

    if isinstance(obj, Mapping):
        if "genes" in obj and "embeddings" in obj:
            genes = obj.get("genes")
            embs = obj.get("embeddings")
            if isinstance(genes, (list, tuple, np.ndarray)) and isinstance(embs, (list, tuple, np.ndarray)):
                n = min(len(genes), len(embs))
                for i in range(n):
                    add_vector(genes[i], embs[i])
        else:
            for gene_raw, value in obj.items():
                add_vector(gene_raw, value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, Mapping):
                gene_raw = item.get("gene") or item.get("symbol") or item.get("gene_symbol")
                value = item.get("embedding") if "embedding" in item else item
                add_vector(gene_raw, value)
    else:
        raise ValueError(f"Unsupported embedding pickle structure: {type(obj)}")

    if not gene_to_vectors:
        raise ValueError("No valid gene embeddings extracted from pickle.")

    dim_counts = Counter(len(vec) for vectors in gene_to_vectors.values() for vec in vectors)
    if not dim_counts:
        raise ValueError("No embedding vectors with valid dimensions were found.")
    raw_dim = dim_counts.most_common(1)[0][0]
    stats["raw_embedding_dim"] = raw_dim

    if raw_dim == 1536:
        project = lambda v: v[:768]
        expected_dim = 768
    elif raw_dim == 768:
        project = lambda v: v
        expected_dim = 768
    elif raw_dim % 2 == 0 and raw_dim > 2:
        half = raw_dim // 2
        project = lambda v: v[:half]
        expected_dim = half
    else:
        raise ValueError(f"Unsupported raw embedding dim={raw_dim} for gene-only projection.")

    final_embeddings: Dict[str, np.ndarray] = {}
    for gene, vectors in gene_to_vectors.items():
        same_dim = [vec for vec in vectors if len(vec) == raw_dim]
        stats["vectors_discarded_dim_mismatch"] += len(vectors) - len(same_dim)
        if not same_dim:
            continue

        projected: List[np.ndarray] = []
        for vec in same_dim:
            try:
                p = project(np.asarray(vec, dtype=np.float32))
                if p.ndim != 1 or len(p) != expected_dim:
                    raise ValueError("Projected embedding has invalid shape.")
                projected.append(p)
            except Exception:
                stats["vectors_discarded_projection_error"] += 1

        if not projected:
            continue
        final_embeddings[gene] = np.vstack(projected).mean(axis=0).astype(np.float32)

    stats["genes_with_vectors_raw"] = len(gene_to_vectors)
    if not final_embeddings:
        raise ValueError("No embeddings remained after gene-only projection.")

    embedding_dim = len(next(iter(final_embeddings.values())))
    return final_embeddings, embedding_dim, stats


def attach_embeddings(
    df: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    embedding_dim: int,
    fill_value: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if df.empty:
        return df, {"rows_missing_embedding": 0, "unique_genes_missing_embedding": 0}

    matrix = np.full((len(df), embedding_dim), fill_value, dtype=np.float32)
    has_embedding = np.zeros(len(df), dtype=np.int8)
    missing_genes: set[str] = set()

    for i, gene in enumerate(df["gene_symbol"].tolist()):
        gene_u = normalize_gene_symbol(gene)
        vec = embeddings.get(gene_u) if gene_u else None
        if vec is None or len(vec) != embedding_dim:
            if gene_u:
                missing_genes.add(gene_u)
            continue
        matrix[i, :] = vec
        has_embedding[i] = 1

    emb_cols = [f"gene_emb_{i:04d}" for i in range(embedding_dim)]
    emb_df = pd.DataFrame(matrix, columns=emb_cols, index=df.index)
    out = pd.concat([df, emb_df], axis=1)
    out["has_gene_embedding"] = has_embedding.astype(int)

    stats = {
        "rows_missing_embedding": int((has_embedding == 0).sum()),
        "unique_genes_missing_embedding": len(missing_genes),
    }
    return out, stats


def write_outputs(df: pd.DataFrame, base_path: Path, *, write_csv: bool, write_parquet: bool) -> List[Path]:
    written: List[Path] = []
    if write_csv:
        p = base_path.with_suffix(".csv")
        df.to_csv(p, index=False)
        written.append(p)
    if write_parquet:
        p = base_path.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        written.append(p)
    return written


def main(config: PipelineConfig = CONFIG) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)

    if not config.gtf_path.exists():
        raise FileNotFoundError(f"GTF not found: {config.gtf_path}")
    if not config.embedding_path.exists():
        raise FileNotFoundError(f"Embedding pickle not found: {config.embedding_path}")
    if config.page_size <= 0:
        raise ValueError("CONFIG.page_size must be > 0")
    if config.window_bp <= 0:
        raise ValueError("CONFIG.window_bp must be > 0")

    client = GraphQLClient(
        url=config.graphql_url,
        timeout_sec=float(config.timeout_sec),
        max_retries=int(config.max_retries),
    )

    print(f"[info] Study: {config.study_id}")
    print(f"[info] Loading protein-coding genes from: {config.gtf_path}")
    gene_df = load_protein_coding_genes_from_gtf(config.gtf_path)
    print(f"[info] Protein-coding genes loaded: {len(gene_df)}")

    study_meta, gwas_cs_rows = fetch_study_credible_sets(client, config.study_id, config.page_size)
    gwas_cs_ids = [str(r["study_locus_id"]) for r in gwas_cs_rows if r.get("study_locus_id")]
    print(f"[info] GWAS credible sets in study: {len(gwas_cs_ids)}")
    if not gwas_cs_ids:
        raise ValueError(f"No credible sets found for study {config.study_id}")

    coloc_records: List[Dict[str, Any]] = []
    for idx, gwas_cs_id in enumerate(gwas_cs_ids, start=1):
        if idx % 10 == 0 or idx == 1 or idx == len(gwas_cs_ids):
            print(f"[info] Fetching colocalisation: {idx}/{len(gwas_cs_ids)}")
        recs = fetch_colocalisations_for_gwas_cs(
            client=client,
            gwas_study_locus_id=gwas_cs_id,
            page_size=config.page_size,
            qtl_study_types=QTL_STUDY_TYPES,
        )
        coloc_records.extend(recs)

    print(f"[info] Colocalisation records: {len(coloc_records)}")
    resolved = resolve_missing_gene_symbols(coloc_records, client)
    if resolved > 0:
        print(f"[info] Gene symbols resolved via target query: {resolved}")

    candidate_df, candidate_stats = build_candidate_gene_rows(
        gwas_cs_rows=gwas_cs_rows,
        gene_df=gene_df,
        window_bp=int(config.window_bp),
    )
    print(f"[info] Candidate rows (gwas_cs, gene) in +/-{config.window_bp}bp window: {len(candidate_df)}")
    print(f"[info] Candidate stats: {candidate_stats}")

    agg_id, agg_symbol = aggregate_colocalisation_by_gene(coloc_records)
    final_df = merge_candidates_with_coloc(
        candidate_df=candidate_df,
        agg_by_gene_id=agg_id,
        agg_by_symbol=agg_symbol,
        h4_threshold=float(config.h4_threshold),
    )

    print(
        f"[info] Fetching ClinVar/EVA datasource scores: disease={config.disease_id} "
        f"datasource={config.datasource_id}"
    )
    eva_df, eva_fetch_stats = fetch_eva_scores_for_disease(
        client=client,
        disease_id=config.disease_id,
        datasource_id=config.datasource_id,
        page_size=config.page_size,
    )
    final_df, label_stats = attach_clinvar_labels(
        feature_df=final_df,
        eva_df=eva_df,
        eva_threshold=float(config.eva_positive_threshold),
    )
    print(f"[info] EVA fetch stats: {eva_fetch_stats}")
    print(
        "[info] Label stats: "
        f"positive_rows={label_stats['positive_rows']} "
        f"positive_unique_genes={label_stats['positive_unique_genes']} "
        f"rows_missing_eva={label_stats['rows_with_missing_eva_score']}"
    )

    embeddings, emb_dim, emb_stats = load_gene_embeddings(config.embedding_path)
    final_df, emb_attach_stats = attach_embeddings(
        df=final_df,
        embeddings=embeddings,
        embedding_dim=emb_dim,
        fill_value=float(config.embedding_fill_value),
    )

    final_df = final_df.sort_values(
        by=["gwas_study_locus_id", "dist_score_500kb_log", "coloc_score", "colocalisation_clpp_max"],
        ascending=[True, False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    raw_coloc_df = pd.DataFrame(coloc_records)

    raw_base = config.out_dir / f"{config.study_id}_raw_gwas_cs_gene_colocalisation"
    final_base = config.out_dir / f"{config.study_id}_cs_gene_candidate_feature_table"
    eva_base = config.out_dir / f"{config.study_id}_clinvar_eva_scores"

    written: List[Path] = []
    if config.write_raw_coloc:
        written.extend(
            write_outputs(
                raw_coloc_df,
                raw_base,
                write_csv=config.write_csv,
                write_parquet=config.write_parquet,
            )
        )
    written.extend(
        write_outputs(
            eva_df,
            eva_base,
            write_csv=config.write_csv,
            write_parquet=config.write_parquet,
        )
    )
    written.extend(
        write_outputs(
            final_df,
            final_base,
            write_csv=config.write_csv,
            write_parquet=config.write_parquet,
        )
    )

    print(f"[info] Embedding dimension: {emb_dim}")
    print(f"[info] Embedding load stats: {emb_stats}")
    print(f"[info] Missing embeddings (rows): {emb_attach_stats['rows_missing_embedding']}")
    print(f"[info] Missing embeddings (unique genes): {emb_attach_stats['unique_genes_missing_embedding']}")

    for path in written:
        print(f"[info] Wrote: {path}")

    print("[done] Pipeline completed.")
    print(
        "[done] Summary: "
        f"study={study_meta.get('study_id')} "
        f"credible_sets={len(gwas_cs_ids)} "
        f"candidate_rows={len(candidate_df)} "
        f"final_rows={len(final_df)} "
        f"unique_genes={final_df['gene_symbol'].nunique(dropna=True) if not final_df.empty else 0} "
        f"rows_with_qtl={(final_df['has_qtl_evidence'] == 1).sum() if 'has_qtl_evidence' in final_df else 0} "
        f"positive_genes={final_df.loc[final_df['label_positive'] == 1, 'gene_symbol'].nunique() if 'label_positive' in final_df else 0}"
    )


if __name__ == "__main__":
    main(CONFIG)

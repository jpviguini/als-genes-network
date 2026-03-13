#!/usr/bin/env python3
"""
Build ALS locus-gene tables from GWAS summary statistics + gene annotation.

This script implements a simple, LD-free locus definition strategy:
1) Filter SNPs by p-value.
2) Sort by p-value ascending.
3) Pick the best SNP as lead.
4) Define locus interval as lead +/- window.
5) Remove SNPs falling in this interval and repeat.

Outputs:
- ALS loci table.
- (locus, gene) candidate table.
- (locus, gene) feature table with distance-based features.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_GWAS = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/data/ALS_GWAS_summary_statistics/"
    "GCST90027164_buildGRCh37.tsv.gz"
)
DEFAULT_GENE_ANNOTATION = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/data/ALS_GWAS_summary_statistics/"
    "gencode.v37lift37.annotation.gtf"
)
DEFAULT_OUT_DIR = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_locus_gene_tables"
)


@dataclass(frozen=True)
class SnpRecord:
    chrom: str
    pos: int
    p_value: float
    snp_id: str
    beta: Optional[float]
    standard_error: Optional[float]


@dataclass(frozen=True)
class Locus:
    locus_id: str
    chrom: str
    lead_snp: str
    lead_pos: int
    lead_p_value: float
    lead_beta: Optional[float]
    lead_standard_error: Optional[float]
    locus_start: int
    locus_end: int


@dataclass(frozen=True)
class GeneCoord:
    gene: str
    chrom: str
    start: int
    end: int
    strand: str
    tss: int


@dataclass
class ChromGeneIndex:
    genes: List[GeneCoord]
    starts: List[int]
    max_gene_len: int


@dataclass(frozen=True)
class GwasQcStats:
    rows_scanned: int
    rows_with_required_fields: int
    rows_missing_required_fields: int
    rows_missing_required_values: int
    chromosome_counts: Dict[str, int]
    min_p_value_seen: Optional[float]
    max_p_value_seen: Optional[float]


def configure_csv_field_limit() -> int:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit = limit // 10


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() == "NA":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() == "NA":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def normalize_chrom(chrom: str) -> str:
    c = str(chrom).strip()
    if not c:
        return c
    if c.lower().startswith("chr"):
        c = c[3:]
    c_u = c.upper()
    aliases = {"23": "X", "24": "Y", "25": "MT", "M": "MT"}
    return aliases.get(c_u, c_u)


def parse_gtf_attributes(raw: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for chunk in raw.strip().split(";"):
        chunk = chunk.strip()
        if not chunk or " " not in chunk:
            continue
        key, value = chunk.split(" ", 1)
        attrs[key.strip()] = value.strip().strip('"')
    return attrs


def pick_column(fieldnames: Sequence[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {f.lower(): f for f in fieldnames}
    for candidate in candidates:
        col = lowered.get(candidate.lower())
        if col is not None:
            return col
    return None


def build_variant_id(
    row: Dict[str, str],
    chrom: str,
    pos: int,
    rsid_col: Optional[str],
    variant_id_col: Optional[str],
    effect_allele_col: Optional[str],
    other_allele_col: Optional[str],
) -> str:
    if rsid_col:
        rsid = str(row.get(rsid_col, "")).strip()
        if rsid and rsid.upper() != "NA":
            return rsid

    if variant_id_col:
        variant_id = str(row.get(variant_id_col, "")).strip()
        if variant_id and variant_id.upper() != "NA":
            return variant_id

    ea = str(row.get(effect_allele_col, "")).strip() if effect_allele_col else ""
    oa = str(row.get(other_allele_col, "")).strip() if other_allele_col else ""
    if ea and oa and ea.upper() != "NA" and oa.upper() != "NA":
        return f"{chrom}:{pos}:{ea}>{oa}"
    return f"{chrom}:{pos}"


def chrom_sort_key(chrom: str) -> Tuple[int, str]:
    chrom_u = chrom.upper()
    mapping = {"X": 23, "Y": 24, "MT": 25}
    if chrom_u in mapping:
        return (mapping[chrom_u], chrom_u)
    try:
        return (int(chrom_u), chrom_u)
    except ValueError:
        return (100, chrom_u)


def interval_distance(pos: int, start: int, end: int) -> int:
    if pos < start:
        return start - pos
    if pos > end:
        return pos - end
    return 0


def validate_arg_path(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{name} is not a file: {path}")


def load_significant_snps(
    gwas_path: Path,
    delimiter: str,
    chrom_col: str,
    pos_col: str,
    pval_col: str,
    rsid_col: Optional[str],
    variant_id_col: Optional[str],
    effect_allele_col: Optional[str],
    other_allele_col: Optional[str],
    beta_col: Optional[str],
    standard_error_col: Optional[str],
    p_threshold: Optional[float],
    max_significant_snps: Optional[int],
) -> Tuple[List[SnpRecord], GwasQcStats]:
    scanned = 0
    rows_with_required_fields = 0
    rows_missing_required_fields = 0
    rows_missing_required_values = 0
    chrom_counts: Counter[str] = Counter()
    min_p_value_seen: Optional[float] = None
    max_p_value_seen: Optional[float] = None
    kept: List[SnpRecord] = []

    with open_text(gwas_path) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"GWAS file has no header: {gwas_path}")
        required = [chrom_col, pos_col, pval_col]
        missing_required = [c for c in required if c not in reader.fieldnames]
        if missing_required:
            raise ValueError(
                f"Missing required GWAS columns in {gwas_path}: {', '.join(missing_required)}"
            )

        for row in reader:
            scanned += 1
            raw_chrom = row.get(chrom_col)
            raw_pos = row.get(pos_col)
            raw_pval = row.get(pval_col)

            if raw_chrom is None or raw_pos is None or raw_pval is None:
                rows_missing_required_fields += 1
                continue

            rows_with_required_fields += 1
            chrom = normalize_chrom(str(raw_chrom).strip())
            pos = to_int(raw_pos)
            p_value = to_float(raw_pval)
            if not chrom or pos is None or p_value is None or not math.isfinite(p_value):
                rows_missing_required_values += 1
                continue

            chrom_counts[chrom] += 1
            min_p_value_seen = p_value if min_p_value_seen is None else min(min_p_value_seen, p_value)
            max_p_value_seen = p_value if max_p_value_seen is None else max(max_p_value_seen, p_value)

            if p_threshold is not None and p_value > p_threshold:
                continue

            snp_id = build_variant_id(
                row=row,
                chrom=chrom,
                pos=pos,
                rsid_col=rsid_col,
                variant_id_col=variant_id_col,
                effect_allele_col=effect_allele_col,
                other_allele_col=other_allele_col,
            )
            kept.append(
                SnpRecord(
                    chrom=chrom,
                    pos=pos,
                    p_value=p_value,
                    snp_id=snp_id,
                    beta=to_float(row.get(beta_col)) if beta_col else None,
                    standard_error=to_float(row.get(standard_error_col))
                    if standard_error_col
                    else None,
                )
            )

            if max_significant_snps is not None and len(kept) >= max_significant_snps:
                break

    kept.sort(key=lambda x: (x.p_value, chrom_sort_key(x.chrom), x.pos, x.snp_id))
    qc = GwasQcStats(
        rows_scanned=scanned,
        rows_with_required_fields=rows_with_required_fields,
        rows_missing_required_fields=rows_missing_required_fields,
        rows_missing_required_values=rows_missing_required_values,
        chromosome_counts=dict(chrom_counts),
        min_p_value_seen=min_p_value_seen,
        max_p_value_seen=max_p_value_seen,
    )
    return kept, qc


def select_lead_loci(significant_snps: Sequence[SnpRecord], lead_window_bp: int) -> List[Locus]:
    covered_by_chrom: DefaultDict[str, List[Tuple[int, int]]] = defaultdict(list)
    selected: List[Locus] = []

    for snp in significant_snps:
        intervals = covered_by_chrom[snp.chrom]
        if any(start <= snp.pos <= end for start, end in intervals):
            continue

        locus_start = max(1, snp.pos - lead_window_bp)
        locus_end = snp.pos + lead_window_bp
        intervals.append((locus_start, locus_end))
        selected.append(
            Locus(
                locus_id="",  # filled later
                chrom=snp.chrom,
                lead_snp=snp.snp_id,
                lead_pos=snp.pos,
                lead_p_value=snp.p_value,
                lead_beta=snp.beta,
                lead_standard_error=snp.standard_error,
                locus_start=locus_start,
                locus_end=locus_end,
            )
        )

    loci: List[Locus] = []
    for i, locus in enumerate(selected, start=1):
        loci.append(
            Locus(
                locus_id=f"L{i}",
                chrom=locus.chrom,
                lead_snp=locus.lead_snp,
                lead_pos=locus.lead_pos,
                lead_p_value=locus.lead_p_value,
                lead_beta=locus.lead_beta,
                lead_standard_error=locus.lead_standard_error,
                locus_start=locus.locus_start,
                locus_end=locus.locus_end,
            )
        )
    return loci


def assign_snp_to_locus(
    chrom: str,
    pos: int,
    loci_by_chrom: Dict[str, List[Locus]],
) -> Optional[Locus]:
    loci = loci_by_chrom.get(chrom)
    if not loci:
        return None
    candidates = [l for l in loci if l.locus_start <= pos <= l.locus_end]
    if not candidates:
        return None
    return min(candidates, key=lambda l: (abs(pos - l.lead_pos), l.lead_p_value, l.lead_pos))


def collect_locus_snp_positions_from_significant(
    loci: Sequence[Locus],
    significant_snps: Sequence[SnpRecord],
    p_threshold: Optional[float],
) -> Dict[str, Dict[str, object]]:
    loci_by_chrom: Dict[str, List[Locus]] = defaultdict(list)
    for locus in loci:
        loci_by_chrom[locus.chrom].append(locus)

    stats: Dict[str, Dict[str, object]] = {
        l.locus_id: {"positions": [], "n_locus_snps": 0, "n_locus_snps_significant": 0}
        for l in loci
    }

    for snp in significant_snps:
        locus = assign_snp_to_locus(snp.chrom, snp.pos, loci_by_chrom)
        if locus is None:
            continue
        entry = stats[locus.locus_id]
        positions = entry["positions"]
        assert isinstance(positions, list)
        positions.append(snp.pos)
        entry["n_locus_snps"] = int(entry["n_locus_snps"]) + 1
        if p_threshold is None or snp.p_value <= p_threshold:
            entry["n_locus_snps_significant"] = int(entry["n_locus_snps_significant"]) + 1

    for locus in loci:
        entry = stats[locus.locus_id]
        positions = entry["positions"]
        assert isinstance(positions, list)
        if locus.lead_pos not in positions:
            positions.append(locus.lead_pos)
            entry["n_locus_snps"] = int(entry["n_locus_snps"]) + 1
            if p_threshold is None or locus.lead_p_value <= p_threshold:
                entry["n_locus_snps_significant"] = int(entry["n_locus_snps_significant"]) + 1
        positions.sort()

    return stats


def collect_locus_snp_positions_from_all(
    loci: Sequence[Locus],
    gwas_path: Path,
    delimiter: str,
    chrom_col: str,
    pos_col: str,
    pval_col: str,
    p_threshold: Optional[float],
) -> Dict[str, Dict[str, object]]:
    loci_by_chrom: Dict[str, List[Locus]] = defaultdict(list)
    for locus in loci:
        loci_by_chrom[locus.chrom].append(locus)

    stats: Dict[str, Dict[str, object]] = {
        l.locus_id: {"positions": [], "n_locus_snps": 0, "n_locus_snps_significant": 0}
        for l in loci
    }

    with open_text(gwas_path) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"GWAS file has no header: {gwas_path}")

        required = [chrom_col, pos_col]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required GWAS columns in {gwas_path}: {', '.join(missing)}")

        for row in reader:
            chrom = normalize_chrom(str(row.get(chrom_col, "")).strip())
            pos = to_int(row.get(pos_col))
            if not chrom or pos is None:
                continue

            locus = assign_snp_to_locus(chrom, pos, loci_by_chrom)
            if locus is None:
                continue

            entry = stats[locus.locus_id]
            positions = entry["positions"]
            assert isinstance(positions, list)
            positions.append(pos)
            entry["n_locus_snps"] = int(entry["n_locus_snps"]) + 1

            p_value = to_float(row.get(pval_col)) if pval_col in row else None
            if p_value is not None and math.isfinite(p_value):
                if p_threshold is None or p_value <= p_threshold:
                    entry["n_locus_snps_significant"] = int(entry["n_locus_snps_significant"]) + 1

    for locus in loci:
        entry = stats[locus.locus_id]
        positions = entry["positions"]
        assert isinstance(positions, list)
        if locus.lead_pos not in positions:
            positions.append(locus.lead_pos)
            entry["n_locus_snps"] = int(entry["n_locus_snps"]) + 1
            if p_threshold is None or locus.lead_p_value <= p_threshold:
                entry["n_locus_snps_significant"] = int(entry["n_locus_snps_significant"]) + 1
        positions.sort()

    return stats


def merge_gene_records(records: Sequence[Tuple[str, str, int, int, str]]) -> List[GeneCoord]:
    grouped: DefaultDict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)
    for gene, chrom, start, end, strand in records:
        s = min(start, end)
        e = max(start, end)
        grouped[(gene, chrom, strand)].append((s, e))

    merged: List[GeneCoord] = []
    for (gene, chrom, strand), intervals in grouped.items():
        intervals.sort(key=lambda t: (t[0], t[1]))
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                tss = cur_e if strand == "-" else cur_s
                merged.append(
                    GeneCoord(
                        gene=gene,
                        chrom=chrom,
                        start=cur_s,
                        end=cur_e,
                        strand=strand,
                        tss=tss,
                    )
                )
                cur_s, cur_e = s, e
        tss = cur_e if strand == "-" else cur_s
        merged.append(
            GeneCoord(
                gene=gene,
                chrom=chrom,
                start=cur_s,
                end=cur_e,
                strand=strand,
                tss=tss,
            )
        )
    return merged


def build_gene_index(records: Sequence[GeneCoord]) -> Dict[str, ChromGeneIndex]:
    by_chrom: DefaultDict[str, List[GeneCoord]] = defaultdict(list)
    for rec in records:
        by_chrom[rec.chrom].append(rec)

    index: Dict[str, ChromGeneIndex] = {}
    for chrom, genes in by_chrom.items():
        genes_sorted = sorted(genes, key=lambda g: (g.start, g.end, g.gene))
        starts = [g.start for g in genes_sorted]
        max_len = max((g.end - g.start + 1 for g in genes_sorted), default=0)
        index[chrom] = ChromGeneIndex(genes=genes_sorted, starts=starts, max_gene_len=max_len)
    return index


def load_genes_from_gtf(gtf_path: Path) -> Dict[str, ChromGeneIndex]:
    raw: List[Tuple[str, str, int, int, str]] = []
    with open_text(gtf_path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _src, feature, start, end, _score, strand, _frame, attrs_raw = parts
            if feature != "gene":
                continue

            chrom_n = normalize_chrom(chrom)
            start_i = to_int(start)
            end_i = to_int(end)
            if not chrom_n or start_i is None or end_i is None:
                continue

            attrs = parse_gtf_attributes(attrs_raw)
            gene_name = (
                attrs.get("gene_name")
                or attrs.get("gene_symbol")
                or attrs.get("Name")
                or attrs.get("gene_id")
            )
            if not gene_name:
                continue
            gene = gene_name.strip().upper()
            if not gene:
                continue
            raw.append((gene, chrom_n, start_i, end_i, strand if strand else "."))

    merged = merge_gene_records(raw)
    return build_gene_index(merged)


def collect_candidate_genes_for_locus(
    chrom_index: ChromGeneIndex,
    locus_start: int,
    locus_end: int,
    gene_window_bp: int,
) -> List[GeneCoord]:
    expanded_start = max(1, locus_start - gene_window_bp)
    expanded_end = locus_end + gene_window_bp

    left_limit = expanded_start - chrom_index.max_gene_len
    lo = bisect_left(chrom_index.starts, left_limit)
    hi = bisect_right(chrom_index.starts, expanded_end)

    candidates: List[GeneCoord] = []
    for gene in chrom_index.genes[lo:hi]:
        if gene.end < expanded_start:
            continue
        if gene.start > expanded_end:
            continue
        candidates.append(gene)
    return candidates


def deduplicate_genes_by_symbol(
    candidate_genes: Sequence[GeneCoord],
    lead_pos: int,
) -> List[GeneCoord]:
    best_by_gene: Dict[str, Tuple[GeneCoord, int]] = {}
    for gene in candidate_genes:
        d = interval_distance(lead_pos, gene.start, gene.end)
        prev = best_by_gene.get(gene.gene)
        if prev is None:
            best_by_gene[gene.gene] = (gene, d)
            continue
        prev_gene, prev_d = prev
        if d < prev_d:
            best_by_gene[gene.gene] = (gene, d)
        elif d == prev_d and (gene.start, gene.end, gene.strand) < (
            prev_gene.start,
            prev_gene.end,
            prev_gene.strand,
        ):
            best_by_gene[gene.gene] = (gene, d)

    deduped = [g for g, _d in best_by_gene.values()]
    deduped.sort(key=lambda g: (g.start, g.end, g.gene))
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ALS locus-gene tables with distance-based features."
    )
    parser.add_argument("--gwas-tsv", type=Path, default=DEFAULT_GWAS)
    parser.add_argument("--gene-annotation-gtf", type=Path, default=DEFAULT_GENE_ANNOTATION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    parser.add_argument("--gwas-delimiter", type=str, default="\t")
    parser.add_argument("--gwas-chrom-col", type=str, default="chromosome")
    parser.add_argument("--gwas-pos-col", type=str, default="base_pair_location")
    parser.add_argument("--gwas-pval-col", type=str, default="p_value")
    parser.add_argument("--gwas-rsid-col", type=str, default="rsid")
    parser.add_argument("--gwas-variant-id-col", type=str, default="variant_id")
    parser.add_argument("--gwas-effect-allele-col", type=str, default="effect_allele")
    parser.add_argument("--gwas-other-allele-col", type=str, default="other_allele")
    parser.add_argument("--gwas-beta-col", type=str, default="beta")
    parser.add_argument("--gwas-standard-error-col", type=str, default="standard_error")

    parser.add_argument(
        "--p-threshold",
        type=float,
        default=5e-8,
        help="P-value threshold used to define significant SNPs and lead loci.",
    )
    parser.add_argument(
        "--lead-window-bp",
        type=int,
        default=500_000,
        help="Locus interval around each lead SNP: lead +/- window.",
    )
    parser.add_argument(
        "--gene-window-bp",
        type=int,
        default=0,
        help="Extra bp added around locus when selecting candidate genes.",
    )
    parser.add_argument(
        "--near-gene-window-bp",
        type=int,
        default=50_000,
        help="Window for counting SNPs near each gene in feature table.",
    )
    parser.add_argument(
        "--locus-snps-source",
        type=str,
        choices=["significant", "all"],
        default="significant",
        help="Which SNP set defines locus SNP statistics and min-distance feature.",
    )
    parser.add_argument(
        "--max-significant-snps",
        type=int,
        default=None,
        help="Optional cap for debugging faster runs.",
    )
    parser.add_argument(
        "--max-loci",
        type=int,
        default=None,
        help="Optional cap on number of loci after lead selection.",
    )
    parser.add_argument(
        "--min-chromosomes-required",
        type=int,
        default=10,
        help=(
            "Minimum number of chromosomes expected in GWAS parsed rows. "
            "Set 0 to disable. Useful to catch truncated files."
        ),
    )
    parser.add_argument(
        "--allow-low-chromosome-coverage",
        action="store_true",
        help="Do not fail when chromosome coverage is below --min-chromosomes-required.",
    )
    parser.add_argument(
        "--allow-malformed-rows",
        action="store_true",
        help="Do not fail when rows are missing required fields (e.g., truncated tail line).",
    )
    return parser.parse_args()


def none_if_missing(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = value.strip()
    return s if s else None


def main() -> None:
    configure_csv_field_limit()
    args = parse_args()

    if args.lead_window_bp < 0:
        raise ValueError("--lead-window-bp must be >= 0")
    if args.gene_window_bp < 0:
        raise ValueError("--gene-window-bp must be >= 0")
    if args.near_gene_window_bp < 0:
        raise ValueError("--near-gene-window-bp must be >= 0")
    if args.p_threshold is not None and args.p_threshold <= 0:
        raise ValueError("--p-threshold must be > 0")

    validate_arg_path(args.gwas_tsv, "--gwas-tsv")
    validate_arg_path(args.gene_annotation_gtf, "--gene-annotation-gtf")

    rsid_col = none_if_missing(args.gwas_rsid_col)
    variant_id_col = none_if_missing(args.gwas_variant_id_col)
    effect_allele_col = none_if_missing(args.gwas_effect_allele_col)
    other_allele_col = none_if_missing(args.gwas_other_allele_col)
    beta_col = none_if_missing(args.gwas_beta_col)
    standard_error_col = none_if_missing(args.gwas_standard_error_col)

    print(f"[info] Loading significant SNPs from: {args.gwas_tsv}")
    significant_snps, qc = load_significant_snps(
        gwas_path=args.gwas_tsv,
        delimiter=args.gwas_delimiter,
        chrom_col=args.gwas_chrom_col,
        pos_col=args.gwas_pos_col,
        pval_col=args.gwas_pval_col,
        rsid_col=rsid_col,
        variant_id_col=variant_id_col,
        effect_allele_col=effect_allele_col,
        other_allele_col=other_allele_col,
        beta_col=beta_col,
        standard_error_col=standard_error_col,
        p_threshold=args.p_threshold,
        max_significant_snps=args.max_significant_snps,
    )
    chroms_present = sorted(qc.chromosome_counts.keys(), key=chrom_sort_key)
    print(f"[info] GWAS rows scanned: {qc.rows_scanned}")
    print(f"[info] GWAS rows with required fields: {qc.rows_with_required_fields}")
    print(f"[info] GWAS rows missing required fields: {qc.rows_missing_required_fields}")
    print(f"[info] GWAS rows with missing/invalid required values: {qc.rows_missing_required_values}")
    print(f"[info] Chromosomes with parsed rows: {len(chroms_present)} -> {', '.join(chroms_present)}")
    if qc.min_p_value_seen is not None and qc.max_p_value_seen is not None:
        print(f"[info] P-value range observed: min={qc.min_p_value_seen:.3e}, max={qc.max_p_value_seen:.3e}")

    if qc.rows_missing_required_fields > 0 and not args.allow_malformed_rows:
        raise RuntimeError(
            "GWAS has rows missing required fields. This usually indicates a malformed/truncated file. "
            f"Rows missing required fields: {qc.rows_missing_required_fields}. "
            "If this is expected, rerun with --allow-malformed-rows."
        )
    if (
        args.min_chromosomes_required > 0
        and len(chroms_present) < args.min_chromosomes_required
        and not args.allow_low_chromosome_coverage
    ):
        raise RuntimeError(
            "GWAS chromosome coverage is unexpectedly low. "
            f"Found {len(chroms_present)} chromosomes ({', '.join(chroms_present)}), "
            f"expected at least {args.min_chromosomes_required}. "
            "This often means the file is partial/truncated. "
            "If this is intentional (subset GWAS), rerun with --allow-low-chromosome-coverage."
        )

    print(f"[info] Significant SNPs retained: {len(significant_snps)}")
    if not significant_snps:
        raise RuntimeError("No significant SNPs found with current filters.")

    loci = select_lead_loci(significant_snps=significant_snps, lead_window_bp=args.lead_window_bp)
    if args.max_loci is not None:
        loci = loci[: args.max_loci]
    print(f"[info] Lead loci defined: {len(loci)}")
    if not loci:
        raise RuntimeError("No loci were defined.")

    print(f"[info] Loading genes from: {args.gene_annotation_gtf}")
    gene_index = load_genes_from_gtf(args.gene_annotation_gtf)
    n_genes = sum(len(v.genes) for v in gene_index.values())
    print(f"[info] Genes indexed: {n_genes} across {len(gene_index)} chromosomes.")

    print(f"[info] Collecting locus SNPs from source: {args.locus_snps_source}")
    if args.locus_snps_source == "significant":
        locus_snp_stats = collect_locus_snp_positions_from_significant(
            loci=loci,
            significant_snps=significant_snps,
            p_threshold=args.p_threshold,
        )
    else:
        locus_snp_stats = collect_locus_snp_positions_from_all(
            loci=loci,
            gwas_path=args.gwas_tsv,
            delimiter=args.gwas_delimiter,
            chrom_col=args.gwas_chrom_col,
            pos_col=args.gwas_pos_col,
            pval_col=args.gwas_pval_col,
            p_threshold=args.p_threshold,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    loci_tsv = args.out_dir / "als_loci.tsv"
    pairs_tsv = args.out_dir / "als_locus_gene_pairs.tsv"
    features_tsv = args.out_dir / "als_locus_gene_features.tsv"

    loci_cols = [
        "locus_id",
        "chromosome",
        "lead_snp",
        "lead_pos",
        "locus_start",
        "locus_end",
        "lead_p_value",
        "lead_beta",
        "lead_standard_error",
        "n_locus_snps",
        "n_locus_snps_significant",
    ]
    pair_cols = [
        "locus_id",
        "chromosome",
        "locus_start",
        "locus_end",
        "lead_snp",
        "lead_pos",
        "lead_p_value",
        "gene",
        "gene_start",
        "gene_end",
        "gene_strand",
        "gene_tss",
    ]
    feature_cols = pair_cols + [
        "dist_to_lead_bp",
        "dist_to_lead_kb",
        "lead_inside_gene",
        "dist_to_tss_bp",
        "dist_to_tss_kb",
        "min_snp_gene_distance_bp",
        "min_snp_gene_distance_kb",
        "n_locus_snps_near_gene",
        "near_gene_window_bp",
        "n_locus_snps_total",
        "n_locus_snps_significant",
    ]

    n_pair_rows = 0
    loci_without_genes = 0

    with open(loci_tsv, "w", encoding="utf-8", newline="") as f_loci, open(
        pairs_tsv, "w", encoding="utf-8", newline=""
    ) as f_pairs, open(features_tsv, "w", encoding="utf-8", newline="") as f_features:
        loci_writer = csv.DictWriter(f_loci, fieldnames=loci_cols, delimiter="\t")
        pair_writer = csv.DictWriter(f_pairs, fieldnames=pair_cols, delimiter="\t")
        feature_writer = csv.DictWriter(f_features, fieldnames=feature_cols, delimiter="\t")
        loci_writer.writeheader()
        pair_writer.writeheader()
        feature_writer.writeheader()

        for locus in loci:
            stat = locus_snp_stats.get(
                locus.locus_id,
                {"positions": [locus.lead_pos], "n_locus_snps": 1, "n_locus_snps_significant": 1},
            )
            positions_obj = stat.get("positions", [locus.lead_pos])
            positions = positions_obj if isinstance(positions_obj, list) else [locus.lead_pos]
            n_locus_snps = int(stat.get("n_locus_snps", len(positions)))
            n_locus_snps_sig = int(stat.get("n_locus_snps_significant", 0))

            loci_writer.writerow(
                {
                    "locus_id": locus.locus_id,
                    "chromosome": locus.chrom,
                    "lead_snp": locus.lead_snp,
                    "lead_pos": locus.lead_pos,
                    "locus_start": locus.locus_start,
                    "locus_end": locus.locus_end,
                    "lead_p_value": locus.lead_p_value,
                    "lead_beta": "" if locus.lead_beta is None else locus.lead_beta,
                    "lead_standard_error": ""
                    if locus.lead_standard_error is None
                    else locus.lead_standard_error,
                    "n_locus_snps": n_locus_snps,
                    "n_locus_snps_significant": n_locus_snps_sig,
                }
            )

            chrom_genes = gene_index.get(locus.chrom)
            if chrom_genes is None:
                loci_without_genes += 1
                continue

            candidate_genes = collect_candidate_genes_for_locus(
                chrom_index=chrom_genes,
                locus_start=locus.locus_start,
                locus_end=locus.locus_end,
                gene_window_bp=args.gene_window_bp,
            )
            candidate_genes = deduplicate_genes_by_symbol(candidate_genes, lead_pos=locus.lead_pos)
            if not candidate_genes:
                loci_without_genes += 1
                continue

            for gene in candidate_genes:
                dist_to_lead = interval_distance(locus.lead_pos, gene.start, gene.end)
                dist_to_tss = abs(locus.lead_pos - gene.tss)
                min_snp_gene_distance = min(
                    interval_distance(pos, gene.start, gene.end) for pos in positions
                )
                n_near_gene = sum(
                    1
                    for pos in positions
                    if interval_distance(pos, gene.start, gene.end) <= args.near_gene_window_bp
                )

                pair_row = {
                    "locus_id": locus.locus_id,
                    "chromosome": locus.chrom,
                    "locus_start": locus.locus_start,
                    "locus_end": locus.locus_end,
                    "lead_snp": locus.lead_snp,
                    "lead_pos": locus.lead_pos,
                    "lead_p_value": locus.lead_p_value,
                    "gene": gene.gene,
                    "gene_start": gene.start,
                    "gene_end": gene.end,
                    "gene_strand": gene.strand,
                    "gene_tss": gene.tss,
                }
                pair_writer.writerow(pair_row)
                feature_writer.writerow(
                    {
                        **pair_row,
                        "dist_to_lead_bp": dist_to_lead,
                        "dist_to_lead_kb": dist_to_lead / 1000.0,
                        "lead_inside_gene": int(dist_to_lead == 0),
                        "dist_to_tss_bp": dist_to_tss,
                        "dist_to_tss_kb": dist_to_tss / 1000.0,
                        "min_snp_gene_distance_bp": min_snp_gene_distance,
                        "min_snp_gene_distance_kb": min_snp_gene_distance / 1000.0,
                        "n_locus_snps_near_gene": n_near_gene,
                        "near_gene_window_bp": args.near_gene_window_bp,
                        "n_locus_snps_total": n_locus_snps,
                        "n_locus_snps_significant": n_locus_snps_sig,
                    }
                )
                n_pair_rows += 1

    print(f"[info] Loci table saved: {loci_tsv}")
    print(f"[info] Locus-gene pairs table saved: {pairs_tsv}")
    print(f"[info] Locus-gene features table saved: {features_tsv}")
    print(f"[info] Pair rows written: {n_pair_rows}")
    print(f"[info] Loci without candidate genes: {loci_without_genes}")


if __name__ == "__main__":
    main()

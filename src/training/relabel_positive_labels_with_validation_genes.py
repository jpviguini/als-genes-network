#!/usr/bin/env python3
"""Relabel `label_positive` using config.VALIDATION_GENES for one-off runs.

This utility is intentionally standalone and does not change default labeling
behavior in the main feature-table pipeline (ClinVar/EVA threshold).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import pandas as pd


def _import_validation_genes() -> Set[str]:
    import sys

    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from config import VALIDATION_GENES  # type: ignore

    out: Set[str] = set()
    for g in VALIDATION_GENES:
        gg = normalize_gene_symbol(g)
        if gg:
            out.add(gg)
    return out


def normalize_gene_symbol(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().upper()
    return s if s else None


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relabel label_positive using config.VALIDATION_GENES."
    )
    parser.add_argument(
        "--input-table",
        type=Path,
        required=True,
        help="Input candidate table (.csv or .parquet).",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        required=True,
        help="Output table path (.csv or .parquet).",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON report path with counts and positive genes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_genes = _import_validation_genes()
    df = load_table(args.input_table).copy()

    if "gene_symbol" not in df.columns:
        raise ValueError("Expected column 'gene_symbol' in input table.")

    gene_norm = df["gene_symbol"].map(normalize_gene_symbol)
    new_label = gene_norm.map(lambda g: 1 if (g is not None and g in val_genes) else 0).astype(int)
    df["label_positive"] = new_label

    save_table(df, args.output_table)

    present_val_genes = sorted(
        {
            g
            for g in gene_norm.dropna().tolist()
            if g in val_genes
        }
    )
    positive_rows = int(new_label.sum())
    positive_unique_genes = int(
        df.loc[df["label_positive"] == 1, "gene_symbol"]
        .map(normalize_gene_symbol)
        .dropna()
        .nunique()
    )

    report: Dict[str, object] = {
        "label_source": "config.VALIDATION_GENES",
        "input_table": str(args.input_table),
        "output_table": str(args.output_table),
        "n_rows": int(len(df)),
        "n_unique_genes": int(df["gene_symbol"].map(normalize_gene_symbol).dropna().nunique()),
        "n_positive_rows": positive_rows,
        "n_positive_unique_genes": positive_unique_genes,
        "validation_genes_total": int(len(val_genes)),
        "validation_genes_present_in_table": int(len(present_val_genes)),
        "positive_genes_present_in_table": present_val_genes,
    }

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[done] Relabeling completed.")
    print(f"[done] Output table: {args.output_table}")
    print(
        "[done] Positives: "
        f"rows={report['n_positive_rows']} unique_genes={report['n_positive_unique_genes']}"
    )


if __name__ == "__main__":
    main()


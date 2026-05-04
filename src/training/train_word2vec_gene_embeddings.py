#!/usr/bin/env python3
"""Train a Word2Vec model on the neurodegenerative corpus (clean standalone script)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from gensim.models import Word2Vec


DEFAULT_CORPUS_CSV = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/data/corpus/preprocessed/"
    "corpus_neurodegenerative_disease_preprocessed.csv"
)
DEFAULT_OUT_DIR = Path(
    "/home/viguinijpv/200.18.99.75:8000/IC/src/data/als_cs_gene_tables/word2vec_tmp/models/word2vec"
)
DEFAULT_TEXT_COL = "text"
DEFAULT_YEAR_COL = "year"
DEFAULT_START_YEAR = 1970
DEFAULT_END_YEAR = 2026
DEFAULT_VECTOR_SIZE = 200
DEFAULT_WINDOW = 5
DEFAULT_MIN_COUNT = 5
DEFAULT_SG = 1
DEFAULT_NEGATIVE = 15
DEFAULT_EPOCHS = 15
DEFAULT_SEED = 42
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class CorpusStats:
    total_rows: int = 0
    rows_in_year_window: int = 0
    rows_with_tokens: int = 0
    total_tokens: int = 0
    min_year_observed: Optional[int] = None
    max_year_observed: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_rows": int(self.total_rows),
            "rows_in_year_window": int(self.rows_in_year_window),
            "rows_with_tokens": int(self.rows_with_tokens),
            "total_tokens": int(self.total_tokens),
            "min_year_observed": self.min_year_observed,
            "max_year_observed": self.max_year_observed,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Word2Vec model from corpus CSV.")
    parser.add_argument("--corpus-csv", type=Path, default=DEFAULT_CORPUS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--text-col", type=str, default=DEFAULT_TEXT_COL)
    parser.add_argument("--year-col", type=str, default=DEFAULT_YEAR_COL)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--vector-size", type=int, default=DEFAULT_VECTOR_SIZE)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--sg", type=int, choices=[0, 1], default=DEFAULT_SG)
    parser.add_argument("--negative", type=int, default=DEFAULT_NEGATIVE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Word2Vec worker threads.",
    )
    return parser.parse_args()


def parse_year(value: object) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def tokenize(text: object) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def row_in_year_window(year: Optional[int], start_year: int, end_year: int) -> bool:
    if year is None:
        return True
    return start_year <= year <= end_year


class CorpusSentenceIterator:
    def __init__(
        self,
        corpus_csv: Path,
        text_col: str,
        year_col: str,
        start_year: int,
        end_year: int,
    ) -> None:
        self.corpus_csv = corpus_csv
        self.text_col = text_col
        self.year_col = year_col
        self.start_year = int(start_year)
        self.end_year = int(end_year)

    def __iter__(self) -> Iterator[List[str]]:
        with self.corpus_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {self.corpus_csv}")
            required = {self.text_col, self.year_col}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

            for row in reader:
                year = parse_year(row.get(self.year_col))
                if not row_in_year_window(year, self.start_year, self.end_year):
                    continue
                tokens = tokenize(row.get(self.text_col, ""))
                if tokens:
                    yield tokens


def scan_corpus(
    corpus_csv: Path,
    text_col: str,
    year_col: str,
    start_year: int,
    end_year: int,
) -> CorpusStats:
    stats = CorpusStats()
    with corpus_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {corpus_csv}")
        required = {text_col, year_col}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row in reader:
            stats.total_rows += 1
            year = parse_year(row.get(year_col))
            if year is not None:
                if stats.min_year_observed is None or year < stats.min_year_observed:
                    stats.min_year_observed = year
                if stats.max_year_observed is None or year > stats.max_year_observed:
                    stats.max_year_observed = year

            if not row_in_year_window(year, start_year, end_year):
                continue
            stats.rows_in_year_window += 1

            tokens = tokenize(row.get(text_col, ""))
            if not tokens:
                continue
            stats.rows_with_tokens += 1
            stats.total_tokens += len(tokens)

    return stats


def write_vocab_table(model: Word2Vec, path: Path, top_n: int = 500) -> None:
    rows = []
    for token in model.wv.index_to_key[:top_n]:
        count = int(model.wv.get_vecattr(token, "count"))
        rows.append((token, count))

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["token", "count"])
        w.writerows(rows)


def main() -> None:
    args = parse_args()

    if not args.corpus_csv.exists():
        raise FileNotFoundError(f"Corpus CSV not found: {args.corpus_csv}")
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Scanning corpus: {args.corpus_csv}")
    corpus_stats = scan_corpus(
        corpus_csv=args.corpus_csv,
        text_col=args.text_col,
        year_col=args.year_col,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
    )
    print(f"[info] Corpus stats: {corpus_stats.as_dict()}")

    sentences = CorpusSentenceIterator(
        corpus_csv=args.corpus_csv,
        text_col=args.text_col,
        year_col=args.year_col,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
    )

    print("[info] Training Word2Vec...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=int(args.vector_size),
        window=int(args.window),
        min_count=int(args.min_count),
        sg=int(args.sg),
        negative=int(args.negative),
        epochs=int(args.epochs),
        workers=int(args.workers),
        seed=int(args.seed),
        alpha=0.05
    )

    model_path = out_dir / "word2vec_neurodegenerative_disease.model"
    kv_path = out_dir / "word2vec_neurodegenerative_disease.kv"
    vocab_path = out_dir / "word2vec_vocab_top500.tsv"
    summary_path = out_dir / "word2vec_training_summary.json"

    model.save(str(model_path))
    model.wv.save(str(kv_path))
    write_vocab_table(model, vocab_path, top_n=500)

    summary = {
        "corpus_csv": str(args.corpus_csv),
        "text_col": str(args.text_col),
        "year_col": str(args.year_col),
        "year_window": {"start_year": int(args.start_year), "end_year": int(args.end_year)},
        "preprocessing": {
            "tokenization": "regex [A-Za-z0-9_]+",
            "lowercase": True,
            "row_filter": "keep rows within year window (rows without parseable year are kept)",
            "empty_token_rows_dropped": True,
        },
        "word2vec_parameters": {
            "vector_size": int(args.vector_size),
            "window": int(args.window),
            "min_count": int(args.min_count),
            "sg": int(args.sg),
            "negative": int(args.negative),
            "epochs": int(args.epochs),
            "workers": int(args.workers),
            "seed": int(args.seed),
        },
        "corpus_stats": corpus_stats.as_dict(),
        "model_stats": {
            "vocabulary_size": int(len(model.wv.index_to_key)),
            "embedding_dimension": int(model.wv.vector_size),
        },
        "outputs": {
            "model_path": str(model_path),
            "keyed_vectors_path": str(kv_path),
            "vocab_top500_tsv": str(vocab_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[info] Vocabulary size: {len(model.wv.index_to_key)}")
    print(f"[info] Embedding dim: {model.wv.vector_size}")
    print(f"[info] Wrote: {model_path}")
    print(f"[info] Wrote: {kv_path}")
    print(f"[info] Wrote: {vocab_path}")
    print(f"[info] Wrote: {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""STRING-based gene network propagation utilities.

This module builds a gene-level graph from STRING protein links and runs
personalized PageRank (random walk with restart) to produce one scalar score
per gene.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse


def normalize_gene_symbol(gene: object) -> Optional[str]:
    if gene is None:
        return None
    s = str(gene).strip().upper()
    return s if s else None


@dataclass
class GeneNetworkModel:
    """Gene-level transition model for random-walk propagation."""

    gene_symbols: np.ndarray
    transition_matrix: sparse.csr_matrix  # row-stochastic

    def __post_init__(self) -> None:
        self.gene_symbols = np.asarray(self.gene_symbols, dtype=object)
        self.transition_matrix = self.transition_matrix.tocsr().astype(np.float64)
        self.gene_to_idx: Dict[str, int] = {
            str(g): i for i, g in enumerate(self.gene_symbols.tolist())
        }

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mat = self.transition_matrix.tocsr()
        np.savez_compressed(
            path,
            gene_symbols=np.asarray(self.gene_symbols, dtype=object),
            transition_data=mat.data.astype(np.float64),
            transition_indices=mat.indices.astype(np.int64),
            transition_indptr=mat.indptr.astype(np.int64),
            transition_shape=np.asarray(mat.shape, dtype=np.int64),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "GeneNetworkModel":
        with np.load(path, allow_pickle=True) as z:
            gene_symbols = np.asarray(z["gene_symbols"], dtype=object)
            data = np.asarray(z["transition_data"], dtype=np.float64)
            indices = np.asarray(z["transition_indices"], dtype=np.int64)
            indptr = np.asarray(z["transition_indptr"], dtype=np.int64)
            shape_arr = np.asarray(z["transition_shape"], dtype=np.int64)
            shape = (int(shape_arr[0]), int(shape_arr[1]))
        mat = sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=np.float64)
        return cls(gene_symbols=gene_symbols, transition_matrix=mat)

    def personalized_pagerank(
        self,
        seed_genes: Iterable[str],
        *,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-9,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        n = int(self.transition_matrix.shape[0])
        out_zero = np.zeros(n, dtype=np.float64)
        seed_set = sorted(
            {normalize_gene_symbol(g) for g in seed_genes if normalize_gene_symbol(g) is not None}
        )
        seed_indices = [self.gene_to_idx[g] for g in seed_set if g in self.gene_to_idx]
        if n == 0 or len(seed_indices) == 0:
            return out_zero, {
                "seed_count": float(len(seed_set)),
                "seed_mapped_count": 0.0,
                "iterations": 0.0,
                "converged": 0.0,
            }

        if alpha < 0.0 or alpha >= 1.0:
            raise ValueError("alpha must be in [0, 1).")
        max_iter = max(int(max_iter), 1)
        tol = float(max(tol, 0.0))

        s = np.zeros(n, dtype=np.float64)
        s[np.asarray(seed_indices, dtype=np.int64)] = 1.0
        s /= float(s.sum())

        x = s.copy()
        pt = self.transition_matrix.transpose().tocsr()
        converged = False
        n_iter_done = 0
        for i in range(max_iter):
            x_new = alpha * (pt @ x) + (1.0 - alpha) * s
            diff = float(np.abs(x_new - x).sum())
            x = x_new
            n_iter_done = i + 1
            if diff <= tol:
                converged = True
                break

        return x, {
            "seed_count": float(len(seed_set)),
            "seed_mapped_count": float(len(seed_indices)),
            "iterations": float(n_iter_done),
            "converged": 1.0 if converged else 0.0,
        }

    def score_genes(
        self,
        genes: Sequence[object],
        scores: np.ndarray,
        *,
        default_score: float = 0.0,
    ) -> np.ndarray:
        out = np.full(len(genes), float(default_score), dtype=np.float64)
        for i, g_raw in enumerate(genes):
            g = normalize_gene_symbol(g_raw)
            if g is None:
                continue
            idx = self.gene_to_idx.get(g)
            if idx is None:
                continue
            out[i] = float(scores[idx])
        return out


def load_string_protein_to_gene_map(
    aliases_path: Path,
    *,
    preferred_sources: Sequence[str] = ("Ensembl_HGNC_symbol", "Ensembl_HGNC"),
) -> Tuple[Dict[str, str], Dict[str, float]]:
    if not aliases_path.exists():
        raise FileNotFoundError(f"STRING aliases file not found: {aliases_path}")

    source_rank = {str(src): i for i, src in enumerate(preferred_sources)}
    protein_to_gene: Dict[str, str] = {}
    protein_to_rank: Dict[str, int] = {}

    lines_seen = 0
    source_hits = 0
    replaced_by_better_source = 0
    conflicting_same_rank = 0
    skipped_bad_gene = 0

    with open(aliases_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            lines_seen += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            protein_id, alias, source = parts[0], parts[1], parts[2]
            rank = source_rank.get(source)
            if rank is None:
                continue
            source_hits += 1
            gene = normalize_gene_symbol(alias)
            if gene is None:
                skipped_bad_gene += 1
                continue
            cur_rank = protein_to_rank.get(protein_id)
            cur_gene = protein_to_gene.get(protein_id)
            if cur_rank is None:
                protein_to_gene[protein_id] = gene
                protein_to_rank[protein_id] = rank
                continue
            if rank < cur_rank:
                protein_to_gene[protein_id] = gene
                protein_to_rank[protein_id] = rank
                replaced_by_better_source += 1
            elif rank == cur_rank and cur_gene != gene:
                conflicting_same_rank += 1

    stats = {
        "alias_lines_seen": float(lines_seen),
        "alias_source_hits": float(source_hits),
        "protein_to_gene_mapped": float(len(protein_to_gene)),
        "replaced_by_better_source": float(replaced_by_better_source),
        "conflicting_same_rank": float(conflicting_same_rank),
        "skipped_bad_gene": float(skipped_bad_gene),
    }
    return protein_to_gene, stats


def build_string_gene_network_model(
    links_path: Path,
    protein_to_gene: Mapping[str, str],
    *,
    min_combined_score: float = 400.0,
) -> Tuple[GeneNetworkModel, Dict[str, float]]:
    if not links_path.exists():
        raise FileNotFoundError(f"STRING links file not found: {links_path}")
    min_score = float(min_combined_score)

    edge_weight_max: Dict[Tuple[str, str], float] = {}
    lines_seen = 0
    skipped_below_threshold = 0
    skipped_unmapped = 0
    skipped_self = 0

    with open(links_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line.startswith("protein1 "):
                continue
            lines_seen += 1
            parts = line.rstrip("\n").split()
            if len(parts) < 3:
                continue
            p1, p2 = parts[0], parts[1]
            try:
                score = float(parts[2])
            except ValueError:
                continue
            if score < min_score:
                skipped_below_threshold += 1
                continue
            g1 = protein_to_gene.get(p1)
            g2 = protein_to_gene.get(p2)
            if g1 is None or g2 is None:
                skipped_unmapped += 1
                continue
            if g1 == g2:
                skipped_self += 1
                continue
            if g1 < g2:
                key = (g1, g2)
            else:
                key = (g2, g1)
            w = score / 1000.0
            prev = edge_weight_max.get(key)
            if prev is None or w > prev:
                edge_weight_max[key] = w

    if not edge_weight_max:
        model = GeneNetworkModel(
            gene_symbols=np.asarray([], dtype=object),
            transition_matrix=sparse.csr_matrix((0, 0), dtype=np.float64),
        )
        return model, {
            "link_lines_seen": float(lines_seen),
            "skipped_below_threshold": float(skipped_below_threshold),
            "skipped_unmapped": float(skipped_unmapped),
            "skipped_self": float(skipped_self),
            "gene_edges": 0.0,
            "gene_nodes": 0.0,
            "min_combined_score": float(min_score),
        }

    genes = sorted({g for pair in edge_weight_max.keys() for g in pair})
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for (g1, g2), w in edge_weight_max.items():
        i = gene_to_idx[g1]
        j = gene_to_idx[g2]
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])

    n = len(genes)
    adj = sparse.csr_matrix((np.asarray(data, dtype=np.float64), (rows, cols)), shape=(n, n), dtype=np.float64)
    row_sum = np.asarray(adj.sum(axis=1)).ravel()
    inv_row_sum = np.zeros_like(row_sum, dtype=np.float64)
    mask = row_sum > 0
    inv_row_sum[mask] = 1.0 / row_sum[mask]
    transition = sparse.diags(inv_row_sum, offsets=0, format="csr") @ adj

    model = GeneNetworkModel(gene_symbols=np.asarray(genes, dtype=object), transition_matrix=transition)
    stats = {
        "link_lines_seen": float(lines_seen),
        "skipped_below_threshold": float(skipped_below_threshold),
        "skipped_unmapped": float(skipped_unmapped),
        "skipped_self": float(skipped_self),
        "gene_edges": float(len(edge_weight_max)),
        "gene_nodes": float(len(genes)),
        "min_combined_score": float(min_score),
    }
    return model, stats


def load_or_build_string_gene_network_model(
    *,
    aliases_path: Path,
    links_path: Path,
    cache_path: Optional[Path] = None,
    min_combined_score: float = 400.0,
    preferred_sources: Sequence[str] = ("Ensembl_HGNC_symbol", "Ensembl_HGNC"),
) -> Tuple[GeneNetworkModel, Dict[str, float]]:
    if cache_path is not None and cache_path.exists():
        model = GeneNetworkModel.load_npz(cache_path)
        return model, {"from_cache": 1.0}

    protein_to_gene, alias_stats = load_string_protein_to_gene_map(
        aliases_path=aliases_path,
        preferred_sources=preferred_sources,
    )
    model, link_stats = build_string_gene_network_model(
        links_path=links_path,
        protein_to_gene=protein_to_gene,
        min_combined_score=min_combined_score,
    )

    if cache_path is not None:
        model.save_npz(cache_path)

    stats: Dict[str, float] = {"from_cache": 0.0}
    stats.update(alias_stats)
    stats.update(link_stats)
    return model, stats

import json
import pickle
import argparse
import csv
import re
import html
from pathlib import Path
from collections import Counter
from itertools import product
from typing import Dict, List, Tuple, Set
from urllib.parse import urljoin
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import matplotlib.pyplot as plt

try:
    import umap.umap_ as umap
except ImportError as e:
    raise ImportError(
        "UMAP is required. Install with: pip install umap-learn"
    ) from e


DEFAULT_HGNC_DOWNLOAD_PAGE = "https://www.genenames.org/download/"
DEFAULT_HGNC_URL_CANDIDATES = [
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
    "https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt",
]
MAX_GRID_COMBINATIONS = 12


def parse_numeric_arg_list(raw_values: List[str], cast_fn, arg_name: str) -> List:
    values = []
    for raw in raw_values:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(cast_fn(token))
            except ValueError as e:
                raise ValueError(f"Invalid value '{token}' for {arg_name}.") from e

    if not values:
        raise ValueError(f"{arg_name} must have at least one value.")

    dedup = []
    seen = set()
    for val in values:
        if val in seen:
            continue
        dedup.append(val)
        seen.add(val)
    return dedup


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot UMAP of article-level embeddings colored by gene."
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="../features/featuresUPPER_pubmedbert_neurodegenerative_disease/features_ALS_pubmedbert.pkl",
        help="Path to feature pickle (gene -> list of concatenated [gene_i]+[disease_i] vectors).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="../scores/neurodegenerative_diseaseUPPER/embedding_umap_pubmedbert_neurodegenerative_disease",
        help="Directory to save UMAP outputs.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gene", "disease", "concat", "all"],
        default="all",
        help="Which embedding view to project.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=30000,
        help="Max number of points to plot (random subsample if exceeded).",
    )
    parser.add_argument(
        "--top-genes",
        type=int,
        default=20,
        help="Number of most frequent genes to keep as individual colors (others become OTHER).",
    )
    parser.add_argument(
        "--min-points-per-gene",
        type=int,
        default=30,
        help="Minimum points for a gene to receive its own color.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=str,
        nargs="+",
        default=["10"],#["5", "10", "15", "30", "50", "100", "200"],
        help="One or more n_neighbors values (space and/or comma separated).",
    )
    parser.add_argument(
        "--min-dist",
        type=str,
        nargs="+",
        default=["0.5"],#["0.0", "0.01", "0.1", "0.3", "0.5", "0.8"],
        help="One or more min_dist values (space and/or comma separated).",
    )
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-interpretation",
        action="store_true",
        help="Disable lightweight interpretation summaries (PCA extremes + centroid distances).",
    )
    parser.add_argument(
        "--interpret-min-gene-points",
        type=int,
        default=20,
        help="Minimum points per gene to include it in interpretation summaries.",
    )
    parser.add_argument(
        "--interpret-max-genes",
        type=int,
        default=250,
        help="Maximum number of genes (by frequency) used for interpretation summaries.",
    )
    parser.add_argument(
        "--interpret-components",
        type=int,
        default=2,
        help="Number of PCA components to summarize for interpretation.",
    )
    parser.add_argument(
        "--interpret-extreme-genes",
        type=int,
        default=12,
        help="Genes to report from each low/high end of each PCA component.",
    )
    parser.add_argument(
        "--interpret-nearest-k",
        type=int,
        default=5,
        help="Nearest centroid neighbors to report for each anchor gene.",
    )

    parser.add_argument(
        "--hgnc-path",
        type=str,
        default="",
        help="Optional local HGNC TSV path (hgnc_complete_set.txt).",
    )
    parser.add_argument(
        "--hgnc-cache-path",
        type=str,
        default="../data/reference/hgnc_complete_set.txt",
        help="Local cache path for HGNC TSV if auto-download is needed.",
    )
    parser.add_argument(
        "--hgnc-url",
        type=str,
        default=DEFAULT_HGNC_DOWNLOAD_PAGE,
        help="HGNC direct TSV URL OR HGNC downloads page URL.",
    )
    parser.add_argument(
        "--no-hgnc-filter",
        action="store_true",
        help="Disable HGNC filtering.",
    )
    parser.add_argument(
        "--no-hgnc-download",
        action="store_true",
        help="Do not auto-download HGNC when local file is missing.",
    )
    args = parser.parse_args()

    try:
        args.n_neighbors = parse_numeric_arg_list(args.n_neighbors, int, "--n-neighbors")
        args.min_dist = parse_numeric_arg_list(args.min_dist, float, "--min-dist")
    except ValueError as e:
        parser.error(str(e))

    if any(v < 2 for v in args.n_neighbors):
        parser.error("--n-neighbors must be >= 2.")
    if any(v < 0 for v in args.min_dist):
        parser.error("--min-dist must be >= 0.")
    if args.interpret_min_gene_points < 1:
        parser.error("--interpret-min-gene-points must be >= 1.")
    if args.interpret_max_genes < 3:
        parser.error("--interpret-max-genes must be >= 3.")
    if args.interpret_components < 1:
        parser.error("--interpret-components must be >= 1.")
    if args.interpret_extreme_genes < 1:
        parser.error("--interpret-extreme-genes must be >= 1.")
    if args.interpret_nearest_k < 1:
        parser.error("--interpret-nearest-k must be >= 1.")

    return args


def load_feature_bags(path: str) -> Dict[str, List[np.ndarray]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError("Feature file must be a dict: gene -> list[np.ndarray].")
    return data


def flatten_gene_bags(gene_bags: Dict[str, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    vectors = []
    genes = []
    for gene, bag in gene_bags.items():
        gene_u = str(gene).strip().upper()
        if not bag:
            continue
        for vec in bag:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size < 2:
                continue
            vectors.append(arr)
            genes.append(gene_u)

    if not vectors:
        raise ValueError("No valid vectors found in feature file.")
    return np.stack(vectors, axis=0), np.asarray(genes)


def subsample_points(X: np.ndarray, genes: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or X.shape[0] <= max_points:
        return X, genes
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_points, replace=False)
    return X[idx], genes[idx]


def split_embedding_views(X_concat: np.ndarray) -> Dict[str, np.ndarray]:
    dim = X_concat.shape[1]
    if dim % 2 != 0:
        raise ValueError(
            f"Concatenated vector dimension must be even to split [gene]+[disease]. Got dim={dim}."
        )
    half = dim // 2
    return {
        "gene": X_concat[:, :half],
        "disease": X_concat[:, half:],
        "concat": X_concat,
    }


def validate_umap_inputs(X: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape={X.shape}.")
    if X.shape[0] < 3:
        raise ValueError(
            f"UMAP requires at least 3 points after filtering/subsampling. Got n_points={X.shape[0]}."
        )
    finite_mask = np.isfinite(X)
    if not np.all(finite_mask):
        n_bad = int(X.size - np.count_nonzero(finite_mask))
        raise ValueError(f"Embedding matrix contains {n_bad} non-finite value(s) (NaN/Inf).")


def get_effective_n_neighbors(requested_n_neighbors: int, n_samples: int) -> int:
    if n_samples < 3:
        raise ValueError(f"UMAP requires at least 3 samples. Got n_samples={n_samples}.")
    return min(requested_n_neighbors, n_samples - 1)


def _fetch_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as response:
        return response.read()


def extract_hgnc_urls_from_download_page(download_page_url: str) -> List[str]:
    page_bytes = _fetch_bytes(download_page_url)
    page_text = page_bytes.decode("utf-8", errors="replace")

    # Busca links diretos para hgnc_complete_set.txt na página de downloads.
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', page_text, flags=re.IGNORECASE)
    out = []
    for href in hrefs:
        href_clean = html.unescape(href).strip()
        href_l = href_clean.lower()
        if "hgnc_complete_set" in href_l and ".txt" in href_l:
            out.append(urljoin(download_page_url, href_clean))
    return out


def download_hgnc_file(candidates: List[str], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    tried = set()
    for url in candidates:
        if not url or url in tried:
            continue
        tried.add(url)
        try:
            print(f"[info] Trying HGNC download URL: {url}")
            content = _fetch_bytes(url)
            output_path.write_bytes(content)
            return url
        except (HTTPError, URLError, TimeoutError, ValueError) as e:
            errors.append(f"{url} -> {e}")

    msg = "\n".join(errors[:8]) if errors else "no valid URLs were provided"
    raise RuntimeError(f"Failed to download HGNC reference. Tried:\n{msg}")


def load_hgnc_symbols(tsv_path: Path) -> Set[str]:
    if not tsv_path.exists():
        raise FileNotFoundError(f"HGNC file not found: {tsv_path}")

    symbols = set()
    with tsv_path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if "symbol" not in header:
            raise ValueError("HGNC TSV does not contain 'symbol' column.")
        symbol_idx = header.index("symbol")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if symbol_idx >= len(parts):
                continue
            symbol = parts[symbol_idx].strip().upper()
            if symbol:
                symbols.add(symbol)
    return symbols


def resolve_hgnc_symbols(
    hgnc_path_arg: str,
    hgnc_cache_path_arg: str,
    hgnc_url: str,
    allow_download: bool,
) -> Tuple[Set[str], str]:
    if hgnc_path_arg.strip():
        path = Path(hgnc_path_arg).expanduser().resolve()
        return load_hgnc_symbols(path), str(path)

    cache_path = Path(hgnc_cache_path_arg).expanduser().resolve()
    if cache_path.exists():
        return load_hgnc_symbols(cache_path), str(cache_path)

    if not allow_download:
        raise FileNotFoundError(
            f"HGNC file not found in cache and download disabled: {cache_path}. "
            "Set --hgnc-path or allow download."
        )

    url_candidates: List[str] = []
    user_url = hgnc_url.strip()
    if user_url:
        if user_url.lower().endswith(".txt"):
            url_candidates.append(user_url)
        else:
            try:
                extracted = extract_hgnc_urls_from_download_page(user_url)
                if extracted:
                    print(f"[info] Found {len(extracted)} HGNC TXT link(s) on download page.")
                    url_candidates.extend(extracted)
                else:
                    print(f"[warn] No HGNC TXT links found on page: {user_url}")
            except Exception as e:
                print(f"[warn] Could not parse HGNC download page ({user_url}): {e}")

    url_candidates.extend(DEFAULT_HGNC_URL_CANDIDATES)
    used_url = download_hgnc_file(url_candidates, cache_path)
    return load_hgnc_symbols(cache_path), f"{cache_path} (downloaded from {used_url})"


def filter_gene_bags_by_hgnc(
    gene_bags: Dict[str, List[np.ndarray]],
    valid_symbols: Set[str],
) -> Tuple[Dict[str, List[np.ndarray]], List[str]]:
    filtered = {}
    removed = []
    for gene, bag in gene_bags.items():
        gene_u = str(gene).strip().upper()
        if gene_u in valid_symbols:
            filtered[gene_u] = bag
        else:
            removed.append(gene_u)
    return filtered, removed


def collapse_gene_labels(
    genes: np.ndarray,
    top_genes: int,
    min_points_per_gene: int,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    counts = Counter(genes.tolist())
    ranked = [g for g, c in counts.most_common() if c >= min_points_per_gene]
    keep = set(ranked[:top_genes])
    collapsed = np.asarray([g if g in keep else "OTHER" for g in genes])
    labels_order = [g for g in ranked[:top_genes] if g in keep]
    if np.any(collapsed == "OTHER"):
        labels_order.append("OTHER")
    return collapsed, labels_order, dict(counts)


def compute_gene_centroids(
    X: np.ndarray,
    genes: np.ndarray,
    min_points_per_gene: int,
    max_genes: int,
) -> Tuple[np.ndarray, List[str], Dict[str, int], Dict[str, int]]:
    all_counts = Counter(genes.tolist())
    selected_genes = [
        g
        for g, c in all_counts.most_common()
        if c >= min_points_per_gene
    ][:max_genes]
    if not selected_genes:
        return (
            np.zeros((0, X.shape[1]), dtype=np.float32),
            [],
            {},
            dict(all_counts),
        )

    gene_to_idx = {g: i for i, g in enumerate(selected_genes)}
    idx = np.asarray([gene_to_idx.get(g, -1) for g in genes], dtype=np.int32)
    valid = idx >= 0

    sums = np.zeros((len(selected_genes), X.shape[1]), dtype=np.float64)
    np.add.at(sums, idx[valid], X[valid].astype(np.float64, copy=False))
    counts_used = np.bincount(idx[valid], minlength=len(selected_genes)).astype(np.int64)

    nonzero_mask = counts_used > 0
    if not np.all(nonzero_mask):
        selected_genes = [
            gene
            for gene, keep in zip(selected_genes, nonzero_mask.tolist())
            if keep
        ]
        sums = sums[nonzero_mask]
        counts_used = counts_used[nonzero_mask]

    centroids = sums / counts_used[:, None]
    count_map = {
        gene: int(count)
        for gene, count in zip(selected_genes, counts_used.tolist())
    }
    return centroids.astype(np.float32), selected_genes, count_map, dict(all_counts)


def run_pca_on_centroids(
    centroids: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, List[float]]:
    if centroids.ndim != 2:
        raise ValueError(f"Expected 2D centroids matrix, got shape={centroids.shape}.")
    if centroids.shape[0] < 2:
        raise ValueError("Need at least 2 centroids to run PCA.")

    X = centroids.astype(np.float64, copy=False)
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(X_centered, full_matrices=False)

    n_components = min(max(1, n_components), vh.shape[0])
    components = vh[:n_components]
    scores = X_centered @ components.T

    if X_centered.shape[0] > 1:
        explained = (singular_values ** 2) / (X_centered.shape[0] - 1)
    else:
        explained = singular_values ** 2
    total = float(np.sum(explained))
    if total <= 0:
        explained_ratio = [0.0] * n_components
    else:
        explained_ratio = [float(v / total) for v in explained[:n_components]]

    return scores, explained_ratio


def run_pca_projection(
    X: np.ndarray,
    n_components: int = 2,
) -> Tuple[np.ndarray, List[float]]:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix for PCA projection, got shape={X.shape}.")
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples to run PCA projection.")

    X64 = X.astype(np.float64, copy=False)
    X_centered = X64 - np.mean(X64, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(X_centered, full_matrices=False)

    n_components = min(max(1, n_components), vh.shape[0])
    components = vh[:n_components]
    scores = X_centered @ components.T

    if X_centered.shape[0] > 1:
        explained = (singular_values ** 2) / (X_centered.shape[0] - 1)
    else:
        explained = singular_values ** 2
    total = float(np.sum(explained))
    if total <= 0:
        explained_ratio = [0.0] * n_components
    else:
        explained_ratio = [float(v / total) for v in explained[:n_components]]

    if scores.shape[1] == 1:
        scores = np.hstack([scores, np.zeros((scores.shape[0], 1), dtype=scores.dtype)])
        explained_ratio = explained_ratio + [0.0]

    return scores[:, :2], explained_ratio[:2]


def pairwise_centroid_distances(
    centroids: np.ndarray,
    metric: str,
) -> Tuple[np.ndarray, str]:
    metric_l = str(metric).strip().lower()
    X = centroids.astype(np.float64, copy=False)
    gram = X @ X.T

    if metric_l in {"cos", "cosine"}:
        norms = np.sqrt(np.maximum(np.diag(gram), 1e-12))
        denom = np.outer(norms, norms)
        similarity = gram / np.maximum(denom, 1e-12)
        similarity = np.clip(similarity, -1.0, 1.0)
        distances = 1.0 - similarity
        np.fill_diagonal(distances, 0.0)
        return distances, "cosine"

    sq = np.diag(gram)
    distances_sq = np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
    distances = np.sqrt(distances_sq)
    np.fill_diagonal(distances, 0.0)
    if metric_l in {"euclidean", "l2", "minkowski"}:
        return distances, "euclidean"
    return distances, "euclidean_fallback"


def summarize_embedding_interpretation(
    X_view: np.ndarray,
    genes: np.ndarray,
    metric: str,
    min_points_per_gene: int,
    max_genes: int,
    n_components: int,
    extreme_genes: int,
    nearest_k: int,
) -> Dict[str, object]:
    centroids, genes_used, counts_used, all_counts = compute_gene_centroids(
        X=X_view,
        genes=genes,
        min_points_per_gene=min_points_per_gene,
        max_genes=max_genes,
    )
    top_gene_counts = dict(
        sorted(all_counts.items(), key=lambda item: item[1], reverse=True)[:50]
    )
    summary = {
        "status": "ok",
        "n_points": int(X_view.shape[0]),
        "n_unique_genes_total": int(len(all_counts)),
        "n_genes_used": int(len(genes_used)),
        "min_points_per_gene": int(min_points_per_gene),
        "max_genes": int(max_genes),
        "top_gene_counts": top_gene_counts,
        "gene_counts_used": counts_used,
    }

    if len(genes_used) < 3:
        summary["status"] = "skipped"
        summary["reason"] = (
            f"Need at least 3 genes passing min_points_per_gene={min_points_per_gene}; "
            f"got {len(genes_used)}."
        )
        summary["principal_components"] = []
        summary["anchor_gene_neighbors"] = {}
        summary["closest_centroid_pairs"] = []
        summary["farthest_centroid_pairs"] = []
        return summary

    pc_scores, explained_ratio = run_pca_on_centroids(
        centroids=centroids,
        n_components=n_components,
    )
    n_genes = len(genes_used)
    k_extreme = min(max(1, extreme_genes), max(1, n_genes // 2))

    principal_components = []
    anchor_genes = set()
    for comp_idx in range(pc_scores.shape[1]):
        scores = pc_scores[:, comp_idx]
        order = np.argsort(scores)
        low_idx = order[:k_extreme]
        high_idx = order[-k_extreme:][::-1]

        low_end = [
            {
                "gene": genes_used[i],
                "score": float(scores[i]),
                "n_points": int(counts_used.get(genes_used[i], 0)),
            }
            for i in low_idx
        ]
        high_end = [
            {
                "gene": genes_used[i],
                "score": float(scores[i]),
                "n_points": int(counts_used.get(genes_used[i], 0)),
            }
            for i in high_idx
        ]
        for item in low_end:
            anchor_genes.add(item["gene"])
        for item in high_end:
            anchor_genes.add(item["gene"])

        principal_components.append(
            {
                "component": f"PC{comp_idx + 1}",
                "explained_variance_ratio": float(explained_ratio[comp_idx]),
                "low_end": low_end,
                "high_end": high_end,
            }
        )

    distances, distance_metric_used = pairwise_centroid_distances(
        centroids=centroids,
        metric=metric,
    )
    idx_by_gene = {gene: i for i, gene in enumerate(genes_used)}
    nearest_k = min(max(1, nearest_k), max(1, n_genes - 1))
    anchor_gene_neighbors = {}
    for gene in sorted(anchor_genes):
        i = idx_by_gene[gene]
        row = distances[i].copy()
        row[i] = np.inf
        nn_idx = np.argsort(row)[:nearest_k]
        anchor_gene_neighbors[gene] = [
            {
                "gene": genes_used[j],
                "distance": float(row[j]),
                "n_points": int(counts_used.get(genes_used[j], 0)),
            }
            for j in nn_idx
        ]

    tri_i, tri_j = np.triu_indices(n_genes, k=1)
    pair_distances = distances[tri_i, tri_j]
    n_pairs = min(15, int(pair_distances.shape[0]))
    pair_order = np.argsort(pair_distances)

    closest_pairs = []
    farthest_pairs = []
    for idx in pair_order[:n_pairs]:
        i = int(tri_i[idx])
        j = int(tri_j[idx])
        closest_pairs.append(
            {
                "gene_a": genes_used[i],
                "gene_b": genes_used[j],
                "distance": float(pair_distances[idx]),
                "n_points_gene_a": int(counts_used.get(genes_used[i], 0)),
                "n_points_gene_b": int(counts_used.get(genes_used[j], 0)),
            }
        )
    for idx in pair_order[-n_pairs:][::-1]:
        i = int(tri_i[idx])
        j = int(tri_j[idx])
        farthest_pairs.append(
            {
                "gene_a": genes_used[i],
                "gene_b": genes_used[j],
                "distance": float(pair_distances[idx]),
                "n_points_gene_a": int(counts_used.get(genes_used[i], 0)),
                "n_points_gene_b": int(counts_used.get(genes_used[j], 0)),
            }
        )

    summary["distance_metric_requested"] = metric
    summary["distance_metric_used"] = distance_metric_used
    summary["principal_components"] = principal_components
    summary["anchor_gene_neighbors"] = anchor_gene_neighbors
    summary["closest_centroid_pairs"] = closest_pairs
    summary["farthest_centroid_pairs"] = farthest_pairs
    return summary


def flatten_pc_extremes_rows(
    mode: str,
    view_summary: Dict[str, object],
) -> List[Dict[str, object]]:
    rows = []
    for component in view_summary.get("principal_components", []):
        component_name = str(component.get("component", ""))
        explained_ratio = float(component.get("explained_variance_ratio", 0.0))
        for side, key in [("low", "low_end"), ("high", "high_end")]:
            genes_side = component.get(key, [])
            for rank, item in enumerate(genes_side, start=1):
                rows.append(
                    {
                        "mode": mode,
                        "component": component_name,
                        "side": side,
                        "rank": rank,
                        "gene": str(item.get("gene", "")),
                        "n_points": int(item.get("n_points", 0)),
                        "score": float(item.get("score", 0.0)),
                        "explained_variance_ratio": explained_ratio,
                    }
                )
    return rows


def write_pc_extremes_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "mode",
        "component",
        "side",
        "rank",
        "gene",
        "n_points",
        "score",
        "explained_variance_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_umap(
    X: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(X)


def build_umap_param_grid(
    n_neighbors_values: List[int],
    min_dist_values: List[float],
    metric: str,
    seed: int,
) -> List[Dict[str, object]]:
    return [
        {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "seed": seed,
        }
        for n_neighbors, min_dist in product(n_neighbors_values, min_dist_values)
    ]


def float_to_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return token or "value"


def plot_failure_panel(ax, title: str, error_text: str):
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        f"UMAP failed\n{error_text[:240]}",
        ha="center",
        va="center",
        fontsize=8,
        color="#8b0000",
        wrap=True,
    )
    ax.set_axis_off()


def plot_umap_panel(
    ax,
    emb2d: np.ndarray,
    labels: np.ndarray,
    labels_order: List[str],
    title: str,
):
    cmap = plt.get_cmap("tab20")
    color_lookup = {}
    color_i = 0
    for label in labels_order:
        if label == "OTHER":
            color_lookup[label] = "#bdbdbd"
        else:
            color_lookup[label] = cmap(color_i % 20)
            color_i += 1

    for label in labels_order:
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            emb2d[mask, 0],
            emb2d[mask, 1],
            s=8,
            alpha=0.7 if label != "OTHER" else 0.35,
            c=[color_lookup[label]],
            label=label,
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, linestyle=":", alpha=0.25)


def plot_pca_panel(
    ax,
    emb2d: np.ndarray,
    labels: np.ndarray,
    labels_order: List[str],
    title: str,
):
    cmap = plt.get_cmap("tab20")
    color_lookup = {}
    color_i = 0
    for label in labels_order:
        if label == "OTHER":
            color_lookup[label] = "#bdbdbd"
        else:
            color_lookup[label] = cmap(color_i % 20)
            color_i += 1

    for label in labels_order:
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            emb2d[mask, 0],
            emb2d[mask, 1],
            s=8,
            alpha=0.7 if label != "OTHER" else 0.35,
            c=[color_lookup[label]],
            label=label,
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle=":", alpha=0.25)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Loading features from: {args.features_path}")
    gene_bags = load_feature_bags(args.features_path)

    hgnc_source_path = None
    removed_genes = []
    if not args.no_hgnc_filter:
        valid_hgnc_symbols, hgnc_source_path = resolve_hgnc_symbols(
            hgnc_path_arg=args.hgnc_path,
            hgnc_cache_path_arg=args.hgnc_cache_path,
            hgnc_url=args.hgnc_url,
            allow_download=(not args.no_hgnc_download),
        )
        n_before = len(gene_bags)
        gene_bags, removed_genes = filter_gene_bags_by_hgnc(gene_bags, valid_hgnc_symbols)
        print(
            f"[info] HGNC filter enabled: kept {len(gene_bags)}/{n_before} genes "
            f"(removed {len(removed_genes)} non-HGNC symbols)."
        )
        if len(gene_bags) == 0:
            raise ValueError("All genes were removed by HGNC filtering. Check HGNC source/path.")
    X_concat, genes = flatten_gene_bags(gene_bags)
    X_concat, genes = subsample_points(X_concat, genes, args.max_points, args.seed)
    validate_umap_inputs(X_concat)

    views = split_embedding_views(X_concat)
    collapsed_labels, labels_order, raw_counts = collapse_gene_labels(
        genes,
        top_genes=args.top_genes,
        min_points_per_gene=args.min_points_per_gene,
    )

    modes = ["gene", "disease", "concat"] if args.mode == "all" else [args.mode]
    pca_coords = {}
    pca_runs = {}
    pca_failed_runs = []
    pca_fig, pca_axes = plt.subplots(
        1,
        len(modes),
        figsize=(7 * len(modes), 6),
        squeeze=False,
    )
    for col_i, mode in enumerate(modes):
        pca_ax = pca_axes[0, col_i]
        try:
            pca2d, explained = run_pca_projection(views[mode], n_components=2)
            pca_key = f"{mode}_pc1_pc2"
            pca_coords[pca_key] = pca2d.astype(np.float32)
            pca_runs[mode] = {
                "coord_key": pca_key,
                "explained_variance_ratio": [float(x) for x in explained],
            }
            plot_pca_panel(
                pca_ax,
                emb2d=pca2d,
                labels=collapsed_labels,
                labels_order=labels_order,
                title=(
                    f"PCA: {mode} | EVR PC1={100 * explained[0]:.1f}%, "
                    f"PC2={100 * explained[1]:.1f}%"
                ),
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            pca_runs[mode] = {"error": err}
            pca_failed_runs.append({"mode": mode, "error": err})
            print(f"[warn] PCA plot failed for mode='{mode}': {err}")
            plot_failure_panel(
                pca_ax,
                title=f"PCA: {mode}",
                error_text=err,
            )

    pca_handles, pca_legend_labels = pca_axes[0, -1].get_legend_handles_labels()
    if pca_handles:
        max_legend = min(len(pca_legend_labels), args.top_genes + 1)
        pca_axes[0, -1].legend(
            pca_handles[:max_legend],
            pca_legend_labels[:max_legend],
            loc="best",
            fontsize=8,
            frameon=True,
        )
    pca_fig.suptitle(
        f"Embedding Space PCA ({len(genes)} points, colored by gene)",
        y=1.02,
        fontsize=14,
    )
    pca_fig.tight_layout(rect=(0, 0, 1, 0.96))
    pca_plot_path = out_dir / f"pca_{args.mode}_colored_by_gene.png"
    pca_fig.savefig(pca_plot_path, dpi=300, bbox_inches="tight")
    plt.close(pca_fig)
    print(f"[info] PCA plot saved to: {pca_plot_path}")

    interpretation_by_mode = {}
    interpretation_rows = []
    if args.no_interpretation:
        print("[info] Embedding interpretation disabled (--no-interpretation).")
    else:
        for mode in modes:
            print(
                f"[info] Building interpretation summary for mode='{mode}' "
                f"(min_gene_points={args.interpret_min_gene_points}, "
                f"max_genes={args.interpret_max_genes}, "
                f"components={args.interpret_components})..."
            )
            mode_summary = summarize_embedding_interpretation(
                X_view=views[mode],
                genes=genes,
                metric=args.metric,
                min_points_per_gene=args.interpret_min_gene_points,
                max_genes=args.interpret_max_genes,
                n_components=args.interpret_components,
                extreme_genes=args.interpret_extreme_genes,
                nearest_k=args.interpret_nearest_k,
            )
            interpretation_by_mode[mode] = mode_summary
            interpretation_rows.extend(
                flatten_pc_extremes_rows(mode=mode, view_summary=mode_summary)
            )

    param_grid = build_umap_param_grid(
        n_neighbors_values=args.n_neighbors,
        min_dist_values=args.min_dist,
        metric=args.metric,
        seed=args.seed,
    )

    ncols = len(modes)
    render_as_grid = len(param_grid) <= MAX_GRID_COMBINATIONS
    if render_as_grid:
        nrows = len(param_grid)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7 * ncols, 5.2 * nrows),
            squeeze=False,
        )
    else:
        print(
            f"[warn] {len(param_grid)} hyperparameter combinations requested. "
            "Saving one figure per combination to avoid a very large panel."
        )
        fig = None
        axes = None

    umap_coords = {}
    runs = []
    failed_runs = []
    plot_paths = []
    warned_n_neighbors = set()
    for row_i, params in enumerate(param_grid):
        requested_n_neighbors = int(params["n_neighbors"])
        n_neighbors = get_effective_n_neighbors(requested_n_neighbors, X_concat.shape[0])
        min_dist = float(params["min_dist"])
        metric = str(params["metric"])
        seed = int(params["seed"])

        if n_neighbors != requested_n_neighbors and requested_n_neighbors not in warned_n_neighbors:
            print(
                f"[warn] Requested n_neighbors={requested_n_neighbors} exceeds maximum valid "
                f"value for n_samples={X_concat.shape[0]}. Using n_neighbors={n_neighbors}."
            )
            warned_n_neighbors.add(requested_n_neighbors)

        if render_as_grid:
            current_fig = fig
            current_axes = axes
        else:
            current_fig, current_axes = plt.subplots(
                1,
                ncols,
                figsize=(7 * ncols, 6),
                squeeze=False,
            )

        run_info = {
            "requested_n_neighbors": requested_n_neighbors,
            "effective_n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "seed": seed,
            "mode_coord_keys": {},
            "mode_errors": {},
        }

        for col_i, mode in enumerate(modes):
            ax = current_axes[row_i, col_i] if render_as_grid else current_axes[0, col_i]
            print(
                f"[info] Running UMAP for mode='{mode}' with {views[mode].shape[0]} points "
                f"(n_neighbors={n_neighbors}, min_dist={min_dist:g}, metric={metric})..."
            )
            try:
                emb2d = run_umap(
                    views[mode],
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    seed=seed,
                )
                coord_key = (
                    f"{mode}_nn{n_neighbors}_md{float_to_token(min_dist)}_"
                    f"metric_{sanitize_token(metric)}"
                )
                umap_coords[coord_key] = emb2d
                run_info["mode_coord_keys"][mode] = coord_key
                plot_umap_panel(
                    ax,
                    emb2d=emb2d,
                    labels=collapsed_labels,
                    labels_order=labels_order,
                    title=f"UMAP: {mode} | nn={n_neighbors}, md={min_dist:g}",
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                run_info["mode_errors"][mode] = err
                failed_runs.append(
                    {
                        "mode": mode,
                        "requested_n_neighbors": requested_n_neighbors,
                        "effective_n_neighbors": n_neighbors,
                        "min_dist": min_dist,
                        "metric": metric,
                        "seed": seed,
                        "error": err,
                    }
                )
                print(
                    f"[warn] UMAP failed for mode='{mode}' "
                    f"(nn={n_neighbors}, min_dist={min_dist:g}, metric={metric}): {err}"
                )
                plot_failure_panel(
                    ax,
                    title=f"UMAP: {mode} | nn={n_neighbors}, md={min_dist:g}",
                    error_text=err,
                )

        if render_as_grid:
            runs.append(run_info)
        else:
            handles, legend_labels = current_axes[0, -1].get_legend_handles_labels()
            if handles:
                max_legend = min(len(legend_labels), args.top_genes + 1)
                current_axes[0, -1].legend(
                    handles[:max_legend],
                    legend_labels[:max_legend],
                    loc="best",
                    fontsize=8,
                    frameon=True,
                )
            current_fig.suptitle(
                (
                    f"Embedding Space UMAP ({len(genes)} points, colored by gene)\n"
                    f"nn={n_neighbors}, min_dist={min_dist:g}, metric={metric}"
                ),
                y=1.03,
                fontsize=14,
            )
            current_fig.tight_layout(rect=(0, 0, 1, 0.95))
            run_plot_path = out_dir / (
                f"umap_{args.mode}_nn{n_neighbors}_md{float_to_token(min_dist)}_"
                f"metric_{sanitize_token(metric)}_colored_by_gene.png"
            )
            current_fig.savefig(run_plot_path, dpi=300, bbox_inches="tight")
            plt.close(current_fig)
            run_info["plot_path"] = str(run_plot_path)
            plot_paths.append(str(run_plot_path))
            runs.append(run_info)

    if render_as_grid:
        handles, legend_labels = axes[0, -1].get_legend_handles_labels()
        if handles:
            max_legend = min(len(legend_labels), args.top_genes + 1)
            axes[0, -1].legend(
                handles[:max_legend],
                legend_labels[:max_legend],
                loc="best",
                fontsize=8,
                frameon=True,
            )

        fig.suptitle(
            (
                f"Embedding Space UMAP ({len(genes)} points, colored by gene)\n"
                f"{len(param_grid)} hyperparameter combination(s)"
            ),
            y=1.01,
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        if len(param_grid) == 1:
            plot_path = out_dir / f"umap_{args.mode}_colored_by_gene.png"
        else:
            plot_path = out_dir / f"umap_{args.mode}_colored_by_gene_hparam_grid.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        for run in runs:
            run["plot_path"] = str(plot_path)
        plot_paths.append(str(plot_path))

    meta = {
        "features_path": args.features_path,
        "hgnc_filter_enabled": bool(not args.no_hgnc_filter),
        "hgnc_source_path": hgnc_source_path,
        "n_removed_non_hgnc_genes": int(len(removed_genes)),
        "removed_non_hgnc_examples": sorted(set(removed_genes))[:50],
        "n_points": int(len(genes)),
        "n_unique_genes": int(len(set(genes.tolist()))),
        "embedding_dim_concat": int(X_concat.shape[1]),
        "modes": modes,
        "umap_params": {
            "n_neighbors_values": args.n_neighbors,
            "min_dist_values": args.min_dist,
            "metric": args.metric,
            "seed": args.seed,
            "n_combinations": len(param_grid),
            "render_strategy": (
                "single_grid" if render_as_grid else "one_figure_per_combination"
            ),
            "max_grid_combinations": MAX_GRID_COMBINATIONS,
        },
        "runs": runs,
        "n_successful_embeddings": int(len(umap_coords)),
        "n_failed_embeddings": int(len(failed_runs)),
        "failed_runs": failed_runs[:200],
        "coloring": {
            "top_genes": args.top_genes,
            "min_points_per_gene": args.min_points_per_gene,
            "legend_labels": labels_order,
        },
        "top_gene_counts": dict(sorted(raw_counts.items(), key=lambda x: x[1], reverse=True)[:50]),
        "plot_path": plot_paths[0] if plot_paths else "",
        "plot_paths": plot_paths,
        "pca": {
            "plot_path": str(pca_plot_path),
            "runs": pca_runs,
            "n_successful": int(len(pca_coords)),
            "n_failed": int(len(pca_failed_runs)),
            "failed_runs": pca_failed_runs,
        },
        "embedding_interpretation": {
            "enabled": bool(not args.no_interpretation),
            "params": {
                "interpret_min_gene_points": args.interpret_min_gene_points,
                "interpret_max_genes": args.interpret_max_genes,
                "interpret_components": args.interpret_components,
                "interpret_extreme_genes": args.interpret_extreme_genes,
                "interpret_nearest_k": args.interpret_nearest_k,
                "distance_metric_requested": args.metric,
            },
            "views": interpretation_by_mode,
        },
    }

    interpretation_json_path = out_dir / "embedding_interpretation.json"
    interpretation_tsv_path = out_dir / "embedding_interpretation_pc_extremes.tsv"
    if interpretation_by_mode:
        with open(interpretation_json_path, "w") as f:
            json.dump(interpretation_by_mode, f, indent=2)
        meta["embedding_interpretation"]["json_path"] = str(interpretation_json_path)
        print(f"[info] Interpretation JSON saved to: {interpretation_json_path}")

    if interpretation_rows:
        write_pc_extremes_tsv(interpretation_tsv_path, interpretation_rows)
        meta["embedding_interpretation"]["pc_extremes_tsv_path"] = str(
            interpretation_tsv_path
        )
        print(f"[info] PCA extreme genes TSV saved to: {interpretation_tsv_path}")

    meta_path = out_dir / "umap_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    coords_path = out_dir / "umap_coords.npz"
    npz_payload = {
        "genes": genes.astype(str),
        "labels_collapsed": collapsed_labels.astype(str),
    }
    for key, value in umap_coords.items():
        npz_payload[f"umap_{key}"] = value.astype(np.float32)
    for key, value in pca_coords.items():
        npz_payload[f"pca_{key}"] = value.astype(np.float32)
    np.savez_compressed(coords_path, **npz_payload)

    if plot_paths:
        if len(plot_paths) == 1:
            print(f"[info] Plot saved to: {plot_paths[0]}")
        else:
            print(f"[info] Saved {len(plot_paths)} plot files to: {out_dir}")
    print(f"[info] Metadata saved to: {meta_path}")
    print(f"[info] Coordinates saved to: {coords_path}")

    if not umap_coords:
        raise RuntimeError("All UMAP runs failed. Check umap_metadata.json for details.")


if __name__ == "__main__":
    main()

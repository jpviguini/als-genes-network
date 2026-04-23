# Build Variant-Gene Candidate Feature Table

This document describes the current pipeline used in this project to build the ALS feature matrix at **variant level**.

Main scripts:

- `src/training/build_variant_gene_candidate_feature_table.py`
- `src/training/train_l1_logreg_clinvar_eva.py` (downstream training)

## Goal

Build a feature matrix for one GWAS study (`GCST90027164`) where each row is:

- `(gwas_variant_id, gene)`

In practice, rows are uniquely identified by:

- `(gwas_study_locus_id, gwas_variant_id, gene_symbol)`  
and when available also by `gene_id_version`.

Candidate genes are all protein-coding genes within `+/- 500 kb` of each GWAS credible-set variant.

## Data sources

### Open Targets GraphQL API

Endpoint:

- `https://api.platform.opentargets.org/api/v4/graphql`

Main objects queried:

1. `study(studyId).credibleSets`
- fetches GWAS credible set loci for `GCST90027164`

2. `credibleSet(studyLocusId).locus`
- fetches all GWAS variants in each study locus
- includes posterior probability and 95/99% credible-set flags

3. disease associations with datasource `eva` (ClinVar)
- used to build:
  - `clinvar_eva_score`
  - `label_positive` (`1` if `clinvar_eva_score >= 0.5`, else `0`)

### Local reference annotation

- `src/data/reference/gencode.v38.annotation.gtf.gz`

Used to extract protein-coding gene coordinates and TSS on GRCh38.

### Local variant-level QTL/coloc evidence (legacy table)

- `src/data/als_cs_gene_tables/GCST90027164_variant_gene_feature_table.parquet`
- fallback: `src/data/als_cs_gene_tables/GCST90027164_variant_gene_feature_table.csv`

Used as a left-joined evidence source at variant-gene level.

### Local embeddings

- `src/features/featuresUPPER_pubmedbert_neurodegenerative_disease/features_ALS_pubmedbert.pkl`

Used to add:

- `has_gene_embedding`
- `gene_emb_0000 ... gene_emb_0767`

## Integration pipeline

1. Fetch GWAS credible sets for `GCST90027164`.
2. For each `gwas_study_locus_id`, fetch all variants from `credibleSet(...).locus`.
3. Build candidate rows `(variant, gene)` using all protein-coding genes in `±500 kb` around each variant.
4. Compute variant-to-gene distance features directly from coordinates.
5. Left-join legacy variant-level QTL/coloc evidence:
   - primary key: `(gwas_study_locus_id, gwas_variant_id, gene_id)`
   - fallback key: `(gwas_study_locus_id, gwas_variant_id, gene_symbol)`
6. Keep rows without QTL evidence and fill QTL numeric features with `0` (text lists as empty string).
7. Attach ClinVar/EVA labels (`label_positive`).
8. Attach gene embeddings.
9. Write final table in CSV and Parquet.

## Feature families currently produced

1. GWAS locus/variant metadata
- credible set/locus identifiers
- lead variant metadata
- per-variant posterior and credible-set flags

2. Geometry/distance features
- `variant_inside_gene`
- `dist_variant_to_gene_bp`, `dist_variant_to_gene_kb`
- `dist_variant_to_tss_bp`, `dist_variant_to_tss_kb`
- `dist_score_500kb_log`

3. QTL/colocalisation summary (from legacy variant-level evidence)
- `colocalisation_h4_max`, `colocalisation_h4_mean`, `colocalisation_h3_mean`
- `colocalisation_clpp_max`, `colocalisation_clpp_mean`
- `qtl_study_locus_count`, `qtl_study_count`, `tissue_count`
- `qtl_variant_posterior_probability_max`, `qtl_variant_posterior_probability_mean`
- `has_qtl_evidence`, `has_strong_coloc_h4`, `coloc_score`
- categorical context columns (`qtl_study_types`, `qtl_projects`, `qtl_tissues`, etc.)

4. Embeddings
- `has_gene_embedding`
- `gene_emb_0000 ... gene_emb_0767`

5. Label
- `clinvar_eva_score`
- `label_positive`

## Placeholder features (currently fixed to zero)

- `coding_score_sum_pip`
- `coding_variant_count`
- `expression_score`

## Outputs

Directory:

- `src/data/als_cs_gene_tables/`

Files:

- `GCST90027164_variant_gene_candidate_feature_table.csv`
- `GCST90027164_variant_gene_candidate_feature_table.parquet`
- `GCST90027164_raw_gwas_cs_locus_variants.csv`
- `GCST90027164_raw_gwas_cs_locus_variants.parquet`

## Run

```bash
/home/viguinijpv/python310/bin/python3.10 src/training/build_variant_gene_candidate_feature_table.py
```

## Current table summary (March 22, 2026)

- rows: `1270`
- unique GWAS loci: `13`
- unique GWAS variants: `136`
- unique genes: `178`
- rows with QTL evidence: `280`
- rows with embeddings: `344`
- positive rows (`ClinVar/EVA >= 0.5`): `22` (from `4` unique genes)

## Feature matrix example (real rows)

Columns shown below are a compact subset for readability.

```csv
gwas_study_locus_id,gwas_variant_id,gene_symbol,variant_inside_gene,dist_variant_to_gene_kb,dist_variant_to_tss_kb,colocalisation_h4_max,colocalisation_clpp_max,qtl_study_locus_count,tissue_count,has_gene_embedding,label_positive
18bb2fbb3a29d7bd192299aa0845ab99,12_57581917_C_T,KIF5A,1,0.0,35.891,0.0,0.0,0.0,0.0,1,1
8903002ec31eb992f90facfbac1f5e9a,9_27321049_T_C,C9ORF72,0,214.591,252.817,0.0,0.0,0.0,0.0,1,1
0c75178b4d87dc1ef949501b64cb1ef3,9_28077490_A_C,LINGO2,1,0.0,592.796,0.0,0.0,0.0,0.0,0,0
15dc87d740ec86e2f8a1db46ea134180,21_44333234_C_A,CFAP410,1,0.0,6.168,0.995948410448162,0.1874478867377733,1.0,1.0,0,0
15dc87d740ec86e2f8a1db46ea134180,21_44333234_C_A,PFKL,0,5.858,33.183,0.0,0.0,0.0,0.0,1,0
15dc87d740ec86e2f8a1db46ea134180,21_44333234_C_A,TRPM2,0,16.929,16.929,0.0,0.0,0.0,0.0,1,0
```

## Downstream logistic regression note

Recommended run:

```bash
/home/viguinijpv/python310/bin/python3.10 src/training/train_l1_logreg_clinvar_eva.py \
  --input-table src/data/als_cs_gene_tables/GCST90027164_variant_gene_candidate_feature_table.csv \
  --cv-group-by gene_id \
  --embedding-mode full
```

Important caveat for this dataset:

- only `4` positive genes exist
- with grouped 5-fold CV, only `2` folds have both classes
- the script now reports:
  - `n_splits_configured`
  - `n_splits_used`
  - `n_splits_skipped_single_class`

Latest grouped-CV summaries (variant-level table):

1. `embedding-mode none`
- `n_splits_used`: `2` (configured `5`, skipped `3`)
- ROC-AUC mean: `0.6955`
- AP mean: `0.1126`

2. `embedding-mode full`
- `n_splits_used`: `2` (configured `5`, skipped `3`)
- ROC-AUC mean: `0.9383`
- AP mean: `0.4591`

Outputs:

- `src/data/als_cs_gene_tables/l1_clinvar_eva_variant_candidate_grouped_gene_none/`
- `src/data/als_cs_gene_tables/l1_clinvar_eva_variant_candidate_grouped_gene_full/`


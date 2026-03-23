# Soil Shazam Data Collector

Build a high-quality, evaluation-ready soil image dataset from web sources with an end-to-end pipeline for collection, filtering, labeling, review, and export.

## Why Soil Shazam

- Scales image collection across multiple sources and queries
- Cleans noisy web data using resize, deduplication, and overlay/soil filtering
- Auto-labels 7 soil properties with CLIP-based prompts
- Supports human-in-the-loop review in the web app
- Produces evaluation metrics and exportable JSON/CSV artifacts

## What It Does

1. `download` images from configured sources
2. `resize` and remove low-quality size outliers
3. `dedup` perceptual duplicates
4. `filter` overlays/watermarks and non-soil images
5. `label` images across 7 soil categories
6. `eval-sample` + `annotate` in web UI for evaluation ground truth
7. `eval-report` to compute metrics
8. `cluster-review` to build cluster-assisted review queues
9. `export` final dataset files (optional)

## Label Categories

- `soil_color`
- `soil_texture`
- `particle_size`
- `crack_presence`
- `rock_fraction`
- `surface_structure`
- `surface_roughness`

## Quick Start

### 1) Install

```bash
pip install -e .
```

### 2) Check CLI

```bash
soil-shazam-data-collector --help
```

### 3) Run Full Pipeline To Labels

```bash
soil-shazam-data-collector run-all --limit 500
# Optional skips:
# soil-shazam-data-collector run-all --skip-resize --skip-dedup --skip-filter
```

### 4) Launch Web App

```bash
soil-shazam-data-collector webapp
```

Open `/` for dashboard, `/annotate` for evaluation review, and `/cluster` for clustering outputs.
The dashboard now includes a **Run Health / Precheck** panel that shows missing prerequisites and which command to run next.

## Step-by-Step Commands

```bash
soil-shazam-data-collector download --source all --limit 500
soil-shazam-data-collector resize
soil-shazam-data-collector dedup
soil-shazam-data-collector filter
soil-shazam-data-collector label
soil-shazam-data-collector eval-sample
soil-shazam-data-collector eval-report
soil-shazam-data-collector cluster-review
```

Recommended production flow:

`download -> resize -> deduplicate -> filter -> label -> eval -> cluster`

Step dependency behavior:

- `resize`, `dedup`, `filter`, and `label` all require available images (with automatic fallback to the best available upstream image output).
- `eval-sample` and `eval-report` require `label`.
- `cluster-review` requires `eval-report` outputs and `data/labeled/embeddings.npz`.

## Evaluation Flow

### Create Sample

```bash
soil-shazam-data-collector eval-sample
```

### Review in Web App (`/annotate`)

- Mark `is_soil` as yes/no
- If soil, confirm or correct all 7 class labels
- Save, skip, go back, and undo from the evaluation interface

### Generate Metrics

```bash
soil-shazam-data-collector eval-report
```

Generated outputs (under `evaluation/`):

- `sample.json`
- `metrics.json`
- `report.md`

### Build Cluster Review Queues

```bash
soil-shazam-data-collector cluster-review
```

Generated outputs (under `data/clustering/<run_id>/`):

- `clusters.json`
- `review_queue.json`
- `suggestions.json`
- `summary.json`

`cluster-review` is read-only in v1 and does not extract CLIP embeddings.
It reuses persisted label embeddings from:

- `data/labeled/embeddings.npz`
- `data/labeled/embeddings_meta.json`

## Configuration

Project settings live in `config/`:

- `pipeline.yaml`: paths, thresholds, processing settings
- `prompts.yaml`: CLIP prompts for filtering and class labeling
- `queries.yaml`: web search query sets

Storage behavior:

- `single_image_store: true` keeps canonical images in `data/deduped/images`.
- `filter` and `label` then write metadata/artifacts (logs, labels, embeddings) without copying images again.
- Unified step state is written to `data/deduped/pipeline_manifest.json`.

## Outputs

The pipeline writes processed datasets and exports to configured output paths in JSON and CSV formats.

## Run Comparison (March 8, 2026)

Two full runs were compared to measure query/source and threshold optimization impact.

### Setup

| Run | Configuration |
|-----|---------------|
| Run 1 (Baseline) | 20 queries, Bing only, limit 200/query, threshold 0.30 |
| Run 2 (Optimized) | 94 queries, Bing + Google + Flickr, limit 200/query, threshold 0.24 |

### Throughput and Yield

| Metric | Run 1 | Run 2 | Change |
|--------|-------|-------|--------|
| Raw images | 1,364 | 12,825 | +840% |
| Final dataset size | 255 | 2,392 | +838% |
| Yield rate | 18.7% | 18.6% | ~same |

### Class Coverage Improvements

| Label slice | Run 1 | Run 2 | Change |
|-------------|-------|-------|--------|
| `soil_texture=gravel` | 2.0% | 11.4% | +9.4pp |
| `crack_presence=high` | 0.0% | 9.7% | +9.7pp |
| `rock_fraction=high` | 0.4% | 10.8% | +10.4pp |
| `surface_roughness=smooth` | 5.0% | 15.6% | +10.6pp |

### Key Findings

- Multi-source collection scaled dataset size by ~9.4x while preserving yield quality.
- Major class-coverage gaps from Run 1 were resolved in Run 2.
- Color balance improved (less brown dominance, better dark/gray coverage).
- Dedup removed ~34% of resized images; overlay filter flagged ~17% in larger crawl.

### Remaining Gaps

- `particle_size=medium` remains relatively low (~8.7%)
- `soil_color=white` remains low (~2.6%)
- `crack_presence=none` remains dominant (~84.5%)

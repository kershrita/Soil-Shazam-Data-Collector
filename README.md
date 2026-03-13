# Soil Shazam Data Collector

Soil Shazam Data Collector is an end-to-end pipeline to build a high-quality soil image dataset from web sources.

It supports:
- large-scale image download
- quality filtering (resolution, overlay/watermark, soil vs non-soil)
- deduplication
- automatic label assignment across 7 soil categories
- human evaluation workflow and metrics reporting
- JSON/CSV export

## Pipeline

1. `download`: crawl images from configured sources and queries
2. `resize`: remove too-small images and resize oversized images
3. `dedup`: remove perceptual duplicates
4. `filter`: reject overlays/watermarks and non-soil images
5. `label`: assign class labels using CLIP prompts
6. `annotate` (web): human evaluation/ground-truth annotation
7. `export`: export final dataset files

## Label Categories

- `soil_color`
- `soil_texture`
- `particle_size`
- `crack_presence`
- `rock_fraction`
- `surface_structure`
- `surface_roughness`

## Installation

```bash
pip install -e .
```

## CLI

Primary command:

```bash
soil-shazam-data-collector --help
```

### Typical usage

Run full pipeline:

```bash
soil-shazam-data-collector run-all --limit 500
```

Run step-by-step:

```bash
soil-shazam-data-collector download --source all --limit 500
soil-shazam-data-collector resize
soil-shazam-data-collector dedup
soil-shazam-data-collector filter
soil-shazam-data-collector label
soil-shazam-data-collector export
```

Start web app:

```bash
soil-shazam-data-collector webapp
```

## Evaluation Workflow

Create evaluation sample:

```bash
soil-shazam-data-collector eval-sample
```

Then open `/annotate` in the web app to label samples:
- mark `is_soil` (yes/no)
- if soil, provide all class labels (keep predicted or correct per class)

Generate metrics and report:

```bash
soil-shazam-data-collector eval-report
```

Outputs are written under the `evaluation/` directory:
- `sample.json`
- `metrics.json`
- `report.md`

## Configuration

Configuration files are in `config/`:
- `pipeline.yaml`: paths, thresholds, processing settings
- `prompts.yaml`: CLIP prompts for filter/label behavior
- `queries.yaml`: search queries

## Output

Final dataset artifacts are exported to the configured dataset/output paths as JSON and CSV.

## Run Comparison Results (March 8, 2026)

Two full runs were compared to measure impact of query/source and threshold optimization.

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
- All major class coverage gaps from Run 1 were resolved in Run 2.
- Color balance improved significantly (less brown dominance, better dark/gray coverage).
- Dedup and overlay filtering behaved as expected at scale:
  - dedup removed ~34% of resized images
  - overlay filter flagged ~17% in the larger, noisier crawl

### Remaining Gaps to Improve

- `particle_size=medium` remains relatively low (~8.7%)
- `soil_color=white` remains low (~2.6%)
- `crack_presence=none` remains dominant (~84.5%)

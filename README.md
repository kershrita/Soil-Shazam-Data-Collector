# Soil Shazam Data Collector

Automated soil image dataset mining pipeline using CLIP. Downloads soil images from the internet, filters non-soil and watermarked images, auto-labels 7 soil features, deduplicates, and exports a clean labeled dataset.

## Pipeline Overview

```
Internet Images (Bing + Google + DuckDuckGo)
        ↓
  Mass Download (30k–50k images)
        ↓
  Resolution Filter (discard <384px, resize >1024px)
        ↓
  Deduplication (perceptual hashing)
        ↓
  Overlay + Soil Filtering (CLIP-based watermark/text + soil detection)
        ↓
  Auto-Labeling (7 features via CLIP)
        ↓
  Manual Verification (5-10% sample)
        ↓
  Clean Dataset (JSON + CSV)
```

## Features Labeled

| Feature | Classes |
|---------|---------|
| soil_color | red, yellow, dark, brown, gray, white |
| soil_texture | sandy, clay, silty, loamy, gravel |
| particle_size | fine, medium, coarse |
| crack_presence | high, moderate, none |
| rock_fraction | high, medium, low, none |
| surface_structure | compact, loose, aggregated |
| surface_roughness | rough, moderate, smooth |

## Quick Start

```bash
# Install
pip install -e .

# Run full pipeline (small test)
soil-collector run-all --limit 50

# Run full pipeline (production)
soil-collector run-all --limit 500

# Run individual steps
soil-collector download --source all --limit 500
soil-collector resize
soil-collector dedup
soil-collector filter
soil-collector label
soil-collector verify
soil-collector export
```

## Configuration

All configuration is in `config/`:

- **`queries.yaml`** — Search queries (add/remove to adjust coverage)
- **`prompts.yaml`** — CLIP prompts for filtering + labeling (tune for accuracy)
- **`pipeline.yaml`** — Thresholds, resolution, batch sizes, paths

Key settings in `pipeline.yaml`:

```yaml
resolution:
  min_shortest_side: 512    # Discard images smaller than this
  max_longest_side: 1024    # Resize larger images
  jpeg_quality: 95          # Output JPEG quality

filter:
  soil_threshold: 0.22      # CLIP similarity cutoff for soil detection
  overlay_margin: 0.07      # Overlay vs clean score margin (watermark/text)

clip:
  model_name: "ViT-L-14"   # Can downgrade to "ViT-B-32" for lower VRAM
  device: "cuda"            # Uses GPU, falls back to CPU
  batch_size: 128           # Tune based on VRAM (128 is fine for 16GB)
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (16GB VRAM recommended)
- ~50GB disk space for raw images

## Output Format

```json
{
  "image": "soil_00412.jpg",
  "soil_color": "brown",
  "soil_texture": "sandy",
  "particle_size": "coarse",
  "rock_fraction": "medium",
  "surface_structure": "loose",
  "crack_presence": "none",
  "surface_roughness": "moderate"
}
```

Final dataset is exported to `data/dataset/` as both `labels.json` and `labels.csv`.

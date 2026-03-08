"""CLI entry point for the soil image data collection pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
import yaml

from soil_collector.utils.logging import setup_logging

app = typer.Typer(
    name="soil-collector",
    help="Automated soil image dataset mining pipeline using CLIP.",
    add_completion=False,
)

# Root of project — locate config relative to this file
_PKG_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_ROOT.parent.parent  # src/../..


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _get_config(config_dir: Path | None = None) -> dict:
    """Load pipeline.yaml config."""
    config_dir = config_dir or _PROJECT_ROOT / "config"
    return _load_yaml(config_dir / "pipeline.yaml")


def _get_prompts(config_dir: Path | None = None) -> dict:
    config_dir = config_dir or _PROJECT_ROOT / "config"
    return _load_yaml(config_dir / "prompts.yaml")


def _get_queries(config_dir: Path | None = None) -> list[str]:
    config_dir = config_dir or _PROJECT_ROOT / "config"
    data = _load_yaml(config_dir / "queries.yaml")
    return data.get("queries", [])


def _resolve_path(cfg_path: str) -> Path:
    """Resolve a config-relative path to absolute."""
    p = Path(cfg_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


# ─── DOWNLOAD ────────────────────────────────────────────────────────────────


@app.command()
def download(
    source: str = typer.Option("all", help="Source: bing, google, flickr, or all"),
    limit: int = typer.Option(0, help="Max images per query per source (0 = use config default)"),
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Download soil images from internet sources."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    cfg = _get_config(config_dir)
    queries = _get_queries(config_dir)
    dl_cfg = cfg["download"]
    per_query_limit = limit if limit > 0 else dl_cfg["limit_per_query_per_source"]
    timeout = dl_cfg.get("timeout", 30)
    raw_dir = _resolve_path(cfg["paths"]["raw"])

    logger.info(f"Downloading images: {len(queries)} queries, limit={per_query_limit}/query/source")

    # Build downloaders
    from soil_collector.downloader.bing import BingDownloader
    from soil_collector.downloader.flickr import FlickrDownloader
    from soil_collector.downloader.google import GoogleDownloader

    downloaders = []
    if source in ("all", "bing"):
        downloaders.append(BingDownloader())
    if source in ("all", "google"):
        downloaders.append(GoogleDownloader())
    if source in ("all", "flickr"):
        downloaders.append(FlickrDownloader())

    total = 0
    for dl in downloaders:
        for query in queries:
            paths = dl.download(query, per_query_limit, raw_dir, timeout)
            total += len(paths)

    logger.info(f"Download complete: {total} total images in {raw_dir}")


# ─── RESIZE ──────────────────────────────────────────────────────────────────


@app.command()
def resize(
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Filter images by resolution and resize oversized ones."""
    setup_logging(log_level)

    cfg = _get_config(config_dir)
    res_cfg = cfg["resolution"]
    raw_dir = _resolve_path(cfg["paths"]["raw"])
    resized_dir = _resolve_path(cfg["paths"]["resized"])

    from soil_collector.filtering.resolution import run_resolution_filter

    run_resolution_filter(
        input_dir=raw_dir,
        output_dir=resized_dir,
        min_shortest_side=res_cfg["min_shortest_side"],
        max_longest_side=res_cfg["max_longest_side"],
        jpeg_quality=res_cfg["jpeg_quality"],
    )


# ─── FILTER ──────────────────────────────────────────────────────────────────


@app.command()
def filter(
    threshold: float = typer.Option(0, help="Soil similarity threshold (0 = use config)"),
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Filter images: remove watermarks and non-soil images using CLIP."""
    setup_logging(log_level)

    cfg = _get_config(config_dir)
    prompts = _get_prompts(config_dir)
    filter_cfg = cfg["filter"]
    clip_cfg = cfg["clip"]

    resized_dir = _resolve_path(cfg["paths"]["resized"])
    filtered_dir = _resolve_path(cfg["paths"]["filtered"])
    logs_dir = _resolve_path(cfg["paths"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    soil_threshold = threshold if threshold > 0 else filter_cfg["soil_threshold"]

    # Load CLIP model
    from soil_collector.utils.clip_model import get_clip_model

    clip_model = get_clip_model(
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        device=clip_cfg["device"],
        batch_size=clip_cfg["batch_size"],
    )

    # Stage 1: Watermark detection
    from soil_collector.filtering.watermark import run_watermark_filter

    watermarked = run_watermark_filter(
        input_dir=resized_dir,
        log_path=logs_dir / "watermark_filter.csv",
        clip_model=clip_model,
        watermark_prompts=prompts["filter_watermark"]["watermark"],
        clean_prompts=prompts["filter_watermark"]["clean"],
        margin=filter_cfg["watermark_margin"],
    )

    # Stage 2: Soil filtering
    from soil_collector.filtering.clip_filter import run_clip_filter

    run_clip_filter(
        input_dir=resized_dir,
        output_dir=filtered_dir,
        log_path=logs_dir / "soil_filter.csv",
        clip_model=clip_model,
        positive_prompts=prompts["filter_soil"]["positive"],
        negative_prompts=prompts["filter_soil"]["negative"],
        threshold=soil_threshold,
        watermarked_stems=watermarked,
    )


# ─── LABEL ───────────────────────────────────────────────────────────────────


@app.command()
def label(
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Auto-label filtered soil images using CLIP similarity scoring."""
    setup_logging(log_level)

    cfg = _get_config(config_dir)
    prompts = _get_prompts(config_dir)
    clip_cfg = cfg["clip"]

    filtered_dir = _resolve_path(cfg["paths"]["filtered"])
    labeled_dir = _resolve_path(cfg["paths"]["labeled"])

    from soil_collector.utils.clip_model import get_clip_model

    clip_model = get_clip_model(
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        device=clip_cfg["device"],
        batch_size=clip_cfg["batch_size"],
    )

    from soil_collector.labeling.clip_labeler import run_clip_labeling

    run_clip_labeling(
        input_dir=filtered_dir,
        output_dir=labeled_dir,
        clip_model=clip_model,
        label_prompts=prompts["labeling"],
    )


# ─── DEDUP ───────────────────────────────────────────────────────────────────


@app.command()
def dedup(
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Deduplicate images: perceptual hashing → CLIP embeddings."""
    setup_logging(log_level)

    cfg = _get_config(config_dir)
    clip_cfg = cfg["clip"]
    dedup_cfg = cfg["dedup"]

    labeled_dir = _resolve_path(cfg["paths"]["labeled"])
    deduped_dir = _resolve_path(cfg["paths"]["deduped"])

    from soil_collector.utils.clip_model import get_clip_model

    clip_model = get_clip_model(
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        device=clip_cfg["device"],
        batch_size=clip_cfg["batch_size"],
    )

    from soil_collector.dedup.deduplicator import run_deduplication

    run_deduplication(
        input_dir=labeled_dir,
        labels_path=labeled_dir / "labels.json",
        output_dir=deduped_dir,
        clip_model=clip_model,
        phash_threshold=dedup_cfg["phash_threshold"],
        clip_cosine_threshold=dedup_cfg["clip_cosine_threshold"],
    )


# ─── VERIFY ──────────────────────────────────────────────────────────────────


@app.command()
def verify(
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Generate verification report for manual review of a random sample."""
    setup_logging(log_level)

    cfg = _get_config(config_dir)
    deduped_dir = _resolve_path(cfg["paths"]["deduped"])
    dataset_dir = _resolve_path(cfg["paths"]["dataset"])
    sample_frac = cfg["verification"]["sample_fraction"]

    from soil_collector.verification.sampler import run_verification_sampling

    report_path = run_verification_sampling(
        input_dir=deduped_dir,
        output_dir=dataset_dir / "verification",
        sample_fraction=sample_frac,
    )
    typer.echo(f"Verification report: {report_path}")
    typer.echo("Edit corrections.json to fix mislabeled entries, then run 'export'.")


# ─── EXPORT ──────────────────────────────────────────────────────────────────


@app.command()
def export(
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Export the final clean dataset as JSON + CSV."""
    setup_logging(log_level)

    cfg = _get_config(config_dir)
    deduped_dir = _resolve_path(cfg["paths"]["deduped"])
    dataset_dir = _resolve_path(cfg["paths"]["dataset"])
    corrections_path = dataset_dir / "verification" / "corrections.json"

    from soil_collector.export.dataset import run_export

    run_export(
        input_dir=deduped_dir,
        corrections_path=corrections_path if corrections_path.exists() else None,
        output_dir=dataset_dir,
    )


# ─── RUN ALL ─────────────────────────────────────────────────────────────────


@app.command(name="run-all")
def run_all(
    source: str = typer.Option("all", help="Download source: bing, google, ddg, or all"),
    limit: int = typer.Option(0, help="Max images per query per source (0 = use config)"),
    threshold: float = typer.Option(0, help="Soil filter threshold (0 = use config)"),
    skip_download: bool = typer.Option(False, help="Skip download step (use existing raw images)"),
    skip_verify: bool = typer.Option(False, help="Skip verification step"),
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Run the complete pipeline end-to-end."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("SOIL IMAGE DATA COLLECTOR — FULL PIPELINE")
    logger.info("=" * 60)

    # Step 1: Download
    if not skip_download:
        logger.info("\n>>> STEP 1/6: Downloading images...")
        download(source=source, limit=limit, config_dir=config_dir, log_level=log_level)
    else:
        logger.info("\n>>> STEP 1/6: Download SKIPPED")

    # Step 2: Resize
    logger.info("\n>>> STEP 2/6: Resolution filter + resize...")
    resize(config_dir=config_dir, log_level=log_level)

    # Step 3: Filter
    logger.info("\n>>> STEP 3/6: Watermark + soil filtering...")
    filter(threshold=threshold, config_dir=config_dir, log_level=log_level)

    # Step 4: Label
    logger.info("\n>>> STEP 4/6: CLIP feature labeling...")
    label(config_dir=config_dir, log_level=log_level)

    # Step 5: Dedup
    logger.info("\n>>> STEP 5/6: Two-stage deduplication...")
    dedup(config_dir=config_dir, log_level=log_level)

    # Step 6: Export (optionally with verify)
    if not skip_verify:
        logger.info("\n>>> STEP 6/6: Verification + export...")
        verify(config_dir=config_dir, log_level=log_level)
    else:
        logger.info("\n>>> STEP 6/6: Export (skipping verification)...")

    export(config_dir=config_dir, log_level=log_level)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()

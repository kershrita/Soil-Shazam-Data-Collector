"""CLI entry point for the soil image data collection pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
import yaml

from .utils import setup_logging

app = typer.Typer(
    name="soil-collector",
    help="Automated soil image dataset mining pipeline using CLIP.",
    add_completion=False,
)

# Root of project — locate config relative to this file
_PKG_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_ROOT.parent  # src → project root


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


def _get_workers(cfg: dict) -> int:
    """Return shared worker count from config (0 → os.cpu_count)."""
    import os
    w = cfg.get("num_workers", 0)
    return w if w > 0 else (os.cpu_count() or 4)
    return p


# ─── DOWNLOAD ────────────────────────────────────────────────────────────────


@app.command()
def download(
    source: str = typer.Option("all", help="Source: bing, google, flickr, or all"),
    limit: int = typer.Option(0, help="Max images per query per source (0 = use config default)"),
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Download soil images from internet sources (parallel)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    cfg = _get_config(config_dir)
    queries = _get_queries(config_dir)
    dl_cfg = cfg["download"]
    workers = dl_cfg.get("workers_per_source", 4)
    request_delay = dl_cfg.get("request_delay", 1.0)
    per_query_limit = limit if limit > 0 else dl_cfg["limit_per_query_per_source"]
    timeout = dl_cfg.get("timeout", 30)
    raw_dir = _resolve_path(cfg["paths"]["raw"])

    # Build downloaders
    from .downloader import BingDownloader, FlickrDownloader, GoogleDownloader

    downloaders = []
    if source in ("all", "bing"):
        downloaders.append(BingDownloader())
    if source in ("all", "google"):
        downloaders.append(GoogleDownloader())
    if source in ("all", "flickr"):
        downloaders.append(FlickrDownloader())

    logger.info(
        f"Downloading images: {len(queries)} queries × {len(downloaders)} sources, "
        f"limit={per_query_limit}/query/source, workers={workers}/source"
    )

    # Thread-safe counter
    _lock = threading.Lock()
    totals: dict[str, int] = {}

    def _run_source(dl):
        """Run all queries for one source using a thread pool."""
        import time
        source_name = dl.source_name
        source_total = 0

        def _download_query(query: str) -> int:
            paths = dl.download(query, per_query_limit, raw_dir, timeout)
            time.sleep(request_delay)  # rate-limit between queries
            return len(paths)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=source_name) as pool:
            futures = {pool.submit(_download_query, q): q for q in queries}
            for future in as_completed(futures):
                try:
                    count = future.result()
                    source_total += count
                except Exception as e:
                    query = futures[future]
                    logger.error(f"[{source_name}] Query '{query}' failed: {e}")

        with _lock:
            totals[source_name] = source_total
        logger.info(f"[{source_name}] Source complete: {source_total} images")

    # Run all sources in parallel (each source gets its own thread)
    source_threads = []
    for dl in downloaders:
        t = threading.Thread(target=_run_source, args=(dl,), name=f"source-{dl.source_name}")
        t.start()
        source_threads.append(t)

    for t in source_threads:
        t.join()

    grand_total = sum(totals.values())
    breakdown = ", ".join(f"{k}={v}" for k, v in totals.items())
    logger.info(f"Download complete: {grand_total} total images ({breakdown}) in {raw_dir}")


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
    workers = _get_workers(cfg)
    raw_dir = _resolve_path(cfg["paths"]["raw"])
    resized_dir = _resolve_path(cfg["paths"]["resized"])

    from .filtering import run_resolution_filter

    run_resolution_filter(
        input_dir=raw_dir,
        output_dir=resized_dir,
        min_shortest_side=res_cfg["min_shortest_side"],
        max_longest_side=res_cfg["max_longest_side"],
        jpeg_quality=res_cfg["jpeg_quality"],
        workers=workers,
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

    deduped_dir = _resolve_path(cfg["paths"]["deduped"])
    filtered_dir = _resolve_path(cfg["paths"]["filtered"])
    logs_dir = _resolve_path(cfg["paths"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    soil_threshold = threshold if threshold > 0 else filter_cfg["soil_threshold"]

    # Load CLIP model
    from .utils import get_clip_model

    clip_model = get_clip_model(
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        device=clip_cfg["device"],
        batch_size=clip_cfg["batch_size"],
    )

    # Stage 1: Overlay detection (watermarks + text)
    from .filtering import run_overlay_filter

    overlay_stems = run_overlay_filter(
        input_dir=deduped_dir,
        log_path=logs_dir / "overlay_filter.csv",
        clip_model=clip_model,
        overlay_prompts=prompts["filter_overlay"]["overlay"],
        clean_prompts=prompts["filter_overlay"]["clean"],
        overlay_margin=filter_cfg["overlay_margin"],
    )

    # Stage 2: Soil filtering
    from .filtering import run_clip_filter

    run_clip_filter(
        input_dir=deduped_dir,
        output_dir=filtered_dir,
        log_path=logs_dir / "soil_filter.csv",
        clip_model=clip_model,
        positive_prompts=prompts["filter_soil"]["positive"],
        negative_prompts=prompts["filter_soil"]["negative"],
        threshold=soil_threshold,
        flagged_stems=overlay_stems,
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

    from .utils import get_clip_model

    clip_model = get_clip_model(
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        device=clip_cfg["device"],
        batch_size=clip_cfg["batch_size"],
    )

    from .labeling import run_clip_labeling

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
    """Deduplicate images using perceptual hashing."""
    setup_logging(log_level)

    cfg = _get_config(config_dir)
    dedup_cfg = cfg["dedup"]

    resized_dir = _resolve_path(cfg["paths"]["resized"])
    deduped_dir = _resolve_path(cfg["paths"]["deduped"])

    from .dedup import run_deduplication

    run_deduplication(
        input_dir=resized_dir,
        output_dir=deduped_dir,
        phash_threshold=dedup_cfg["phash_threshold"],
        num_workers=_get_workers(cfg),
    )


# ─── EXPORT ──────────────────────────────────────────────────────────────────


@app.command()
def export(
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Export the final clean dataset as JSON + CSV."""
    setup_logging(log_level)

    cfg = _get_config(config_dir)
    labeled_dir = _resolve_path(cfg["paths"]["labeled"])
    dataset_dir = _resolve_path(cfg["paths"]["dataset"])
    corrections_path = dataset_dir / "verification" / "corrections.json"

    from .export import run_export

    run_export(
        input_dir=labeled_dir,
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
        logger.info("\n>>> STEP 1/5: Downloading images...")
        download(source=source, limit=limit, config_dir=config_dir, log_level=log_level)
    else:
        logger.info("\n>>> STEP 1/5: Download SKIPPED")

    # Step 2: Resize
    logger.info("\n>>> STEP 2/5: Resolution filter + resize...")
    resize(config_dir=config_dir, log_level=log_level)

    # Step 3: Dedup
    logger.info("\n>>> STEP 3/5: Perceptual hash deduplication...")
    dedup(config_dir=config_dir, log_level=log_level)

    # Step 4: Filter
    logger.info("\n>>> STEP 4/5: Overlay + soil filtering...")
    filter(threshold=threshold, config_dir=config_dir, log_level=log_level)

    # Step 5: Label + Export
    logger.info("\n>>> STEP 5/5: CLIP feature labeling + export...")
    label(config_dir=config_dir, log_level=log_level)
    export(config_dir=config_dir, log_level=log_level)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


# ─── EVALUATION ──────────────────────────────────────────────────────────────


@app.command(name="eval-sample")
def eval_sample(
    n: int = typer.Option(100, help="Number of accepted images to sample"),
    n_rejected: int = typer.Option(30, help="Number of rejected images to sample"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Sample random images from the dataset for accuracy evaluation."""
    setup_logging(log_level)

    cfg = _get_config(config_dir)
    dataset_dir = _resolve_path(cfg["paths"]["dataset"])
    eval_dir = dataset_dir.parent / "evaluation"

    from .evaluation import create_eval_sample

    sample_path = create_eval_sample(
        dataset_dir=dataset_dir,
        output_dir=eval_dir,
        n_accepted=n,
        n_rejected=n_rejected,
        seed=seed,
    )
    typer.echo(f"\nSample created: {sample_path}")
    typer.echo("Next: run 'soil-collector webapp' and go to /annotate to label the sample.")


@app.command(name="eval-report")
def eval_report(
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Compute accuracy metrics and generate a shareable report."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    cfg = _get_config(config_dir)
    eval_dir = _resolve_path(cfg["paths"]["dataset"]).parent / "evaluation"

    from .evaluation import compute_metrics, generate_report

    metrics = compute_metrics(eval_dir)
    if not metrics:
        typer.echo("No annotated samples found. Annotate via the webapp first.")
        raise typer.Exit(1)

    report_path = generate_report(eval_dir)

    # Print summary
    summary = metrics.get("summary", {})
    sample = metrics.get("sample_size", {})
    filt = metrics.get("filter", {})

    typer.echo("\n" + "=" * 60)
    typer.echo("EVALUATION RESULTS")
    typer.echo("=" * 60)
    typer.echo(f"Images annotated:       {sample.get('total_annotated', 0)}")
    typer.echo(f"Filter precision:       {_fmt_pct(filt.get('precision'))}")
    typer.echo(f"Filter recall:          {_fmt_pct(filt.get('recall'))}")
    typer.echo(f"Overall label accuracy: {_fmt_pct(summary.get('overall_label_accuracy'))}")

    ranking = summary.get("category_ranking", [])
    if ranking:
        typer.echo("\nPer-category accuracy:")
        for cat, acc in ranking:
            typer.echo(f"  {cat:25s} {_fmt_pct(acc)}")

    typer.echo(f"\nFull report: {report_path}")
    typer.echo(f"Raw metrics: {eval_dir / 'metrics.json'}")


def _fmt_pct(value) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


# ─── VALIDATE (web UI) ──────────────────────────────────────────────────────


@app.command()
def webapp(
    port: int = typer.Option(5000, help="Port to run the validation server on"),
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
):
    """Launch the validation web UI to browse and correct pipeline outputs."""
    from .webapp import create_app

    web_app = create_app(config_dir=config_dir)
    typer.echo(f"Starting validation UI at http://{host}:{port}")
    web_app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    app()

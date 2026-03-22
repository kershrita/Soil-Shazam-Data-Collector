"""CLI entry point for the soil image data collection pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
import yaml

from utils import collect_image_paths, setup_logging

app = typer.Typer(
    name="soil-shazam-data-collector",
    help="Soil Shazam Data Collector: automated soil image dataset mining pipeline using CLIP.",
    add_completion=False,
)

RECOMMENDED_FLOW = "download -> resize -> deduplicate -> filter -> label -> eval -> cluster"


class DownloadSource(str, Enum):
    all = "all"
    bing = "bing"
    google = "google"
    flickr = "flickr"


# Root of project; locate config relative to this file
_PKG_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_ROOT.parent


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _get_config(config_dir: Path | None = None) -> dict:
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
    p = Path(cfg_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


def _get_workers(cfg: dict) -> int:
    import os

    w = cfg.get("num_workers", 0)
    return w if w > 0 else (os.cpu_count() or 4)


def _image_count(directory: Path) -> int:
    if not directory.exists():
        return 0
    return len(collect_image_paths(directory))


def _find_first_input(
    candidates: list[tuple[str, Path]],
) -> tuple[str, Path, int] | None:
    for label, path in candidates:
        count = _image_count(path)
        if count > 0:
            return label, path, count
    return None


def _labels_file(base_dir: Path) -> Path | None:
    labels_full = base_dir / "labels_full.json"
    labels_basic = base_dir / "labels.json"
    if labels_full.exists():
        return labels_full
    if labels_basic.exists():
        return labels_basic
    return None


def _exit_with_prereq(
    message: str,
    run_first: str | None = None,
    exit_code: int = 1,
) -> None:
    typer.echo(f"Error: {message}")
    if run_first:
        typer.echo(f"Run this first: {run_first}")
    typer.echo(f"Recommended flow: {RECOMMENDED_FLOW}")
    raise typer.Exit(exit_code)


def _fmt_pct(value) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


@app.command()
def download(
    source: DownloadSource = typer.Option(DownloadSource.all, help="Source: bing, google, flickr, or all"),
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
    max_retries = max(0, int(dl_cfg.get("max_retries", 0)))
    raw_dir = _resolve_path(cfg["paths"]["raw"])

    from downloader import BingDownloader, FlickrDownloader, GoogleDownloader

    downloaders = []
    if source in (DownloadSource.all, DownloadSource.bing):
        downloaders.append(BingDownloader())
    if source in (DownloadSource.all, DownloadSource.google):
        downloaders.append(GoogleDownloader())
    if source in (DownloadSource.all, DownloadSource.flickr):
        downloaders.append(FlickrDownloader())

    if not downloaders:
        logger.error("No downloaders selected for source=%s", source.value)
        raise typer.Exit(2)

    logger.info(
        "Downloading images: %s queries x %s sources, limit=%s/query/source, workers=%s/source",
        len(queries),
        len(downloaders),
        per_query_limit,
        workers,
    )

    lock = threading.Lock()
    totals: dict[str, int] = {}

    def run_source(dl):
        import time

        source_name = dl.source_name
        source_total = 0

        def download_query(query: str) -> int:
            paths = dl.download(
                query,
                per_query_limit,
                raw_dir,
                timeout,
                max_retries=max_retries,
            )
            time.sleep(request_delay)
            return len(paths)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=source_name) as pool:
            futures = {pool.submit(download_query, q): q for q in queries}
            for future in as_completed(futures):
                try:
                    source_total += future.result()
                except Exception as err:  # noqa: BLE001
                    query = futures[future]
                    logger.error("[%s] Query '%s' failed: %s", source_name, query, err)

        with lock:
            totals[source_name] = source_total
        logger.info("[%s] Source complete: %s images", source_name, source_total)

    threads = []
    for dl in downloaders:
        t = threading.Thread(target=run_source, args=(dl,), name=f"source-{dl.source_name}")
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    grand_total = sum(totals.values())
    breakdown = ", ".join(f"{k}={v}" for k, v in totals.items())
    logger.info("Download complete: %s total images (%s) in %s", grand_total, breakdown, raw_dir)


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

    raw_count = _image_count(raw_dir)
    if raw_count == 0:
        _exit_with_prereq(
            message=f"No downloaded images found in {raw_dir}.",
            run_first="soil-shazam-data-collector download",
        )

    from filtering import run_resolution_filter

    run_resolution_filter(
        input_dir=raw_dir,
        output_dir=resized_dir,
        min_shortest_side=res_cfg["min_shortest_side"],
        max_longest_side=res_cfg["max_longest_side"],
        resize_mode=res_cfg.get("resize_mode", "shortest_side"),
        jpeg_quality=res_cfg["jpeg_quality"],
        workers=workers,
    )


@app.command()
def dedup(
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Deduplicate images using perceptual hashing."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    cfg = _get_config(config_dir)
    dedup_cfg = cfg["dedup"]

    raw_dir = _resolve_path(cfg["paths"]["raw"])
    resized_dir = _resolve_path(cfg["paths"]["resized"])
    deduped_dir = _resolve_path(cfg["paths"]["deduped"])

    selected = _find_first_input(
        [
            ("resize", resized_dir),
            ("download", raw_dir),
        ]
    )
    if not selected:
        _exit_with_prereq(
            message="No images available for deduplication.",
            run_first="soil-shazam-data-collector download",
        )

    source_step, source_dir, source_count = selected
    if source_step != "resize":
        logger.warning(
            "dedup: using %s images (%s) because resized output is missing",
            source_step,
            source_dir,
        )
    logger.info("dedup: input source=%s count=%s", source_step, source_count)

    from dedup import run_deduplication

    run_deduplication(
        input_dir=source_dir,
        output_dir=deduped_dir,
        phash_threshold=dedup_cfg["phash_threshold"],
        num_workers=_get_workers(cfg),
    )


@app.command()
def filter(
    threshold: float = typer.Option(0, help="Soil similarity threshold (0 = use config)"),
    resume: bool = typer.Option(
        False,
        help="Resume from existing filtered outputs instead of recomputing from scratch.",
    ),
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Filter images: remove watermarks and non-soil images using CLIP."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    cfg = _get_config(config_dir)
    prompts = _get_prompts(config_dir)
    filter_cfg = cfg["filter"]
    clip_cfg = cfg["clip"]

    raw_dir = _resolve_path(cfg["paths"]["raw"])
    resized_dir = _resolve_path(cfg["paths"]["resized"])
    deduped_dir = _resolve_path(cfg["paths"]["deduped"])
    filtered_dir = _resolve_path(cfg["paths"]["filtered"])
    logs_dir = _resolve_path(cfg["paths"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    selected = _find_first_input(
        [
            ("dedup", deduped_dir),
            ("resize", resized_dir),
            ("download", raw_dir),
        ]
    )
    if not selected:
        _exit_with_prereq(
            message="No images available for filter.",
            run_first="soil-shazam-data-collector download",
        )

    source_step, source_dir, source_count = selected
    if source_step != "dedup":
        logger.warning(
            "filter: using %s images (%s) because dedup output is missing",
            source_step,
            source_dir,
        )
    logger.info("filter: input source=%s count=%s", source_step, source_count)

    if not resume and filtered_dir.exists():
        import shutil

        shutil.rmtree(filtered_dir)

    soil_threshold = threshold if threshold > 0 else filter_cfg["soil_threshold"]

    from utils import get_clip_model

    clip_model = get_clip_model(
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        device=clip_cfg["device"],
        batch_size=clip_cfg["batch_size"],
    )

    from filtering import run_clip_filter, run_overlay_filter

    flagged_names = run_overlay_filter(
        input_dir=source_dir,
        log_path=logs_dir / "overlay_filter.csv",
        clip_model=clip_model,
        overlay_prompts=prompts["filter_overlay"]["overlay"],
        clean_prompts=prompts["filter_overlay"]["clean"],
        overlay_margin=filter_cfg["overlay_margin"],
    )

    run_clip_filter(
        input_dir=source_dir,
        output_dir=filtered_dir,
        log_path=logs_dir / "soil_filter.csv",
        clip_model=clip_model,
        positive_prompts=prompts["filter_soil"]["positive"],
        negative_prompts=prompts["filter_soil"]["negative"],
        threshold=soil_threshold,
        flagged_names=flagged_names,
        resume=resume,
    )

    run_cfg_path = logs_dir / "filter_run_config.json"
    run_cfg = {
        "soil_threshold": float(soil_threshold),
        "overlay_margin": float(filter_cfg["overlay_margin"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    run_cfg_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")


@app.command()
def label(
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Auto-label images using CLIP similarity scoring."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    cfg = _get_config(config_dir)
    prompts = _get_prompts(config_dir)
    clip_cfg = cfg["clip"]

    raw_dir = _resolve_path(cfg["paths"]["raw"])
    resized_dir = _resolve_path(cfg["paths"]["resized"])
    deduped_dir = _resolve_path(cfg["paths"]["deduped"])
    filtered_dir = _resolve_path(cfg["paths"]["filtered"])
    labeled_dir = _resolve_path(cfg["paths"]["labeled"])

    selected = _find_first_input(
        [
            ("filter", filtered_dir),
            ("dedup", deduped_dir),
            ("resize", resized_dir),
            ("download", raw_dir),
        ]
    )
    if not selected:
        _exit_with_prereq(
            message="No images available for label.",
            run_first="soil-shazam-data-collector download",
        )

    source_step, source_dir, source_count = selected
    if source_step != "filter":
        logger.warning(
            "label: using %s images (%s) because filtered output is missing",
            source_step,
            source_dir,
        )
    logger.info("label: input source=%s count=%s", source_step, source_count)

    from utils import get_clip_model

    clip_model = get_clip_model(
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        device=clip_cfg["device"],
        batch_size=clip_cfg["batch_size"],
    )

    from labeling import run_clip_labeling

    run_clip_labeling(
        input_dir=source_dir,
        output_dir=labeled_dir,
        clip_model=clip_model,
        label_prompts=prompts["labeling"],
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        persist_embeddings=True,
    )


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

    if _labels_file(labeled_dir) is None:
        _exit_with_prereq(
            message=f"Missing labeled outputs in {labeled_dir}.",
            run_first="soil-shazam-data-collector label",
        )

    from export import run_export

    run_export(
        input_dir=labeled_dir,
        corrections_path=corrections_path if corrections_path.exists() else None,
        output_dir=dataset_dir,
    )


@app.command(name="run-all")
def run_all(
    source: DownloadSource = typer.Option(
        DownloadSource.all,
        help="Download source: bing, google, flickr, or all",
    ),
    limit: int = typer.Option(0, help="Max images per query per source (0 = use config)"),
    threshold: float = typer.Option(0, help="Soil filter threshold (0 = use config)"),
    skip_download: bool = typer.Option(False, help="Skip download step (use existing raw images)"),
    skip_resize: bool = typer.Option(False, help="Skip resize step"),
    skip_dedup: bool = typer.Option(False, help="Skip dedup step"),
    skip_filter: bool = typer.Option(False, help="Skip filter step"),
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Run pipeline to labeled outputs with optional step skipping."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 64)
    logger.info("SOIL IMAGE DATA COLLECTOR - RUN ALL")
    logger.info("Recommended flow: %s", RECOMMENDED_FLOW)
    logger.info("=" * 64)

    if not skip_download:
        logger.info("Step: download")
        download(source=source, limit=limit, config_dir=config_dir, log_level=log_level)
    else:
        logger.info("Step skipped: download")

    if not skip_resize:
        logger.info("Step: resize")
        resize(config_dir=config_dir, log_level=log_level)
    else:
        logger.info("Step skipped: resize")

    if not skip_dedup:
        logger.info("Step: dedup")
        dedup(config_dir=config_dir, log_level=log_level)
    else:
        logger.info("Step skipped: dedup")

    if not skip_filter:
        logger.info("Step: filter")
        filter(threshold=threshold, resume=False, config_dir=config_dir, log_level=log_level)
    else:
        logger.info("Step skipped: filter")

    logger.info("Step: label")
    label(config_dir=config_dir, log_level=log_level)

    logger.info("=" * 64)
    logger.info("RUN-ALL COMPLETE")
    logger.info("=" * 64)


@app.command(name="eval-sample")
def eval_sample(
    n: int = typer.Option(100, help="Number of accepted images to sample"),
    n_rejected: int = typer.Option(30, help="Number of rejected images to sample"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Sample random images from label outputs for accuracy evaluation."""
    setup_logging(log_level)

    cfg = _get_config(config_dir)
    labeled_dir = _resolve_path(cfg["paths"]["labeled"])
    eval_dir = labeled_dir.parent / "evaluation"

    if _labels_file(labeled_dir) is None:
        _exit_with_prereq(
            message=f"No label outputs found in {labeled_dir}.",
            run_first="soil-shazam-data-collector label",
        )

    from evaluation import create_eval_sample

    sample_path = create_eval_sample(
        dataset_dir=labeled_dir,
        output_dir=eval_dir,
        n_accepted=n,
        n_rejected=n_rejected,
        seed=seed,
    )
    typer.echo(f"\nSample created: {sample_path}")
    typer.echo("Next: run 'soil-shazam-data-collector webapp' and open /annotate.")


@app.command(name="eval-report")
def eval_report(
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Compute accuracy metrics and generate a shareable report."""
    setup_logging(log_level)

    cfg = _get_config(config_dir)
    labeled_dir = _resolve_path(cfg["paths"]["labeled"])
    eval_dir = labeled_dir.parent / "evaluation"
    sample_path = eval_dir / "sample.json"

    if _labels_file(labeled_dir) is None:
        _exit_with_prereq(
            message="eval-report requires labels from the label step.",
            run_first="soil-shazam-data-collector label",
        )
    if not sample_path.exists():
        _exit_with_prereq(
            message=f"Missing evaluation sample at {sample_path}.",
            run_first="soil-shazam-data-collector eval-sample",
        )

    from evaluation import compute_metrics, generate_report

    metrics = compute_metrics(eval_dir)
    if not metrics:
        typer.echo("No annotated samples found. Annotate the eval sample in the web app first.")
        raise typer.Exit(1)

    report_path = generate_report(eval_dir)

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


@app.command(name="cluster-review")
def cluster_review(
    max_images: int = typer.Option(
        0,
        help="Optional cap for accepted images to process (0 = all).",
    ),
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Build cluster-assisted review queues and conservative label suggestions."""
    setup_logging(log_level)

    cfg = _get_config(config_dir)
    clustering_cfg = cfg.get("clustering", {})

    labeled_dir = _resolve_path(cfg["paths"]["labeled"])
    data_root = labeled_dir.parent
    eval_dir = data_root / "evaluation"
    clustering_root = data_root / "clustering"

    labels_path = _labels_file(labeled_dir)
    if labels_path is None:
        _exit_with_prereq(
            message=f"cluster-review requires labels from {labeled_dir}.",
            run_first="soil-shazam-data-collector label",
        )

    embeddings_path = labeled_dir / "embeddings.npz"
    if not embeddings_path.exists():
        _exit_with_prereq(
            message=(
                "cluster-review requires persisted label embeddings at "
                f"{embeddings_path}."
            ),
            run_first="soil-shazam-data-collector label",
        )

    sample_path = eval_dir / "sample.json"
    metrics_path = eval_dir / "metrics.json"
    if not sample_path.exists():
        _exit_with_prereq(
            message=f"cluster-review requires evaluation sample at {sample_path}.",
            run_first="soil-shazam-data-collector eval-sample",
        )
    if not metrics_path.exists():
        _exit_with_prereq(
            message=f"cluster-review requires evaluation metrics at {metrics_path}.",
            run_first="soil-shazam-data-collector eval-report",
        )

    from clustering import run_cluster_review

    summary = run_cluster_review(
        labeled_dir=labeled_dir,
        eval_dir=eval_dir,
        output_root=clustering_root,
        cluster_cfg=clustering_cfg,
        max_images=max_images,
    )

    counts = summary.get("counts", {})
    quality = summary.get("quality_controls", {})
    artifacts = summary.get("artifacts", {})

    typer.echo("\n" + "=" * 60)
    typer.echo("CLUSTER REVIEW COMPLETE")
    typer.echo("=" * 60)
    typer.echo(f"Accepted images processed: {counts.get('accepted_images', 0)}")
    typer.echo(f"Clusters built:            {counts.get('clusters', 0)}")
    typer.echo(f"Review queue items:        {counts.get('review_queue_items', 0)}")
    typer.echo(f"Suggestion items:          {counts.get('suggestion_items', 0)}")
    typer.echo(f"Suggested label slots:     {counts.get('suggested_category_slots', 0)}")
    typer.echo(f"Source labels unchanged:   {quality.get('source_labels_unchanged')}")

    concentration = (quality.get("priority_concentration") or {}).get("concentration_factor")
    if concentration is not None:
        typer.echo(f"Top-slice concentration:   {concentration:.3f}x")

    typer.echo(f"\nRun directory:  {artifacts.get('run_dir')}")
    typer.echo(f"clusters.json:  {artifacts.get('clusters')}")
    typer.echo(f"review_queue:   {artifacts.get('review_queue')}")
    typer.echo(f"suggestions:    {artifacts.get('suggestions')}")
    typer.echo(f"summary:        {artifacts.get('summary')}")


@app.command()
def webapp(
    port: int = typer.Option(5000, help="Port to run the Soil Shazam Data Collector web app on"),
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    debug: bool = typer.Option(False, help="Enable Flask debug mode (development only)."),
    config_dir: Optional[Path] = typer.Option(None, help="Path to config directory"),
):
    """Launch the Soil Shazam Data Collector web app."""
    from webapp import create_app

    web_app = create_app(config_dir=config_dir)
    typer.echo(f"Starting Soil Shazam Data Collector at http://{host}:{port}")
    web_app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    app()

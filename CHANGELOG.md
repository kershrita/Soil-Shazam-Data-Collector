# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- Resolved package/runtime import issues by replacing fragile parent-relative imports in pipeline modules with stable absolute imports.
- Fixed Windows dedup runtime failure by removing multiprocessing dependency in duplicate retrieval and using a single-process matching path.
- Fixed evaluation metric edge case where `recall=None` could raise during F1 computation.
- Fixed export edge case for empty labels to produce valid empty artifacts instead of crashing.
- Fixed web app debug safety by defaulting CLI web launch to `debug=False` and exposing an explicit `--debug` flag.
- Fixed CLI source help/validation drift (`ddg` mismatch) and added stricter source handling via enum options.
- Fixed filter rerun/resume behavior by adding explicit resume handling and consistent existing-output checks.
- Fixed dedup rerun behavior by synchronizing output to current surviving files and removing stale artifacts.
- Fixed API pagination guardrails by bounding `/api/images` `per_page`.
- Fixed retry/config drift by wiring `download.max_retries` into downloader implementations.
- Fixed resolution config drift by wiring `resize_mode` through resize pipeline utilities.
- Fixed Pylance/type warnings in downloader retry helpers and web API dict typing.
- Fixed PIL resampling typing warning by using `Image.Resampling.LANCZOS`.
- Fixed `utils.__all__` unsupported entries warning while preserving lazy-loading behavior.

## [0.1.0] - 2026-03-17

### Added
- Initial public codebase baseline for Soil Shazam Data Collector.
- End-to-end CLI pipeline (`download`, `resize`, `dedup`, `filter`, `label`, `export`, `run-all`).
- Multi-source downloading support (Bing, Google, Flickr).
- CLIP-based soil filtering, overlay detection, and multi-category labeling.
- Evaluation workflow with sampling, annotation support, metrics, and markdown reporting.
- Flask web application for pipeline browsing, annotation, and evaluation dashboards.
- Dataset export outputs (JSON/CSV/full labels with scores).
- Config-driven pipeline settings, prompts, and queries under `config/`.


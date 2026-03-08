"""Structured logging setup for the pipeline."""

import logging
import sys

# Noisy third-party loggers to silence
_QUIET_LOGGERS = [
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "huggingface_hub.utils._headers",
    "imagededup",
    "imagededup.methods.hashing",
    "imagededup.handlers.search.retrieval",
]


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Silence noisy / duplicate third-party loggers
    for name in _QUIET_LOGGERS:
        log = logging.getLogger(name)
        log.setLevel(logging.WARNING)
        log.propagate = False

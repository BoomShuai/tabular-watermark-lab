"""Deterministic, restartable TabularMark full experiment runner."""

from .core import (
    WatermarkKey,
    decision_threshold,
    detect_categorical,
    detect_numeric,
    embed_categorical,
    embed_numeric,
    make_key,
    z_statistic,
)

__all__ = [
    "WatermarkKey",
    "decision_threshold",
    "detect_categorical",
    "detect_numeric",
    "embed_categorical",
    "embed_numeric",
    "make_key",
    "z_statistic",
]

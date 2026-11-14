"""Independent, auditable extensions for the public TabularMark artifacts."""

from .rq4 import (  # noqa: F401
    WatermarkBundle,
    apply_alteration,
    build_alteration_plan,
    build_rq4_watermarks,
    detect_histmark,
    detect_semmark,
    detect_tabularmark,
    load_covertype,
    prepare_balanced_subset,
    run_alteration_comparison,
)

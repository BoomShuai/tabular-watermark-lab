#!/usr/bin/env python3
"""Thin executable wrapper for ``python -m tabularmark_runner``."""

from tabularmark_runner.runner import main


if __name__ == "__main__":
    raise SystemExit(main())

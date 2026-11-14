from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tabularmark_ext.rq4 import (  # noqa: E402
    apply_alteration,
    build_alteration_plan,
    build_rq4_watermarks,
    detect_tabularmark,
    prepare_balanced_subset,
    run_alteration_comparison,
)


def toy_covertype(rows_per_class: int = 30) -> pd.DataFrame:
    rows = []
    for label in range(1, 8):
        for offset in range(rows_per_class):
            rows.append(
                {
                    "feature_a": label * 100 + offset,
                    "feature_b": offset % 5,
                    "Cover_Type": label,
                }
            )
    return pd.DataFrame(rows)


class RQ4DeterminismTest(unittest.TestCase):
    def setUp(self) -> None:
        self.subset = prepare_balanced_subset(toy_covertype(), per_class=20, seed=73)

    def test_watermark_build_is_deterministic(self) -> None:
        first = build_rq4_watermarks(
            self.subset,
            seed=73,
            n_key_cells=40,
            watermark_length=20,
            semmark_votes_per_bit=2,
        )
        second = build_rq4_watermarks(
            self.subset,
            seed=73,
            n_key_cells=40,
            watermark_length=20,
            semmark_votes_per_bit=2,
        )
        self.assertEqual(sorted(first), ["HistMark", "SemMark", "TabularMark"])
        for scheme in first:
            pd.testing.assert_frame_equal(first[scheme].data, second[scheme].data)
            self.assertEqual(first[scheme].state, second[scheme].state)
        self.assertEqual(detect_tabularmark(first["TabularMark"].data, first["TabularMark"].state)["mismatch_percentage"], 0.0)

    def test_attack_plan_is_nested_and_shared(self) -> None:
        plan = build_alteration_plan(len(self.subset), list(range(1, 8)), seed=73, repeat=0)
        attack_30 = apply_alteration(self.subset, 0.30, plan)
        attack_50 = apply_alteration(self.subset, 0.50, plan)
        prefix = plan["row_order"][: int(len(self.subset) * 0.30)]
        self.assertEqual(
            attack_30.iloc[prefix]["Cover_Type"].tolist(),
            attack_50.iloc[prefix]["Cover_Type"].tolist(),
        )
        pd.testing.assert_frame_equal(self.subset, apply_alteration(self.subset, 0.0, plan))

    def test_comparison_repeats_exactly(self) -> None:
        bundles = build_rq4_watermarks(
            self.subset,
            seed=73,
            n_key_cells=40,
            watermark_length=20,
            semmark_votes_per_bit=2,
        )
        first_repeats, first_summary = run_alteration_comparison(
            bundles, [0.0, 0.5, 1.0], repeats=2, seed=73
        )
        second_repeats, second_summary = run_alteration_comparison(
            bundles, [0.0, 0.5, 1.0], repeats=2, seed=73
        )
        pd.testing.assert_frame_equal(first_repeats, second_repeats)
        pd.testing.assert_frame_equal(first_summary, second_summary)
        self.assertTrue(first_repeats["mismatch_percentage"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()

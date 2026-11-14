from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tabularmark_ext.notebook_audit import audit_official_tree  # noqa: E402


class NotebookAuditTest(unittest.TestCase):
    def test_missing_input_and_stored_output_are_not_reported_as_run(self) -> None:
        old_path = '/' + 'home/' + 'example/old.csv'
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 7,
                    "metadata": {},
                    "source": ["import pandas as pd\n", "pd.read_csv('../data/missing.csv')\n"],
                    "outputs": [
                        {
                            "output_type": "stream",
                            "name": "stdout",
                            "text": ["read " + old_path + "\n"],
                        }
                    ],
                }
            ],
            "metadata": {"kernelspec": {"name": "python3"}, "language_info": {"version": "3.10"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "experiments" / "compare" / "case.ipynb"
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(notebook), encoding="utf-8")
            first = audit_official_tree(root, include_python=False)
            second = audit_official_tree(root, include_python=False)
        self.assertEqual(first["source_tree_sha256"], second["source_tree_sha256"])
        unit = first["units"][0]
        self.assertEqual(unit["rq"], "RQ4")
        self.assertEqual(unit["status"], "BLOCKED_MISSING_EXTERNAL_INPUT")
        self.assertEqual(unit["missing_external_inputs"], ["../data/missing.csv"])
        self.assertIn(old_path, unit["stored_output_absolute_paths"])


if __name__ == "__main__":
    unittest.main()

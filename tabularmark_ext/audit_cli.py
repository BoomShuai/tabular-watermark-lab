#!/usr/bin/env python3
"""Generate a static coverage CSV and status JSON for official execution units."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tabularmark_ext.notebook_audit import audit_official_tree, write_audit_outputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Static-only audit: notebook cells are never executed and stored outputs remain unverified."
    )
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    parser.add_argument(
        "--notebooks-only",
        action="store_true",
        help="Exclude official .py entry points from the matrix",
    )
    args = parser.parse_args()
    source = args.official_root.resolve()
    outputs = [args.json_output.resolve(), args.csv_output.resolve()]
    if outputs[0] == outputs[1] or any(path == source or source in path.parents or path.exists() for path in outputs):
        parser.error('Outputs must be new, distinct files outside the audited source tree')
    payload = audit_official_tree(args.official_root, include_python=not args.notebooks_only)
    write_audit_outputs(payload, args.json_output, args.csv_output)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

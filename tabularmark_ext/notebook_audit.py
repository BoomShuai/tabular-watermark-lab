"""Static coverage/status audit for the public TabularMark execution units.

The scanner never executes notebook cells.  Stored notebook outputs are treated
as historical, unverified evidence rather than as results from the current run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


RQ_BY_PATH = {
    "detectability": "RQ1",
    "non_intrusiveness": "RQ2",
    "robustness": "RQ3",
    "compare": "RQ4",
    "trade-offs": "RQ5",
    "discussion": "DISCUSSION",
}

LITERAL_ASSIGNMENT = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*(?:[rubfRUBF]{0,2})?(['\"])([^'\"]+)\2\s*(?:#.*)?$"
)
CALL_ARGUMENT = re.compile(
    r"(?P<call>(?:pd\.)?read_csv|(?:pd\.)?read_excel|np\.load|np\.save|"
    r"(?:[A-Za-z_]\w*\.)?to_csv|open)\s*\(\s*(?P<arg>[A-Za-z_]\w*|"
    r"(?:[rubfRUBF]{0,2})?['\"][^'\"]+['\"])(?P<rest>[^\n]*)"
)
STRING_LITERAL = re.compile(r"^(?:[rubfRUBF]{0,2})?(['\"])([^'\"]+)\1$")
IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s+import|"
    r"import\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*))",
    re.MULTILINE,
)
ABSOLUTE_PATH_PATTERN = re.compile(r"/(?:home|Users|content|mnt|workspace)/[^\s'\"<>]+")


def _rq_for(relative_path: Path) -> str:
    parts = set(relative_path.parts)
    for marker, rq in RQ_BY_PATH.items():
        if marker in parts:
            return rq
    return "UNMAPPED"


def _source_tree_digest(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _code_and_notebook_metadata(path: Path) -> Dict[str, Any]:
    if path.suffix == ".py":
        code = path.read_text(encoding="utf-8", errors="replace")
        return {
            "code": code,
            "code_cells": 0,
            "executed_code_cells": 0,
            "output_cells": 0,
            "stored_error_outputs": [],
            "stored_output_text": "",
            "kernel_name": None,
            "kernel_python_version": None,
        }

    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    sources: List[str] = []
    output_text: List[str] = []
    errors: List[Dict[str, str]] = []
    output_cells = 0
    executed = 0
    for cell in code_cells:
        source = cell.get("source", [])
        sources.append("".join(source) if isinstance(source, list) else str(source))
        if cell.get("execution_count") is not None:
            executed += 1
        outputs = cell.get("outputs", []) or []
        if outputs:
            output_cells += 1
        for output in outputs:
            if output.get("output_type") == "error":
                errors.append(
                    {"ename": str(output.get("ename", "")), "evalue": str(output.get("evalue", ""))}
                )
            text = output.get("text")
            if isinstance(text, list):
                output_text.append("".join(str(value) for value in text))
            elif text is not None:
                output_text.append(str(text))
            data = output.get("data", {})
            plain = data.get("text/plain") if isinstance(data, dict) else None
            if isinstance(plain, list):
                output_text.append("".join(str(value) for value in plain))
            elif plain is not None:
                output_text.append(str(plain))
    metadata = notebook.get("metadata", {})
    return {
        "code": "\n\n".join(sources),
        "code_cells": len(code_cells),
        "executed_code_cells": executed,
        "output_cells": output_cells,
        "stored_error_outputs": errors,
        "stored_output_text": "\n".join(output_text),
        "kernel_name": metadata.get("kernelspec", {}).get("name"),
        "kernel_python_version": metadata.get("language_info", {}).get("version"),
    }


def _literal_value(token: str, variables: Mapping[str, str]) -> Optional[str]:
    token = token.strip()
    literal = STRING_LITERAL.match(token)
    if literal:
        value = literal.group(2)
        # Dynamic f-strings cannot be resolved soundly by this static scanner.
        prefix = token[: token.find(literal.group(1))].lower()
        if "f" in prefix or "{" in value:
            return None
        return value
    return variables.get(token)


def _extract_path_operations(code: str) -> Tuple[List[str], List[str], List[str]]:
    variables: Dict[str, str] = {}
    reads: List[str] = []
    writes: List[str] = []
    dynamic: List[str] = []
    for line_number, line in enumerate(code.splitlines(), start=1):
        assignment = LITERAL_ASSIGNMENT.match(line)
        if assignment:
            variables[assignment.group(1)] = assignment.group(3)
        for match in CALL_ARGUMENT.finditer(line):
            call = match.group("call")
            token = match.group("arg")
            rest = match.group("rest")
            path_value = _literal_value(token, variables)
            if path_value is None:
                dynamic.append(f"line {line_number}: {call}({token}, ...)")
                continue
            if call.endswith("to_csv") or call == "np.save":
                writes.append(path_value)
                continue
            if call == "open":
                mode_match = re.match(r"\s*,\s*['\"]([^'\"]+)['\"]", rest)
                mode = mode_match.group(1) if mode_match else "r"
                if any(flag in mode for flag in ("w", "a", "x", "+")):
                    writes.append(path_value)
                else:
                    reads.append(path_value)
                continue
            reads.append(path_value)
    return sorted(set(reads)), sorted(set(writes)), sorted(set(dynamic))


def _resolve_path(raw: str, unit_path: Path) -> Path:
    expanded = Path(raw).expanduser()
    return expanded if expanded.is_absolute() else (unit_path.parent / expanded).resolve()


def _imports(code: str) -> List[str]:
    modules = []
    for match in IMPORT_PATTERN.finditer(code):
        module = match.group(1) or match.group(2)
        modules.append(module.split(".", 1)[0])
    return sorted(set(modules))


def _status_for(unit: Mapping[str, Any]) -> str:
    if unit["missing_external_inputs"]:
        return "BLOCKED_MISSING_EXTERNAL_INPUT"
    if unit["missing_generated_dependencies"]:
        return "ORDER_DEPENDENT_GENERATED_INPUT"
    if unit["stored_error_outputs"]:
        return "STORED_ERROR_OUTPUT"
    if unit["unit_type"] == "python_script":
        return "STATIC_SCRIPT_UNVERIFIED"
    if unit["code_cells"] == 0:
        return "NO_CODE_CELLS"
    if unit["executed_code_cells"] == unit["code_cells"]:
        return "STORED_OUTPUTS_UNVERIFIED"
    if unit["executed_code_cells"] > 0:
        return "PARTIAL_STORED_OUTPUTS_UNVERIFIED"
    return "NOT_EXECUTED_NO_STORED_RUN"


def audit_official_tree(official_root: Path, include_python: bool = True) -> Dict[str, Any]:
    """Return a deterministic static audit for notebooks and optional scripts."""

    official_root = Path(official_root).expanduser().resolve()
    if not official_root.is_dir():
        raise FileNotFoundError(f"Official root does not exist: {official_root}")
    patterns = ["*.ipynb"] + (["*.py"] if include_python else [])
    paths = sorted({path for pattern in patterns for path in official_root.rglob(pattern)})
    if not paths:
        raise ValueError(f"No execution units found under {official_root}")

    provisional: List[MutableMapping[str, Any]] = []
    produced_paths: Dict[Path, List[str]] = defaultdict(list)
    for path in paths:
        metadata = _code_and_notebook_metadata(path)
        reads, writes, dynamic = _extract_path_operations(metadata["code"])
        relative = path.relative_to(official_root)
        row: MutableMapping[str, Any] = {
            "source_path": relative.as_posix(),
            "unit_type": "notebook" if path.suffix == ".ipynb" else "python_script",
            "rq": _rq_for(relative),
            "experiment_area": relative.parent.as_posix(),
            "code_cells": int(metadata["code_cells"]),
            "executed_code_cells": int(metadata["executed_code_cells"]),
            "output_cells": int(metadata["output_cells"]),
            "stored_error_outputs": metadata["stored_error_outputs"],
            "kernel_name": metadata["kernel_name"],
            "kernel_python_version": metadata["kernel_python_version"],
            "imports": _imports(metadata["code"]),
            "declared_reads": reads,
            "declared_writes": writes,
            "dynamic_path_operations": dynamic,
            "stored_output_absolute_paths": sorted(
                set(ABSOLUTE_PATH_PATTERN.findall(metadata["stored_output_text"]))
            ),
            "_absolute_path": path,
        }
        for raw in writes:
            produced_paths[_resolve_path(raw, path)].append(relative.as_posix())
        provisional.append(row)

    rows: List[Dict[str, Any]] = []
    for row in provisional:
        path = row.pop("_absolute_path")
        available: List[str] = []
        missing_external: List[str] = []
        missing_generated: List[Dict[str, Any]] = []
        for raw in row["declared_reads"]:
            resolved = _resolve_path(raw, path)
            if resolved.exists():
                available.append(raw)
            elif resolved in produced_paths:
                missing_generated.append(
                    {"path": raw, "producers": sorted(produced_paths[resolved])}
                )
            else:
                missing_external.append(raw)
        row["available_inputs"] = sorted(available)
        row["missing_external_inputs"] = sorted(missing_external)
        row["missing_generated_dependencies"] = missing_generated
        row["status"] = _status_for(row)
        rows.append(dict(row))

    status_counts = Counter(row["status"] for row in rows)
    rq_counts: Dict[str, Dict[str, int]] = {}
    for rq in sorted(set(row["rq"] for row in rows)):
        rq_rows = [row for row in rows if row["rq"] == rq]
        rq_counts[rq] = {
            "total": len(rq_rows),
            "notebooks": sum(row["unit_type"] == "notebook" for row in rq_rows),
            "python_scripts": sum(row["unit_type"] == "python_script" for row in rq_rows),
            "blocked_missing_external_input": sum(
                row["status"] == "BLOCKED_MISSING_EXTERNAL_INPUT" for row in rq_rows
            ),
        }

    return {
        "schema_version": 1,
        "audit_method": "STATIC_ONLY_NO_CELL_EXECUTION",
        "official_root": str(official_root),
        "source_tree_sha256": _source_tree_digest(paths, official_root),
        "status_semantics": {
            "BLOCKED_MISSING_EXTERNAL_INPUT": "At least one statically resolved read path is absent.",
            "ORDER_DEPENDENT_GENERATED_INPUT": "An absent input has a producer elsewhere in the tree.",
            "STORED_ERROR_OUTPUT": "The notebook contains a historical error output.",
            "STORED_OUTPUTS_UNVERIFIED": (
                "All code cells have execution counters; outputs were not rerun or validated here."
            ),
            "PARTIAL_STORED_OUTPUTS_UNVERIFIED": "Only some code cells have execution counters.",
            "NOT_EXECUTED_NO_STORED_RUN": "No code cell has a stored execution counter.",
            "STATIC_SCRIPT_UNVERIFIED": "Python source was inspected but not executed.",
            "NO_CODE_CELLS": "Notebook contains no code cells.",
        },
        "limitations": [
            "Dynamic, computed, globbed, and environment-dependent paths are reported but not resolved.",
            "A stored execution counter is not evidence that current code produced the stored output.",
            "Dependency importability and semantic correctness require a separate controlled run.",
        ],
        "summary": {
            "total_units": len(rows),
            "notebooks": sum(row["unit_type"] == "notebook" for row in rows),
            "python_scripts": sum(row["unit_type"] == "python_script" for row in rows),
            "status_counts": dict(sorted(status_counts.items())),
            "rq_counts": rq_counts,
        },
        "units": rows,
    }


CSV_COLUMNS = [
    "source_path",
    "unit_type",
    "rq",
    "experiment_area",
    "status",
    "code_cells",
    "executed_code_cells",
    "output_cells",
    "kernel_name",
    "kernel_python_version",
    "missing_external_input_count",
    "missing_external_inputs",
    "missing_generated_dependency_count",
    "missing_generated_dependencies",
    "dynamic_path_operation_count",
    "stored_output_absolute_path_count",
    "imports",
]


def write_audit_outputs(payload: Mapping[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for unit in payload["units"]:
            row = {
                "source_path": unit["source_path"],
                "unit_type": unit["unit_type"],
                "rq": unit["rq"],
                "experiment_area": unit["experiment_area"],
                "status": unit["status"],
                "code_cells": unit["code_cells"],
                "executed_code_cells": unit["executed_code_cells"],
                "output_cells": unit["output_cells"],
                "kernel_name": unit["kernel_name"] or "",
                "kernel_python_version": unit["kernel_python_version"] or "",
                "missing_external_input_count": len(unit["missing_external_inputs"]),
                "missing_external_inputs": json.dumps(
                    unit["missing_external_inputs"], ensure_ascii=False, sort_keys=True
                ),
                "missing_generated_dependency_count": len(unit["missing_generated_dependencies"]),
                "missing_generated_dependencies": json.dumps(
                    unit["missing_generated_dependencies"], ensure_ascii=False, sort_keys=True
                ),
                "dynamic_path_operation_count": len(unit["dynamic_path_operations"]),
                "stored_output_absolute_path_count": len(unit["stored_output_absolute_paths"]),
                "imports": ";".join(unit["imports"]),
            }
            writer.writerow(row)

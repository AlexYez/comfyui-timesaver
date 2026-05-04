"""Static collector for the public node contracts in this pack.

Walks every Python file under ``nodes/`` (recursively) and extracts the
metadata that defines each node's public contract: node_id, class name,
file path, category, display name, and the API style (V1 vs V3).

The result is the snapshot consumed by ``tests/test_node_contracts.py``.
The collector uses AST parsing only — it does NOT import the modules,
so it works without ``comfy_api`` / ``folder_paths`` / ``torch``.

Usage:

    python tools/build_node_contracts.py            # write tests/contracts/node_contracts.json
    python tools/build_node_contracts.py --check    # exit non-zero on drift
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NODES_DIR = PACKAGE_ROOT / "nodes"
SNAPSHOT_PATH = PACKAGE_ROOT / "tests" / "contracts" / "node_contracts.json"


@dataclass
class NodeContract:
    node_id: str
    class_name: str
    python_file: str
    api: str
    category: str | None = None
    display_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "api": self.api,
            "class_name": self.class_name,
            "python_file": self.python_file,
            "category": self.category,
            "display_name": self.display_name,
        }
        return {k: v for k, v in data.items() if v is not None or k in {"category", "display_name"}}


@dataclass
class _ModuleScan:
    path: Path
    tree: ast.Module
    classes: dict[str, ast.ClassDef] = field(default_factory=dict)
    mappings: dict[str, str] = field(default_factory=dict)
    display_names: dict[str, str] = field(default_factory=dict)


def _iter_node_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        relative = path.relative_to(root)
        if any(part.startswith("_") for part in relative.parts):
            continue
        yield path


def _collect_module_scan(path: Path) -> _ModuleScan:
    # utf-8-sig strips the byte-order mark some files carry on Windows.
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    scan = _ModuleScan(path=path, tree=tree)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            scan.classes[node.name] = node
            continue

        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue

        target = node.targets[0]
        if isinstance(target, ast.Name):
            if target.id == "NODE_CLASS_MAPPINGS":
                scan.mappings.update(_dict_to_str_class_pairs(node.value))
            elif target.id == "NODE_DISPLAY_NAME_MAPPINGS":
                scan.display_names.update(_dict_to_str_str_pairs(node.value))
            continue

        # Index-style: NODE_CLASS_MAPPINGS["X"] = Y or NODE_DISPLAY_NAME_MAPPINGS["X"] = "..."
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            container = target.value.id
            key_node = target.slice
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                key = key_node.value
                if container == "NODE_CLASS_MAPPINGS" and isinstance(node.value, ast.Name):
                    scan.mappings[key] = node.value.id
                elif container == "NODE_DISPLAY_NAME_MAPPINGS" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    scan.display_names[key] = node.value.value

    return scan


def _dict_to_str_class_pairs(value: ast.expr) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(value, ast.Dict):
        return result
    for key_node, val_node in zip(value.keys, value.values):
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str) and isinstance(val_node, ast.Name):
            result[key_node.value] = val_node.id
    return result


def _dict_to_str_str_pairs(value: ast.expr) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(value, ast.Dict):
        return result
    for key_node, val_node in zip(value.keys, value.values):
        if (
            isinstance(key_node, ast.Constant)
            and isinstance(key_node.value, str)
            and isinstance(val_node, ast.Constant)
            and isinstance(val_node.value, str)
        ):
            result[key_node.value] = val_node.value
    return result


def _detect_api_style(class_def: ast.ClassDef) -> str:
    for stmt in class_def.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "define_schema":
            return "v3"
        if isinstance(stmt, ast.AsyncFunctionDef) and stmt.name == "define_schema":
            return "v3"
    return "v1"


def _extract_v1_category(class_def: ast.ClassDef) -> str | None:
    for stmt in class_def.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and target.id == "CATEGORY":
                if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    return stmt.value.value
    return None


def _extract_v3_schema_kwargs(class_def: ast.ClassDef) -> dict[str, str]:
    """Return node_id/category/display_name from a V3 define_schema return."""

    for stmt in class_def.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if stmt.name != "define_schema":
            continue
        for inner in ast.walk(stmt):
            if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Call):
                call = inner.value
                func_repr = _attr_chain(call.func)
                if func_repr in {"IO.Schema", "io.Schema", "Schema"}:
                    return _kwargs_to_str_dict(call.keywords)
    return {}


def _attr_chain(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_attr_chain(node.value)}.{node.attr}"
    return ""


def _kwargs_to_str_dict(keywords: list[ast.keyword]) -> dict[str, str]:
    out: dict[str, str] = {}
    for kw in keywords:
        if kw.arg is None:
            continue
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            out[kw.arg] = kw.value.value
    return out


def collect_contracts() -> dict[str, NodeContract]:
    contracts: dict[str, NodeContract] = {}

    for path in _iter_node_files(NODES_DIR):
        scan = _collect_module_scan(path)
        relative_path = path.relative_to(PACKAGE_ROOT).as_posix()

        for node_id, class_name in scan.mappings.items():
            class_def = scan.classes.get(class_name)
            if class_def is None:
                # Class might be imported from another module. Record what we can.
                contracts[node_id] = NodeContract(
                    node_id=node_id,
                    class_name=class_name,
                    python_file=relative_path,
                    api="unknown",
                    display_name=scan.display_names.get(node_id),
                )
                continue

            api = _detect_api_style(class_def)
            if api == "v3":
                schema_kwargs = _extract_v3_schema_kwargs(class_def)
                category = schema_kwargs.get("category")
                display_name = schema_kwargs.get("display_name") or scan.display_names.get(node_id)
            else:
                category = _extract_v1_category(class_def)
                display_name = scan.display_names.get(node_id)

            contracts[node_id] = NodeContract(
                node_id=node_id,
                class_name=class_name,
                python_file=relative_path,
                api=api,
                category=category,
                display_name=display_name,
            )

    return contracts


def serialize_snapshot(contracts: dict[str, NodeContract]) -> str:
    payload = {node_id: contract.to_dict() for node_id, contract in sorted(contracts.items())}
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def write_snapshot(contracts: dict[str, NodeContract]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(serialize_snapshot(contracts), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with non-zero code if the regenerated snapshot differs.",
    )
    args = parser.parse_args(argv)

    contracts = collect_contracts()
    serialized = serialize_snapshot(contracts)

    if args.check:
        if not SNAPSHOT_PATH.exists():
            print(f"Snapshot missing: {SNAPSHOT_PATH}", file=sys.stderr)
            return 2
        existing = SNAPSHOT_PATH.read_text(encoding="utf-8")
        if existing != serialized:
            print("Node contract snapshot drift detected.", file=sys.stderr)
            print(f"Update with: python {Path(__file__).relative_to(PACKAGE_ROOT)}", file=sys.stderr)
            return 1
        print(f"Snapshot OK: {len(contracts)} nodes")
        return 0

    write_snapshot(contracts)
    print(f"Wrote {len(contracts)} contracts -> {SNAPSHOT_PATH.relative_to(PACKAGE_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Tier-1 structural + schema validator for a NengoZoo submission.

Usage:
    python tools/validate_submission.py submissions/basal-ganglia

Exits 0 on success, 1 on failure of any blocking check.

Blocking checks:
  * Required files exist (type-aware: src/ required for component/network only)
  * metadata.yaml parses and matches schema
  * metadata.name matches folder name
  * entry_point exists on disk
  * Exactly one importable package under src/ (if src/ exists)

Informational checks (do not affect exit code, but earn badges):
  * NengoGUI-readiness — entry_point or `nengogui.script` imports and exposes
    a top-level `model: nengo.Network`
  * .cfg name resolution — every bare name in <script>.cfg resolves in the
    script's module namespace after import
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import uuid
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. `pip install pyyaml`")
    sys.exit(2)

try:
    import jsonschema
except ImportError:
    print("ERROR: jsonschema is required. `pip install jsonschema`")
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schema" / "metadata.schema.json"

# Files every submission must carry.
COMMON_REQUIRED = [
    "README.md",
    "metadata.yaml",
    "LICENSE",
    "requirements.txt",
    "tests/test_runs.py",
]
# Files required only when the submission has a reusable library API.
LIBRARY_API_REQUIRED = ["src"]

DEFAULT_ENTRY_POINT = "examples/example_usage.py"

# Identifiers referenced by NengoGUI .cfg files we care about.
CFG_NAME_PATTERNS = [
    re.compile(r"nengo_gui\.components\.[A-Za-z_]+\(([A-Za-z_][A-Za-z0-9_]*)\)"),
    re.compile(r"_viz_config\[([A-Za-z_][A-Za-z0-9_]*)\]"),
]


def check(label: str, ok: bool, detail: str = "", blocking: bool = True) -> bool:
    mark = "✓" if ok else ("✗" if blocking else "⚠")
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def has_library_api(submission_type: str) -> bool:
    """component/network REQUIRE a library API; model does NOT."""
    return submission_type in ("component", "network")


def required_paths(submission_type: str) -> list[str]:
    paths = list(COMMON_REQUIRED)
    if has_library_api(submission_type):
        paths.extend(LIBRARY_API_REQUIRED)
    return paths


def import_script(script_path: Path, extra_sys_path: list[Path] | None = None):
    """Import a Python file by path, returning the module object or None.

    Adds `extra_sys_path` to sys.path during import and restores it after.
    Uses a fresh module name to avoid caching collisions.
    """
    mod_name = f"_zoo_validate_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, str(script_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    original_path = list(sys.path)
    try:
        for p in (extra_sys_path or []):
            sys.path.insert(0, str(p))
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"      (import error: {type(e).__name__}: {e})")
        return None
    finally:
        sys.path[:] = original_path
        sys.modules.pop(mod_name, None)


def extract_cfg_names(cfg_path: Path) -> set[str]:
    """Bare identifiers referenced by a NengoGUI .cfg file."""
    names: set[str] = set()
    text = cfg_path.read_text(encoding="utf-8")
    for pat in CFG_NAME_PATTERNS:
        for m in pat.finditer(text):
            names.add(m.group(1))
    # `_viz_*` are .cfg-internal locals, not script-namespace references.
    return {n for n in names if not n.startswith("_viz_")}


def check_gui_shape(script_path: Path, submission_dir: Path) -> tuple[bool, dict]:
    """Try to import the script; return (gui_ready, module_namespace)."""
    src_dir = submission_dir / "src"
    extra_paths = [src_dir] if src_dir.is_dir() else []
    module = import_script(script_path, extra_sys_path=extra_paths)
    if module is None:
        return False, {}

    model_obj = getattr(module, "model", None)
    try:
        import nengo
    except ImportError:
        return False, {}

    is_network = isinstance(model_obj, nengo.Network)
    return is_network, vars(module)


def validate(submission_dir: Path) -> bool:
    print(f"Validating: {submission_dir.relative_to(REPO_ROOT)}")
    all_ok = True

    # --- Schema + metadata first (we need `type` to know what else to check)
    meta_path = submission_dir / "metadata.yaml"
    if not meta_path.exists():
        check("present: metadata.yaml", False, "missing")
        return False

    try:
        # encoding="utf-8" — Python on Windows with non-Western locales otherwise
        # defaults to the OEM code page (e.g. CP932) and chokes on UTF-8 content.
        with meta_path.open(encoding="utf-8") as f:
            metadata = yaml.safe_load(f)
        check("metadata.yaml parses as YAML", True)
    except yaml.YAMLError as e:
        check("metadata.yaml parses as YAML", False, str(e))
        return False

    try:
        with SCHEMA_PATH.open(encoding="utf-8") as f:
            schema = json.load(f)
    except FileNotFoundError:
        check(f"schema present at {SCHEMA_PATH}", False)
        return False

    validator_obj = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator_obj.iter_errors(metadata), key=lambda e: e.path)
    if errors:
        check("metadata.yaml matches schema", False)
        for err in errors:
            path = ".".join(str(p) for p in err.absolute_path) or "<root>"
            print(f"      - {path}: {err.message}")
        all_ok = False
    else:
        check("metadata.yaml matches schema", True)

    submission_type = metadata.get("type", "unknown")

    # --- Structural checks (type-aware) -----------------------------------
    for rel in required_paths(submission_type):
        p = submission_dir / rel
        all_ok &= check(f"present: {rel}", p.exists(),
                        "" if p.exists() else f"missing {rel}")

    # --- Name must match folder -------------------------------------------
    declared_name = metadata.get("name")
    folder_name = submission_dir.name
    all_ok &= check(
        "metadata.name matches folder name",
        declared_name == folder_name,
        f"name={declared_name!r}, folder={folder_name!r}",
    )

    # --- Entry point must exist (unless ci_runnable: false) ----------------
    ci_runnable = metadata.get("ci_runnable", True)
    entry_point = metadata.get("entry_point")
    if ci_runnable:
        entry_point = entry_point or DEFAULT_ENTRY_POINT
        ep_path = submission_dir / entry_point
        all_ok &= check(
            f"entry point present: {entry_point}",
            ep_path.exists(),
            "" if ep_path.exists() else "declared in metadata but missing on disk",
        )
    else:
        # CI-structure-only submission. entry_point is optional. If declared,
        # we still verify it exists; if not, we just note the policy.
        if entry_point:
            ep_path = submission_dir / entry_point
            all_ok &= check(
                f"entry point present (optional, ci_runnable=false): {entry_point}",
                ep_path.exists(),
                "" if ep_path.exists() else "declared in metadata but missing on disk",
            )
        else:
            ep_path = None
            check("ci_runnable=false (structure-only submission)", True,
                  "CI skips run/tests; author-claimed for declared backends",
                  blocking=False)

    # --- Library API package check (whenever src/ exists) -----------------
    src_dir = submission_dir / "src"
    if src_dir.is_dir():
        packages = [p for p in src_dir.iterdir()
                    if p.is_dir() and (p / "__init__.py").exists()]
        all_ok &= check(
            "exactly one importable package under src/",
            len(packages) == 1,
            f"found: {[p.name for p in packages]}",
        )
    elif has_library_api(submission_type):
        # already flagged by the required-paths loop above; nothing to add.
        pass

    # ---------- Informational: NengoGUI badges ----------------------------
    # Decide which file (if any) is the GUI candidate.
    nengogui = metadata.get("nengogui") or {}
    gui_script_rel = nengogui.get("script") or entry_point
    gui_script_path = submission_dir / gui_script_rel if gui_script_rel else None

    if gui_script_path and gui_script_path.exists() and gui_script_path.suffix == ".py":
        gui_ready, ns = check_gui_shape(gui_script_path, submission_dir)
        check(
            f"nengogui-ready: {gui_script_rel}",
            gui_ready,
            "has top-level `model: nengo.Network`" if gui_ready else "no top-level `model: nengo.Network` (informational)",
            blocking=False,
        )

        # .cfg validation, if a matching one exists.
        cfg_rel = nengogui.get("config") or (gui_script_rel + ".cfg")
        cfg_path = submission_dir / cfg_rel
        if gui_ready and cfg_path.exists():
            referenced = extract_cfg_names(cfg_path)
            missing = sorted(n for n in referenced if n not in ns)
            check(
                f"nengogui .cfg names resolve: {cfg_rel}",
                not missing,
                ("all " + str(len(referenced)) + " names resolve") if not missing else f"missing: {missing}",
                blocking=False,
            )

    print(f"Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: validate_submission.py <path-to-submission-dir>")
        return 2
    target = Path(argv[1]).resolve()
    if not target.is_dir():
        print(f"ERROR: not a directory: {target}")
        return 2
    return 0 if validate(target) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

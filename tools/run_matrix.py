#!/usr/bin/env python3
"""
Run the NengoZoo Tier-1 review (validator + pytest) against each Nengo
version in a matrix, recording the result per submission.

Writes ``submissions/<name>/.tested-on.json`` per submission:

    {
      "tested_at": "2026-05-15T20:15:00Z",
      "python":    "3.10.12",
      "numpy":     "2.2.6",
      "matrix": [
        {"nengo": "3.2.0", "status": "pass", "details": "5 tests, validator OK"},
        {"nengo": "4.1.0", "status": "pass", "details": "5 tests, validator OK"}
      ]
    }

The build_site script reads these files and renders the tested-on badges.

Usage:
    python tools/run_matrix.py --nengo 3.2.0 4.1.0

WARNING: this installs each Nengo version in the *current* Python
environment (in sequence). For an isolated run, invoke inside a fresh venv,
or let GitHub Actions handle the full matrix.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except ImportError:
    print("ERROR: `packaging` is required. `pip install packaging`")
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
VALIDATOR = REPO_ROOT / "tools" / "validate_submission.py"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def pip_install_nengo(version: str) -> str:
    """Install nengo==version (and a compatible numpy). Returns error msg or ''."""
    # Older Nengo (3.x) breaks on NumPy>=2; pin numpy<2 when installing 3.x.
    nengo_v = Version(version)
    numpy_constraint = "numpy<2" if nengo_v.major < 4 else "numpy"

    proc = run([
        sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages",
        f"nengo=={version}", numpy_constraint,
    ])
    if proc.returncode != 0:
        return f"pip install failed: {proc.stderr.strip()[:300]}"
    return ""


def clear_nengo_decoder_cache() -> None:
    """Nuke Nengo's on-disk decoder cache.

    Pickled cache files reference internal numpy module paths (e.g.
    `numpy._core.numeric` in NumPy 2.x) that don't exist in older NumPy.
    Without this clear, stale cache entries from a previous matrix cell
    cause ModuleNotFoundError on load when we downgrade NumPy.
    """
    candidates = [
        Path.home() / ".cache" / "nengo" / "decoders",
        Path.home() / "Library" / "Caches" / "nengo" / "decoders",  # macOS
        Path(os.environ.get("XDG_CACHE_HOME", "")) / "nengo" / "decoders"
            if os.environ.get("XDG_CACHE_HOME") else None,
    ]
    for c in candidates:
        if c and c.is_dir():
            shutil.rmtree(c, ignore_errors=True)


def test_submission(sub_dir: Path) -> tuple[bool, str]:
    """Run validator + pytest on a submission. Returns (passed, details)."""
    val = run([sys.executable, str(VALIDATOR), str(sub_dir)])
    if val.returncode != 0:
        tail = val.stdout.strip().splitlines()[-3:]
        return False, "validator FAIL: " + " | ".join(tail)

    env = {
        **os.environ,
        "PYTHONPATH": f"src:.{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "MPLBACKEND": "Agg",
    }
    pyt = run(
        [sys.executable, "-m", "pytest", "tests/", "-q",
         "--basetemp", f"/tmp/pyz-matrix-{sub_dir.name}",
         "-p", "no:cacheprovider"],
        cwd=sub_dir, env=env,
    )
    last = (pyt.stdout.strip().splitlines() or [""])[-1]
    if pyt.returncode != 0:
        return False, f"pytest FAIL: {last}"
    return True, f"validator OK, pytest: {last}"


def matrix_for_submission(sub_dir: Path, nengo_versions: list[str]) -> dict:
    metadata = (sub_dir / "metadata.yaml").read_text()
    # crude one-line extraction of nengo_version
    import yaml
    declared = SpecifierSet(yaml.safe_load(metadata)["nengo_version"])

    results = []
    for v in nengo_versions:
        ver_obj = Version(v)
        if ver_obj not in declared:
            results.append({"nengo": v, "status": "skip",
                            "details": f"out of declared range {declared}"})
            continue

        print(f"  [{sub_dir.name}] installing nengo=={v} ...", flush=True)
        err = pip_install_nengo(v)
        if err:
            results.append({"nengo": v, "status": "fail", "details": err})
            continue

        # Stale decoder cache from a previous matrix cell with a different
        # NumPy version will deserialize-fail. Clear before testing.
        clear_nengo_decoder_cache()

        ok, details = test_submission(sub_dir)
        results.append({
            "nengo": v,
            "status": "pass" if ok else "fail",
            "details": details,
        })
        print(f"    → {results[-1]['status']}: {details}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nengo", nargs="+", required=True,
                   help="Nengo versions to test against (e.g. --nengo 3.2.0 4.1.0)")
    p.add_argument("--submission", help="Limit to one submission (optional)")
    p.add_argument("--restore", default=None,
                   help="Nengo version to reinstall at the end (default: leave whatever ran last)")
    args = p.parse_args()

    import numpy
    try:
        import nengo  # noqa
        starting_nengo = nengo.__version__
    except Exception:
        starting_nengo = None

    targets = sorted(SUBMISSIONS_DIR.iterdir()) if not args.submission \
        else [SUBMISSIONS_DIR / args.submission]
    targets = [t for t in targets if (t / "metadata.yaml").exists()]

    for sub_dir in targets:
        print(f"=== {sub_dir.name} ===")
        results = matrix_for_submission(sub_dir, args.nengo)
        payload = {
            "tested_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python":    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy":     numpy.__version__,
            "matrix":    results,
        }
        (sub_dir / ".tested-on.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  wrote {sub_dir.name}/.tested-on.json")

    if args.restore:
        print(f"\nRestoring nengo=={args.restore}...")
        pip_install_nengo(args.restore)
    elif starting_nengo:
        print(f"\nLeft nengo at {args.nengo[-1]} (started session at {starting_nengo}).")


if __name__ == "__main__":
    main()

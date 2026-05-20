#!/usr/bin/env python3
"""
Shared helper for building a downloadable zip of a submission folder.

Lives on its own (rather than inside build_site.py) so that tools which
shouldn't pull in the site-build dependency stack (jinja2, markdown, pygments)
— notably sync_zenodo.py — can produce byte-for-byte identical archives. The
site download button and the Zenodo deposit must zip the same way, or a user's
cited artifact wouldn't match what they downloaded.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

# Files / directories we never want inside a downloadable zip.
ZIP_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                    ".tox", ".venv", "node_modules", ".git", ".DS_Store"}
ZIP_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def make_submission_zip(sub_dir: Path, version: str, out_path: Path) -> int:
    """Zip a submission folder to out_path. Returns the byte size."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    archive_root = f"{sub_dir.name}-{version}"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(sub_dir.rglob("*")):
            if any(part in ZIP_EXCLUDE_DIRS for part in path.parts):
                continue
            if path.suffix in ZIP_EXCLUDE_SUFFIXES:
                continue
            if path.is_file():
                arcname = f"{archive_root}/{path.relative_to(sub_dir)}"
                zf.write(path, arcname)
    return out_path.stat().st_size

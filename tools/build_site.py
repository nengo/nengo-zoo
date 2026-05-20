#!/usr/bin/env python3
"""
Build the static NengoZoo site from the submissions/ directory.

Walks every folder under submissions/, loads metadata.yaml + README.md,
detects NengoGUI-readiness by re-using the validator's check, copies any
figures, and renders:

  - site/index.html         (card-grid landing page)
  - site/submissions/<n>/index.html   (one per submission)
  - site/style.css

Pure Python; depends only on PyYAML, Jinja2, and the `markdown` library.

Usage:
    python tools/build_site.py [--out site] [--strict]

"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:
    import yaml
    import markdown
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    from pygments.formatters import HtmlFormatter
except ImportError as e:
    print(f"ERROR: missing dependency: {e}")
    print("Install with: pip install pyyaml jinja2 markdown pygments")
    sys.exit(2)

# Reuse the validator's GUI-shape check so the badges agree.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_submission import check_gui_shape  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
ASSETS_DIR = Path(__file__).resolve().parent / "site_assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"
CSS_SOURCE = ASSETS_DIR / "style.css"

MARKDOWN_EXTENSIONS = ["fenced_code", "tables", "sane_lists", "codehilite"]

# Files / directories we never want inside a downloadable zip.
ZIP_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                    ".tox", ".venv", "node_modules", ".git", ".DS_Store"}
ZIP_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def make_submission_zip(sub_dir: Path, version: str, out_path: Path) -> int:
    """Zip a submission folder to out_path. Returns the byte size."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    archive_root = f"{sub_dir.name}-{version}"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sub_dir.rglob("*"):
            if any(part in ZIP_EXCLUDE_DIRS for part in path.parts):
                continue
            if path.suffix in ZIP_EXCLUDE_SUFFIXES:
                continue
            if path.is_file():
                arcname = f"{archive_root}/{path.relative_to(sub_dir)}"
                zf.write(path, arcname)
    return out_path.stat().st_size


def format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f} {unit}" if unit == "B" else f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


# --- Discussion (community-signal) fetch -----------------------------------
# At build time we ask GitHub's GraphQL API for the 👍 reaction count and
# the reply count of each submission's Discussion thread, then bake those
# numbers into the rendered pages. The site is static, so counts are only
# fresh as of the last build — pages.yml runs on every push to main, after
# each Tier-1 CI run, and on a cron schedule to keep staleness bounded even
# during quiet periods.
#
# Repo coordinates are taken from GITHUB_REPOSITORY (set automatically inside
# Actions) and fall back to env-vars / defaults so local rebuilds still work.
# A missing or unreadable token degrades to "no badges" rather than failing
# the whole build — community signal is nice-to-have, not load-bearing.

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

DISCUSSION_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    discussion(number: $number) {
      url
      upvoteCount
      comments { totalCount }
    }
  }
}
"""


def _resolve_repo_slug() -> tuple[str, str] | None:
    """Determine (owner, name) of the host repo for GraphQL queries."""
    slug = os.environ.get("GITHUB_REPOSITORY")
    if slug and "/" in slug:
        owner, name = slug.split("/", 1)
        return owner, name
    # Allow explicit overrides for local builds.
    owner = os.environ.get("NENGOZOO_REPO_OWNER")
    name = os.environ.get("NENGOZOO_REPO_NAME")
    if owner and name:
        return owner, name
    return None


def fetch_discussion_signal(discussion_number: int,
                            owner: str, name: str,
                            token: str) -> dict | None:
    """Return {url, stars, comments} for one Discussion, or None on any failure."""
    payload = json.dumps({
        "query": DISCUSSION_QUERY,
        "variables": {"owner": owner, "name": name, "number": discussion_number},
    }).encode("utf-8")
    req = urllib.request.Request(
        GITHUB_GRAPHQL_URL,
        data=payload,
        headers={
            "Authorization": f"bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "nengozoo-build-site",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        print(f"  WARN: discussion #{discussion_number} fetch failed ({e}); skipping signal")
        return None

    disc = (body.get("data") or {}).get("repository", {}).get("discussion")
    if not disc:
        # Could be: discussion deleted, number wrong, or insufficient perms.
        errors = body.get("errors")
        if errors:
            print(f"  WARN: discussion #{discussion_number}: {errors[0].get('message', errors)}")
        else:
            print(f"  WARN: discussion #{discussion_number} not found")
        return None

    # "Stars" = GitHub's native discussion upvotes (the ↑ arrow on the top
    # post), not 👍 emoji reactions. Upvotes are one-click and purpose-built
    # for ranking, so they're the signal users actually reach for.
    return {
        "url": disc["url"],
        "stars": disc.get("upvoteCount", 0),
        "comments": (disc.get("comments") or {}).get("totalCount", 0),
    }


def fetch_all_discussion_signals(submissions: list[dict]) -> None:
    """Annotate each submission dict with a 'discussion_signal' field (or None)."""
    # Default-attach None so templates can render unconditionally.
    for s in submissions:
        s["discussion_signal"] = None

    repo = _resolve_repo_slug()
    if repo is None:
        print("  (no GITHUB_REPOSITORY set; skipping discussion signal fetch)")
        return
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("  (no GITHUB_TOKEN set; skipping discussion signal fetch)")
        return

    owner, name = repo
    with_discussions = [s for s in submissions if s.get("discussion")]
    if not with_discussions:
        return
    print(f"  fetching discussion signal for {len(with_discussions)} submission(s)…")
    for s in with_discussions:
        signal = fetch_discussion_signal(s["discussion"], owner, name, token)
        if signal:
            s["discussion_signal"] = signal


def first_line_after_h1(md_text: str) -> str:
    """Extract the line right after the H1 — used as the card tagline."""
    lines = md_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("# ") and i + 1 < len(lines):
            for follow in lines[i + 1:]:
                stripped = follow.strip()
                if stripped:
                    return stripped
            break
    return ""


def load_submission(sub_dir: Path) -> dict:
    """Parse one submission folder into a render-ready dict."""
    # encoding="utf-8" — see note in validate_submission.py; otherwise breaks
    # on Windows with non-Western default code pages (e.g. CP932).
    meta = yaml.safe_load((sub_dir / "metadata.yaml").read_text(encoding="utf-8"))
    readme_md = (sub_dir / "README.md").read_text(encoding="utf-8")

    # Inspect the GUI-candidate script (entry_point or nengogui.script)
    # to decide whether to show the GUI badge.
    nengogui = meta.get("nengogui") or {}
    gui_script_rel = nengogui.get("script") or meta.get("entry_point", "examples/example_usage.py")
    gui_script_path = sub_dir / gui_script_rel
    gui_ready = False
    if gui_script_path.exists() and gui_script_path.suffix == ".py":
        try:
            gui_ready, _ = check_gui_shape(gui_script_path, sub_dir)
        except Exception:
            gui_ready = False

    figures_src = sub_dir / "figures"
    figures: list[dict] = []
    if figures_src.is_dir():
        for img in sorted(figures_src.iterdir()):
            if img.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif"}:
                figures.append({"name": img.stem, "src": img, "filename": img.name})

    # Tested-on matrix (written by tools/run_matrix.py or by CI).
    tested_on = None
    tested_on_path = sub_dir / ".tested-on.json"
    if tested_on_path.exists():
        import json as _json
        try:
            tested_on = _json.loads(tested_on_path.read_text(encoding="utf-8"))
        except _json.JSONDecodeError:
            tested_on = None

    ci_runnable = meta.get("ci_runnable", True)
    declared_entry = meta.get("entry_point")
    # Only show an entry-point row when one was actually declared (or when
    # we're running a default-CI-runnable submission).
    rendered_entry = declared_entry if declared_entry \
        else ("examples/example_usage.py" if ci_runnable else None)

    return {
        "name":           meta["name"],
        "type":           meta["type"],
        "description":    meta.get("description", "").strip(),
        "tagline":        first_line_after_h1(readme_md),
        "tags":           meta.get("tags", []),
        "version":        meta["version"],
        "license":        meta["license"],
        "nengo_version":  meta["nengo_version"],
        "backends":       meta.get("backends", ["core"]),
        "complexity":     meta["complexity"],
        "authors":        meta.get("authors", []),
        "entry_point":    rendered_entry,
        "ci_runnable":    ci_runnable,
        "paper":          meta.get("paper"),
        "related":        meta.get("related", []),
        "nengogui_ready": gui_ready,
        "tested_on":      tested_on,
        # GitHub Discussion number. The discussion_signal dict (stars,
        # comments, url) is attached later by fetch_all_discussion_signals,
        # since fetching is a build-wide concern that wants a single token
        # check and would be wasteful per-submission.
        "discussion":     meta.get("discussion"),
        "readme_md":      readme_md,
        "figures_src":    figures,        # absolute source paths
        "sub_dir":        sub_dir,
        # Explicit index.html so the link works on both real web servers
        # and from the local filesystem (file://). On a server it's a no-op;
        # locally it prevents the browser from showing a directory listing.
        "url":            f"submissions/{meta['name']}/index.html",
    }


def render(out_root: Path) -> int:
    # Try to wipe any stale build; fall back to overwriting in place if we
    # can't remove individual files (e.g. due to filesystem permissions).
    if out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    # ----- Load all submissions -----
    if not SUBMISSIONS_DIR.is_dir():
        print(f"ERROR: no submissions/ found at {SUBMISSIONS_DIR}")
        return 1

    submissions = []
    for sub_dir in sorted(SUBMISSIONS_DIR.iterdir()):
        if not sub_dir.is_dir() or sub_dir.name.startswith("."):
            continue
        if not (sub_dir / "metadata.yaml").exists():
            print(f"  skipping {sub_dir.name}: no metadata.yaml")
            continue
        try:
            submissions.append(load_submission(sub_dir))
            print(f"  loaded {sub_dir.name}")
        except Exception as e:
            print(f"  ERROR loading {sub_dir.name}: {type(e).__name__}: {e}")
            return 1

    # ----- Pull community-signal counts (one network round-trip per submission
    #       that has a discussion number; silently no-ops without a token).
    fetch_all_discussion_signals(submissions)

    # ----- Set up Jinja2 -----
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    base_ctx = {
        "submission_count": len(submissions),
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
    }

    # ----- Render index -----
    index_html = env.get_template("index.html").render(
        submissions=submissions,
        css_path="style.css",
        code_css_path="code.css",
        logo_path="nengozoo-logo.svg",
        home_path="index.html",
        contributing_path="contributing.html",
        **base_ctx,
    )
    (out_root / "index.html").write_text(index_html)

    # ----- Render per-submission pages + copy figures -----
    for s in submissions:
        sub_out = out_root / "submissions" / s["name"]
        sub_out.mkdir(parents=True, exist_ok=True)

        # Copy figures into the rendered site and rewrite URLs.
        figures_rendered = []
        if s["figures_src"]:
            fig_out_dir = sub_out / "figures"
            fig_out_dir.mkdir(exist_ok=True)
            for f in s["figures_src"]:
                shutil.copy2(f["src"], fig_out_dir / f["filename"])
                figures_rendered.append({"name": f["name"], "url": f"figures/{f['filename']}"})

        # Build a downloadable zip of the submission folder.
        zip_name = f"{s['name']}-{s['version']}.zip"
        zip_path = sub_out / zip_name
        zip_size = make_submission_zip(s["sub_dir"], s["version"], zip_path)
        download = {"filename": zip_name, "size": format_size(zip_size)}

        readme_html = markdown.markdown(s["readme_md"], extensions=MARKDOWN_EXTENSIONS)

        # Build a render context with paths relative to this page's location.
        ctx = {
            **s,
            "figures": figures_rendered,
            "readme_html": readme_html,
            "download": download,
            "css_path": "../../style.css",
            "code_css_path": "../../code.css",
            "logo_path": "../../nengozoo-logo.svg",
            "home_path": "../../index.html",
            "contributing_path": "../../contributing.html",
            **base_ctx,
        }
        html = env.get_template("submission.html").render(s=ctx, **ctx)
        (sub_out / "index.html").write_text(html)

    # ----- Copy CSS + logo to site root -----
    shutil.copy2(CSS_SOURCE, out_root / "style.css")
    logo_source = ASSETS_DIR / "nengozoo-logo.svg"
    if logo_source.exists():
        shutil.copy2(logo_source, out_root / "nengozoo-logo.svg")

    # ----- Render CONTRIBUTING.md as a static page -----
    contributing_src = REPO_ROOT / "CONTRIBUTING.md"
    if contributing_src.exists():
        contributing_html = markdown.markdown(
            contributing_src.read_text(encoding="utf-8"), extensions=MARKDOWN_EXTENSIONS
        )
        page = env.get_template("page.html").render(
            page_title="Submit to NengoZoo",
            page_content=contributing_html,
            css_path="style.css",
            code_css_path="code.css",
            logo_path="nengozoo-logo.svg",
            home_path="index.html",
            contributing_path="contributing.html",
            **base_ctx,
        )
        (out_root / "contributing.html").write_text(page)

    # ----- Generate Pygments syntax-highlighting CSS -----
    pyg_css = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    (out_root / "code.css").write_text(
        "/* Pygments syntax highlighting (theme: friendly) */\n" + pyg_css + "\n"
    )

    print()
    print(f"Built {len(submissions)} submission page{'' if len(submissions)==1 else 's'} → {out_root}")
    print(f"Open: {out_root / 'index.html'}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="site", help="Output directory (default: site/)")
    args = parser.parse_args()

    out_root = REPO_ROOT / args.out
    sys.exit(render(out_root))


if __name__ == "__main__":
    main()

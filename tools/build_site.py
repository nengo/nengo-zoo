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
import subprocess
import sys
import urllib.error
import urllib.request
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

# Reuse the validator's GUI-shape check so the badges agree, and the shared
# zip helper so the site download matches what Zenodo archives.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_submission import check_gui_shape  # noqa: E402
from submission_zip import make_submission_zip  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
ASSETS_DIR = Path(__file__).resolve().parent / "site_assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"
CSS_SOURCE = ASSETS_DIR / "style.css"

MARKDOWN_EXTENSIONS = ["fenced_code", "tables", "sane_lists", "codehilite"]


def format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f} {unit}" if unit == "B" else f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


def version_sort_key(vstr: str):
    """Numeric sort key for a semver-ish string ('0.10.0' > '0.9.0')."""
    core = str(vstr).split("-")[0]
    try:
        return tuple(int(x) for x in core.split("."))
    except ValueError:
        return (0,)


# --- Versions / per-version DOIs + zips ------------------------------------
# The version LIST and the downloadable zips come from git tags
# (<name>-v<version>, created by auto-tag.yml) — the durable, hard-to-fumble
# record of what was released. Per-version DOIs are looked up from Zenodo at
# build time (published records are public, so no token needed) and joined to
# the tags by version string. If Zenodo is unreachable, versions + zips still
# render; only the DOI links blink out until the next build.

ZENODO_DEFAULT_BASE = "https://sandbox.zenodo.org/api"


def fetch_concept_versions(record_recid: int, base: str) -> dict[str, str]:
    """Return {version: doi} for every published version sharing a concept
    with the given record, via Zenodo's /records/{id}/versions endpoint.
    (The legacy `conceptrecid:` search query returns nothing on the current
    InvenioRDM Zenodo, so we use the versions endpoint instead.) Empty dict
    on any failure — caller degrades gracefully."""
    # Bare endpoint — no ?size= param. The current Zenodo rejects size=100 with
    # a 400, and the default page comfortably covers our handful of versions
    # per submission. (Paginate here only if a submission ever exceeds it.)
    url = f"{base.rstrip('/')}/records/{record_recid}/versions"
    req = urllib.request.Request(url, headers={"User-Agent": "nengozoo-build-site",
                                               "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:  # subclass of URLError — catch first
        detail = e.read().decode("utf-8", errors="replace")[:200].replace("\n", " ")
        print(f"  WARN: Zenodo version lookup failed for record {record_recid} "
              f"(HTTP {e.code}: {detail})")
        return {}
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        print(f"  WARN: Zenodo version lookup failed for record {record_recid} ({e})")
        return {}

    out: dict[str, str] = {}
    for hit in (body.get("hits", {}).get("hits") or []):
        meta = hit.get("metadata") or {}
        version = meta.get("version")
        # DOI: top-level "doi", or metadata.doi, or pids.doi.identifier.
        doi = (hit.get("doi") or meta.get("doi")
               or (hit.get("pids", {}).get("doi") or {}).get("identifier"))
        if version and doi:
            out[str(version)] = doi
    return out


def fetch_all_zenodo_versions(submissions: list[dict]) -> None:
    """Attach a {version: doi} map to each submission as 'zenodo_doi_map'.
    Gated on ZENODO_BASE_URL being set, mirroring the discussions fetch's
    token gate — so local builds without it are fast and offline."""
    for s in submissions:
        s["zenodo_doi_map"] = {}

    base = os.environ.get("ZENODO_BASE_URL")
    if not base:
        print("  (no ZENODO_BASE_URL set; skipping per-version DOI lookup)")
        return
    # The versions endpoint is keyed off a concrete version record id; use the
    # latest (version_recid), falling back to concept_recid.
    with_records = [s for s in submissions
                    if (s.get("zenodo") or {}).get("version_recid")
                    or (s.get("zenodo") or {}).get("concept_recid")]
    if not with_records:
        return
    print(f"  fetching Zenodo version DOIs for {len(with_records)} submission(s)…")
    for s in with_records:
        zen = s["zenodo"]
        rec = zen.get("version_recid") or zen.get("concept_recid")
        s["zenodo_doi_map"] = fetch_concept_versions(rec, base)


def submission_tag_versions(name: str) -> list[str]:
    """Versions released for a submission, from its <name>-v* git tags."""
    prefix = f"{name}-v"
    try:
        raw = subprocess.check_output(
            ["git", "tag", "--list", f"{prefix}*"],
            cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    versions = [t[len(prefix):] for t in raw.split() if t.startswith(prefix)]
    return sorted(set(versions), key=version_sort_key, reverse=True)


def build_version_zip(name: str, version: str, out_path: Path) -> int | None:
    """Archive a submission's subtree at its version tag into out_path.
    Uses `git archive`, so only tracked files are included (correct for a
    released snapshot). Returns byte size, or None if the tag/subtree is
    missing."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    treeish = f"{name}-v{version}:submissions/{name}"
    try:
        with out_path.open("wb") as fh:
            subprocess.check_call(
                ["git", "archive", "--format=zip",
                 f"--prefix={name}-{version}/", treeish],
                cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.DEVNULL,
            )
    except subprocess.CalledProcessError:
        if out_path.exists():
            out_path.unlink()
        print(f"  WARN: could not archive {name} v{version} (tag/subtree missing)")
        return None
    return out_path.stat().st_size


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
        # Zenodo DOI block (bot-managed): concept_doi, version_doi, version, …
        "zenodo":         meta.get("zenodo"),
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

    # ----- Pull per-version DOIs from Zenodo (gated on ZENODO_BASE_URL).
    fetch_all_zenodo_versions(submissions)

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
        favicon_path="favicon.svg",
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

        # Build the Versions list: git tags are the authoritative set of
        # released versions; archive each into versions/<name>-<ver>.zip and
        # join the per-version DOI (Zenodo lookup, plus the top-level latest as
        # a fallback so the newest version always has its DOI).
        doi_map = dict(s.get("zenodo_doi_map") or {})
        zen = s.get("zenodo") or {}
        if zen.get("version") and zen.get("version_doi"):
            doi_map.setdefault(str(zen["version"]), zen["version_doi"])

        versions = []
        for ver in submission_tag_versions(s["name"]):
            vzip = f"{s['name']}-{ver}.zip"
            vsize = build_version_zip(s["name"], ver, sub_out / "versions" / vzip)
            versions.append({
                "version": ver,
                "doi": doi_map.get(ver),
                "zip_url": f"versions/{vzip}" if vsize is not None else None,
                "zip_size": format_size(vsize) if vsize is not None else None,
            })

        # The prominent Download button serves the latest *released* (tagged)
        # artifact — the very same file the Versions dropdown links, so a given
        # version has one canonical zip. Only when there's no usable tag yet
        # (e.g. a brand-new submission auto-tag hasn't processed) do we fall
        # back to a working-tree zip so the button still works.
        if versions and versions[0].get("zip_url"):
            latest = versions[0]
            download = {
                "name": f"{s['name']}-{latest['version']}.zip",
                "href": latest["zip_url"],
                "size": latest["zip_size"],
            }
        else:
            zip_name = f"{s['name']}-{s['version']}.zip"
            zip_size = make_submission_zip(s["sub_dir"], s["version"], sub_out / zip_name)
            download = {"name": zip_name, "href": zip_name, "size": format_size(zip_size)}

        readme_html = markdown.markdown(s["readme_md"], extensions=MARKDOWN_EXTENSIONS)

        # Build a render context with paths relative to this page's location.
        ctx = {
            **s,
            "figures": figures_rendered,
            "readme_html": readme_html,
            "download": download,
            "versions": versions,
            "css_path": "../../style.css",
            "code_css_path": "../../code.css",
            "logo_path": "../../nengozoo-logo.svg",
            "favicon_path": "../../favicon.svg",
            "home_path": "../../index.html",
            "contributing_path": "../../contributing.html",
            **base_ctx,
        }
        html = env.get_template("submission.html").render(s=ctx, **ctx)
        (sub_out / "index.html").write_text(html)

    # ----- Copy CSS + logo + favicon + CNAME to site root -----
    shutil.copy2(CSS_SOURCE, out_root / "style.css")
    logo_source = ASSETS_DIR / "nengozoo-logo.svg"
    if logo_source.exists():
        shutil.copy2(logo_source, out_root / "nengozoo-logo.svg")
    favicon_source = ASSETS_DIR / "favicon.svg"
    if favicon_source.exists():
        shutil.copy2(favicon_source, out_root / "favicon.svg")
    # GitHub Pages custom-domain marker. The site deploys via Actions
    # (upload-pages-artifact), so the domain must travel inside the artifact —
    # there's no gh-pages branch to hold it. Copied only if present, which keeps
    # the build portable: drop a site_assets/CNAME with your domain to enable a
    # custom domain, or delete it to fall back to the *.github.io default.
    cname_source = ASSETS_DIR / "CNAME"
    if cname_source.exists():
        shutil.copy2(cname_source, out_root / "CNAME")

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
            favicon_path="favicon.svg",
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

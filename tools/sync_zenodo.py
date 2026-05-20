#!/usr/bin/env python3
"""
Mint a Zenodo DOI for each submission whose current version hasn't been
deposited yet, and write the resulting DOIs back into its metadata.yaml.

Per-submission DOIs (not the repo-scoped native GitHub integration, which
would archive the whole monorepo as one record). Each submission gets its own
Zenodo deposition lineage: a stable concept DOI plus a version DOI per release.

Idempotent: a submission is (re)deposited only when its top-level `version`
differs from the `version` recorded in its `zenodo` block — so re-running, or
running over unchanged submissions, mints nothing. That's what makes the
workflow safe to chain after auto-tag and to trigger manually for backfill.

Environment:
    ZENODO_TOKEN      required — a Zenodo personal access token with
                      deposit:write + deposit:actions
    ZENODO_BASE_URL   API base; default https://sandbox.zenodo.org/api
                      (use https://zenodo.org/api for production)

Exit codes:
    0  success (including "nothing to deposit")
    1  one or more depositions failed
    2  misconfiguration (no token)

Usage:
    python tools/sync_zenodo.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. `pip install pyyaml`")
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submission_zip import make_submission_zip  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
DEFAULT_BASE_URL = "https://sandbox.zenodo.org/api"


class ZenodoError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Thin REST client
# --------------------------------------------------------------------------

def _request(method: str, url: str, token: str,
             json_body: dict | None = None, data: bytes | None = None,
             extra_headers: dict | None = None) -> dict:
    """Make one authenticated request; return parsed JSON (or {}). Raises
    ZenodoError with the response body on any HTTP error so failures are
    debuggable in the workflow log."""
    headers = {"Authorization": f"Bearer {token}"}
    body: bytes | None = None
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif data is not None:
        body = data
        headers["Content-Type"] = "application/octet-stream"
    if extra_headers:
        headers.update(extra_headers)

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise ZenodoError(f"{method} {url} -> HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise ZenodoError(f"{method} {url} -> {e}") from None


# --------------------------------------------------------------------------
# Metadata mapping
# --------------------------------------------------------------------------

def valid_orcid(orcid: str) -> bool:
    """Validate an ORCID's ISO 7064 Mod 11-2 check digit. Zenodo rejects
    syntactically-fine-but-checksum-invalid ORCIDs (e.g. placeholder
    0000-0000-0000-0000) at publish time, so we screen them out first."""
    digits = orcid.replace("-", "").strip()
    if len(digits) != 16 or not digits[:15].isdigit():
        return False
    total = 0
    for d in digits[:15]:
        total = (total + int(d)) * 2
    result = (12 - (total % 11)) % 11
    expected = "X" if result == 10 else str(result)
    return digits[15].upper() == expected


def zenodo_metadata(meta: dict) -> dict:
    """Build Zenodo deposition metadata from a submission's metadata.yaml."""
    creators = []
    for a in meta.get("authors", []) or []:
        c = {"name": a.get("name", "Unknown")}
        if a.get("affiliation"):
            c["affiliation"] = a["affiliation"]
        orcid = a.get("orcid")
        if orcid and valid_orcid(orcid):
            c["orcid"] = orcid
        elif orcid:
            # Don't let a bad optional field block the DOI — drop it, but say so.
            print(f"  WARN: dropping invalid ORCID {orcid!r} for {c['name']!r}")
        creators.append(c)
    if not creators:
        creators = [{"name": "NengoZoo"}]

    description = (meta.get("description") or meta["name"]).strip() or meta["name"]
    md = {
        "title": f"{meta['name']} (NengoZoo submission) v{meta['version']}",
        "upload_type": "software",
        "description": description,
        "creators": creators,
        "version": str(meta["version"]),
        "keywords": meta.get("tags", []) or [],
        "access_right": "open",
    }
    # Zenodo (InvenioRDM) expects SPDX license ids in lower case. Best-effort:
    # if the id is wrong the metadata PUT will fail loudly and we adjust.
    if meta.get("license"):
        md["license"] = str(meta["license"]).lower()
    return md


# --------------------------------------------------------------------------
# Deposition flows
# --------------------------------------------------------------------------

def _replace_files(base: str, dep: dict, zip_path: Path, token: str) -> None:
    """Remove any files already on the deposition (a new-version draft inherits
    the previous version's files) and upload the freshly-built zip."""
    dep_id = dep["id"]
    for f in dep.get("files", []) or []:
        _request("DELETE", f"{base}/deposit/depositions/{dep_id}/files/{f['id']}", token)
    bucket = dep["links"]["bucket"]
    _request("PUT", f"{bucket}/{zip_path.name}", token, data=zip_path.read_bytes())


def deposit(base: str, token: str, meta: dict, zip_path: Path,
            prev_recid: int | None) -> dict:
    """Create (or version) a deposition, upload the zip, set metadata, publish.
    Returns the published record dict."""
    if prev_recid:
        # New version of an existing concept: cut a draft off the last version.
        resp = _request("POST",
                        f"{base}/deposit/depositions/{prev_recid}/actions/newversion",
                        token)
        draft_url = resp["links"]["latest_draft"]
        dep = _request("GET", draft_url, token)
    else:
        dep = _request("POST", f"{base}/deposit/depositions", token, json_body={})

    dep_id = dep["id"]
    _replace_files(base, dep, zip_path, token)
    _request("PUT", f"{base}/deposit/depositions/{dep_id}", token,
             json_body={"metadata": zenodo_metadata(meta)})
    return _request("POST", f"{base}/deposit/depositions/{dep_id}/actions/publish", token)


# --------------------------------------------------------------------------
# metadata.yaml write-back (text-based, preserves comments/formatting)
# --------------------------------------------------------------------------

def write_zenodo_block(meta_path: Path, block: dict) -> None:
    """Replace (or append) the top-level `zenodo:` block in metadata.yaml.
    Text-based so existing comments and field ordering survive — a yaml.dump
    round-trip would discard the template's inline comments."""
    lines = meta_path.read_text(encoding="utf-8").splitlines()

    # Strip any pre-existing `zenodo:` block (the key line + indented children).
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("zenodo:"):
            block_form = not line[len("zenodo:"):].strip()
            i += 1
            if block_form:
                while i < n and lines[i].startswith((" ", "\t")):
                    i += 1
            continue
        out.append(line)
        i += 1

    text = "\n".join(out).rstrip("\n") + "\n"
    text += "\nzenodo:\n"
    text += f"  concept_recid: {block['concept_recid']}\n"
    text += f"  concept_doi: \"{block['concept_doi']}\"\n"
    text += f"  version_doi: \"{block['version_doi']}\"\n"
    text += f"  version_recid: {block['version_recid']}\n"
    text += f"  version: \"{block['version']}\"\n"
    meta_path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> int:
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ERROR: ZENODO_TOKEN not set")
        return 2
    base = os.environ.get("ZENODO_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    print(f"Zenodo base: {base}")

    deposited, skipped, failed = 0, 0, 0
    for meta_path in sorted(SUBMISSIONS_DIR.glob("*/metadata.yaml")):
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        sub_name = meta.get("name", meta_path.parent.name)
        version = str(meta["version"])
        zen = meta.get("zenodo") or {}

        # Already deposited this exact version?
        if zen.get("version") == version and zen.get("version_doi"):
            print(f"  skip: {sub_name} v{version} already on Zenodo "
                  f"({zen['version_doi']})")
            skipped += 1
            continue

        prev_recid = zen.get("version_recid")
        action = "new version" if prev_recid else "first deposit"
        print(f"  depositing: {sub_name} v{version} ({action})")

        try:
            with tempfile.TemporaryDirectory() as td:
                zip_path = Path(td) / f"{sub_name}-{version}.zip"
                make_submission_zip(meta_path.parent, version, zip_path)
                record = deposit(base, token, meta, zip_path, prev_recid)
        except (ZenodoError, KeyError) as e:
            print(f"  FAIL: {sub_name}: {e}")
            failed += 1
            continue

        block = {
            "concept_recid": record.get("conceptrecid"),
            "concept_doi": record.get("conceptdoi"),
            "version_doi": record.get("doi"),
            "version_recid": record.get("id"),
            "version": version,
        }
        if not (block["version_doi"] and block["version_recid"]):
            print(f"  FAIL: {sub_name}: published but response missing DOI/id: {record}")
            failed += 1
            continue

        write_zenodo_block(meta_path, block)
        print(f"  deposited: {sub_name} v{version} → {block['version_doi']} "
              f"(concept {block['concept_doi']})")
        deposited += 1

    print(f"\nDone. deposited={deposited} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

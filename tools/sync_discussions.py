#!/usr/bin/env python3
"""
Ensure every submission has a GitHub Discussion thread, and write the
resulting discussion number back into its metadata.yaml.

For each submissions/*/metadata.yaml that lacks a `discussion:` field, this
creates a Discussion in the target category (default "Submissions") via the
GitHub GraphQL API and appends `discussion: <number>` to the file. Submissions
that already carry a number are left untouched, so the script is idempotent —
running it repeatedly only ever creates threads for the ones still missing one.

That idempotence is what lets a single manual run (workflow_dispatch) backfill
every existing submission, while the same script on a post-merge push handles
new submissions one at a time.

Environment:
    GITHUB_TOKEN          required — needs `discussions: write` on the repo
    GITHUB_REPOSITORY     "owner/name" (set automatically inside Actions)
    DISCUSSION_CATEGORY   category name to create threads in (default: Submissions)

Exit codes:
    0  success (including "nothing to do")
    1  one or more discussions failed to create
    2  misconfiguration (no token, repo, or category not found)

Usage:
    python tools/sync_discussions.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. `pip install pyyaml`")
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

# Look up the repo node id and all discussion categories in one round-trip.
REPO_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    id
    discussionCategories(first: 50) {
      nodes { id name }
    }
  }
}
"""

CREATE_MUTATION = """
mutation($repoId: ID!, $categoryId: ID!, $title: String!, $body: String!) {
  createDiscussion(input: {
    repositoryId: $repoId, categoryId: $categoryId, title: $title, body: $body
  }) {
    discussion { number url }
  }
}
"""

# updateDiscussion needs the discussion's node id, not its number — so we
# resolve the id by number first, then update the body. Used only by the
# --refresh-bodies path.
DISCUSSION_ID_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    discussion(number: $number) { id }
  }
}
"""

UPDATE_MUTATION = """
mutation($id: ID!, $body: String!) {
  updateDiscussion(input: { discussionId: $id, body: $body }) {
    discussion { number url }
  }
}
"""


def graphql(query: str, variables: dict, token: str) -> dict:
    """POST a GraphQL query; return the `data` block. Raises on transport or
    GraphQL-level errors so callers can decide per-call how to handle them."""
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    req = urllib.request.Request(
        GITHUB_GRAPHQL_URL,
        data=payload,
        headers={
            "Authorization": f"bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "nengozoo-sync-discussions",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    if body.get("errors"):
        raise RuntimeError(body["errors"][0].get("message", str(body["errors"])))
    return body["data"]


def discussion_body(name: str, description: str) -> str:
    """The top-level post for a submission's thread."""
    desc = description.strip()
    return (
        f"Community thread for the **{name}** submission.\n\n"
        + (f"{desc}\n\n" if desc else "")
        + "Upvote this thread (the ↑ arrow at the top) to “star” the "
        "submission — that's the count shown on the zoo site. Reply below "
        "with questions, results, or suggestions.\n\n"
        "_This thread was created automatically when the submission was merged._"
    )


def append_discussion_field(meta_path: Path, number: int) -> None:
    """Append `discussion: <number>` to a metadata.yaml without disturbing the
    rest of the file. YAML mappings are unordered, so a trailing top-level key
    is valid — and a textual append preserves all existing comments/formatting
    that a yaml.safe_dump round-trip would destroy."""
    text = meta_path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    text += f"discussion: {number}\n"
    meta_path.write_text(text, encoding="utf-8")


def refresh_body(owner: str, name: str, number: int,
                 sub_name: str, description: str, token: str) -> str:
    """Rewrite an existing discussion's body to the current canonical text.
    Resolves the node id by number, then runs updateDiscussion. Returns the
    thread URL. Raises on failure so the caller can count it."""
    data = graphql(DISCUSSION_ID_QUERY,
                   {"owner": owner, "name": name, "number": number}, token)
    disc = (data.get("repository") or {}).get("discussion")
    if not disc:
        raise RuntimeError(f"discussion #{number} not found")
    body = discussion_body(sub_name, description)
    result = graphql(UPDATE_MUTATION, {"id": disc["id"], "body": body}, token)
    return result["updateDiscussion"]["discussion"]["url"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-bodies", action="store_true",
        help="Also rewrite the body text of submissions that ALREADY have a "
             "discussion, bringing them in line with the current template. "
             "Use after changing the body wording. Overwrites manual edits.",
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN not set")
        return 2

    slug = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in slug:
        print("ERROR: GITHUB_REPOSITORY not set (expected 'owner/name')")
        return 2
    owner, name = slug.split("/", 1)

    category_name = os.environ.get("DISCUSSION_CATEGORY", "Submissions")

    # Read every submission's metadata once, then split into work buckets.
    metas = [(p, yaml.safe_load(p.read_text(encoding="utf-8")))
             for p in sorted(SUBMISSIONS_DIR.glob("*/metadata.yaml"))]
    to_create = [(p, m) for p, m in metas if not m.get("discussion")]
    to_refresh = [(p, m) for p, m in metas if m.get("discussion")] \
        if args.refresh_bodies else []

    created, refreshed, failed, skipped = 0, 0, 0, 0

    # --- Create threads for submissions that don't have one yet -------------
    if to_create:
        # Category id is only needed when we actually create something.
        try:
            data = graphql(REPO_QUERY, {"owner": owner, "name": name}, token)
        except (urllib.error.URLError, RuntimeError, KeyError) as e:
            print(f"ERROR: could not query repository: {e}")
            return 2
        repo = data["repository"]
        repo_id = repo["id"]
        categories = {c["name"]: c["id"] for c in repo["discussionCategories"]["nodes"]}
        if category_name not in categories:
            print(f"ERROR: discussion category {category_name!r} not found. "
                  f"Available: {sorted(categories)}")
            return 2
        category_id = categories[category_name]

        for meta_path, meta in to_create:
            sub_name = meta.get("name", meta_path.parent.name)
            body = discussion_body(sub_name, meta.get("description", ""))
            try:
                result = graphql(CREATE_MUTATION, {
                    "repoId": repo_id, "categoryId": category_id,
                    "title": sub_name, "body": body,
                }, token)
                disc = result["createDiscussion"]["discussion"]
                number, url = disc["number"], disc["url"]
            except (urllib.error.URLError, RuntimeError, KeyError) as e:
                print(f"  FAIL (create): {sub_name}: {e}")
                failed += 1
                continue
            append_discussion_field(meta_path, number)
            print(f"  created: {sub_name} → discussion #{number} ({url})")
            created += 1
    else:
        print("  (no submissions need a new discussion)")

    # --- Optionally rewrite existing thread bodies --------------------------
    for meta_path, meta in to_refresh:
        sub_name = meta.get("name", meta_path.parent.name)
        number = meta["discussion"]
        try:
            url = refresh_body(owner, name, number, sub_name,
                               meta.get("description", ""), token)
        except (urllib.error.URLError, RuntimeError, KeyError) as e:
            print(f"  FAIL (refresh): {sub_name} #{number}: {e}")
            failed += 1
            continue
        print(f"  refreshed: {sub_name} → discussion #{number} ({url})")
        refreshed += 1

    print(f"\nDone. created={created} refreshed={refreshed} "
          f"skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

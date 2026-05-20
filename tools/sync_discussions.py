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
        + "👍 react to this top post to “star” the submission. "
        "Reply below with questions, results, or suggestions.\n\n"
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


def main() -> int:
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

    # Resolve repo id + the target category id.
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

    # Walk submissions; create a thread for any that lack a discussion number.
    created, failed, skipped = 0, 0, 0
    for meta_path in sorted(SUBMISSIONS_DIR.glob("*/metadata.yaml")):
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        sub_name = meta.get("name", meta_path.parent.name)
        if meta.get("discussion"):
            print(f"  skip: {sub_name} already has discussion #{meta['discussion']}")
            skipped += 1
            continue

        title = sub_name
        body = discussion_body(sub_name, meta.get("description", ""))
        try:
            result = graphql(CREATE_MUTATION, {
                "repoId": repo_id, "categoryId": category_id,
                "title": title, "body": body,
            }, token)
            disc = result["createDiscussion"]["discussion"]
            number, url = disc["number"], disc["url"]
        except (urllib.error.URLError, RuntimeError, KeyError) as e:
            print(f"  FAIL: {sub_name}: {e}")
            failed += 1
            continue

        append_discussion_field(meta_path, number)
        print(f"  created: {sub_name} → discussion #{number} ({url})")
        created += 1

    print(f"\nDone. created={created} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

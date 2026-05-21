# CLAUDE.md

Guidance for Claude (and other agents) working in this repo. Two common modes:
**creating/editing a submission**, and **working on the tooling**. Find your
mode below.

## What this is

NengoZoo is a community library of [Nengo](https://www.nengo.ai/) models,
components, and networks. Each entry lives in `submissions/<name>/`. CI
validates and tests every submission; a static site is generated from
`submissions/` and deployed to GitHub Pages; merged versions are tagged, get a
GitHub Discussion thread, and are minted a Zenodo DOI — all automatically.

## Repo layout

- `submissions/<name>/` — one folder per submission (the content).
- `template/` — copy this to start a new submission.
- `schema/metadata.schema.json` — the contract every `metadata.yaml` must meet.
- `tools/` — `build_site.py` (static site), `validate_submission.py` (Tier-1
  checks), `sync_discussions.py` + `sync_zenodo.py` (merge-time bots),
  `submission_zip.py` (shared zip helper), `run_matrix.py` (local test matrix).
- `.github/workflows/` — `review.yml` (validate + test matrix + version-bump
  enforcement), `pages.yml` (build + deploy, plus build-time discussion/DOI
  fetches), `auto-tag.yml`, `auto-discussion.yml`, `zenodo.yml`.
- `site/` — generated output; never edit by hand.

---

## Mode A — creating or editing a submission

Read `CONTRIBUTING.md` for the full walkthrough; the essentials:

- Start from `template/`: `cp -r template submissions/<kebab-name>`.
- **Three types** (`metadata.yaml` `type:`): `component`, `network`, `model`.
  `component`/`network` need a `src/` library package; a `model` (often a
  single NengoGUI script) does not.
- **`metadata.yaml` rules** (enforced by the schema + validator):
  - `name` must be lowercase kebab-case, must **equal the folder name**, and
    must be **unique** across `submissions/`.
  - Bump `version:` on *any* change to an existing submission (CI fails a
    changed submission whose version didn't move).
  - Do **not** hand-edit the bot-managed blocks: `discussion:` and `zenodo:`
    are written by workflows on merge.
  - List only `backends` you've actually tested; CI verifies `core` only.
  - Heavy non-core deps (e.g. nengo_dl + GPU)? Set `ci_runnable: false` — CI
    then does structure/schema checks only.
- **Required files**: `README.md`, `metadata.yaml`, `LICENSE`,
  `requirements.txt`, `tests/test_runs.py` (plus `src/` for component/network).
- Keep the entry point and tests fast (well under ~60s) — CI runs them.
- **Validate before opening a PR**:
  `python tools/validate_submission.py submissions/<name>` → want `Result: PASS`.

---

## Mode B — working on the tooling

Build the site locally: `python tools/build_site.py` (writes `site/`). The
discussion-count and per-version-DOI fetches are skipped unless `GITHUB_TOKEN`
and `ZENODO_BASE_URL` are set, so a bare local build is fast and offline.

The pipeline is repo-as-source-of-truth: git tags are the authoritative list of
released versions; metadata blocks (`discussion:`, `zenodo:`) are bot-written
caches; the site is a periodically-rebuilt snapshot.

### Gotchas (non-obvious, learned the hard way)

- **CI diffs need full history.** Any workflow step that does
  `git show <base_sha>:...` or diffs against the PR base needs
  `fetch-depth: 0` on its checkout — a shallow clone makes the base commit
  unreachable, and checks silently mis-pass.
- **Git pathspecs are relative to `working-directory`.** If a step sets
  `working-directory: submissions/<x>`, anchor pathspecs to the repo root with
  the `:/` prefix (e.g. `:/submissions/<x>/`) or they match nothing.
- **The built-in `GITHUB_TOKEN` does not trigger other workflows.** Pushes/tags
  it makes won't fire downstream `push`/`pull_request` runs — chain via
  `workflow_run` (that fires on workflow *completion*, regardless of token).
- **`GITHUB_TOKEN` can create but not edit Discussions.** `createDiscussion`
  works; `updateDiscussion` returns "Resource not accessible by integration" —
  needs a PAT. (That's why `sync_discussions.py --refresh-bodies` is manual.)
- **`actions/upload-artifact@v4` skips dot-directories** unless
  `include-hidden-files: true`.
- **Zenodo (InvenioRDM):** list a concept's versions via
  `/api/records/{recid}/versions` — the legacy `conceptrecid:` *search* query
  returns nothing. Don't append `?size=` (it 400s). ORCIDs must pass the
  ISO-7064 checksum or the deposit is rejected (we drop invalid ones).
  `license` must be a lowercase SPDX id. DOIs currently point at **sandbox**;
  flip `ZENODO_BASE_URL` in *both* `zenodo.yml` and `pages.yml` at production.

### Conventions

- Text-edit `metadata.yaml` to preserve comments — never `yaml.dump` round-trip
  it (the bots append/replace blocks textually for this reason).
- The site download and the Versions-dropdown zip for a version are the same
  `git archive` artifact; keep them unified if you touch zip building.

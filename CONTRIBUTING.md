# Submitting to NengoZoo

Thanks for wanting to share a Nengo model, network, or component. This page walks you through the whole submission process. It's designed to be doable in an afternoon, even if you've never opened a pull request before.

If you get stuck on anything, open an issue on this repo — it's a sign the docs need fixing, not a sign you did something wrong.

---

## TL;DR

1. **Fork** this repo on GitHub.
2. **Copy** `template/` to `submissions/<your-name>/`.
3. **Fill in** the files (mostly `metadata.yaml`, `README.md`, and your code).
4. **Validate** locally: `python tools/validate_submission.py submissions/<your-name>`.
5. **Open a pull request.** CI runs, a maintainer reviews, and it's in.

The rest of this page explains each step.

---

## What kind of submissions are accepted?

Three flavors. Pick the one that fits best — a maintainer can re-tag during review if you choose differently.

| Type        | What it is                                            | Example                              |
|-------------|-------------------------------------------------------|--------------------------------------|
| `component` | A small reusable building block.                      | A custom neuron model, learning rule, solver, synapse, or process. |
| `network`   | A reusable subnetwork that drops into a parent sim.    | A basal-ganglia network, working memory, action selection. |
| `model`     | A complete, runnable model — often paper-linked.       | Lamprey locomotion, an MNIST convnet, a Lorenz attractor. |

If your submission is meant to be `import`-ed into someone else's network, it's a *component* or *network*. If it's meant to be run end-to-end (especially in NengoGUI), it's a *model*.

---

## Step 1 — Fork and clone

On GitHub, click **Fork** at the top right of [`nengo/nengo-zoo`](#) (placeholder URL — replace once the zoo is live). Then on your machine:

```bash
git clone https://github.com/<your-username>/nengo-zoo
cd nengo-zoo
git checkout -b add-<your-submission-name>
```

The branch name is just a hint to reviewers; anything reasonable works.

## Step 2 — Copy the template

Pick a kebab-case name for your submission (lowercase letters, digits, dashes — no underscores, no spaces). It must match the folder you create.

```bash
cp -r template submissions/working-memory     # ← whatever your name is
cd submissions/working-memory
```

You'll see this layout:

```
working-memory/
├── README.md             ← write a description here
├── metadata.yaml         ← fill in version, authors, tags, etc.
├── LICENSE               ← GPLv2 by default; keep unless you have a reason
├── requirements.txt      ← what your code needs to run
├── src/my_submission/    ← rename `my_submission` to match your name (underscores OK in Python package name)
├── examples/             ← runnable demos
└── tests/                ← at minimum, one test that builds and runs 100ms
```

For a **model** that's a single NengoGUI script (no library API), you don't need `src/`. Put the script at the submission root (like `lorenz.py` does).

## Step 3 — Fill in `metadata.yaml`

This is the single most important file. CI validates it against [`schema/metadata.schema.json`](schema/metadata.schema.json), and the website renders the data on your submission's page.

```yaml
name: working-memory                 # MUST match folder name
type: network                        # component | network | model
version: 0.1.0                       # bump on every release
authors:
  - name: Jane Doe
    affiliation: University of Waterloo
    orcid: 0000-0000-0000-0000       # optional but encouraged
license: GPL-2.0-or-later
description: >
  A two-paragraph (max) description of what it does and what it's
  useful for. This shows up on the card and as the page subtitle.
tags:
  - working-memory                   # lowercase, hyphens only
  - attractor
nengo_version: ">=3.2,<5"            # PEP 440 spec
backends:                            # ones you've tested locally
  - core
complexity: intermediate             # beginner | intermediate | advanced
paper:                               # optional
  citation: "Doe et al. (2025). Title. Journal."
  doi: 10.1234/example
entry_point: examples/example_usage.py    # CI runs this
```

A few less-obvious fields:

- **`backends`**: list everything you've actually tested on (`core`, `nengo_dl`, `nengo_loihi`, `nengo_ocl`, etc.). CI only verifies `core`; the others are shown as "author-claimed" badges. Honesty is appreciated — broken claims show up fast.
- **`nengogui` block** (optional): if your library API has a NengoGUI demo script alongside it (like the lamprey submission does), declare it here. The site auto-awards a NengoGUI-ready badge.
- **`ci_runnable: false`** (optional escape hatch): set this if your submission needs heavy non-core backends (e.g. nengo_dl + TensorFlow + Loihi). CI will validate structure only, and the site will display a "CI structure-only" badge. Use sparingly.

## Step 4 — Write your code

Where it goes:

- **Library API** (`src/<your_pkg>/`): the Python module(s) people will `import`. Keep a clean public API — re-export the names you want surfaced in `src/<your_pkg>/__init__.py`.
- **NengoGUI script** (at submission root, or under `examples/nengogui/`): a top-level `with nengo.Network() as model:` block. Required if you're a `model`-type submission unless you've set `ci_runnable: false`.
- **Example** (`examples/example_usage.py` or whatever your `entry_point` is): a runnable demo. Should complete in under ~60 seconds wall-clock — CI runs it on every PR.
- **Tests** (`tests/test_runs.py`): at minimum, three tests — `test_imports`, `test_builds`, `test_runs_100ms`. See any existing submission for the pattern.

A few rules of thumb:

- **Keep simulations short.** If your model genuinely needs a long training run, put that in `examples/run_training.py` (gated behind `if __name__ == "__main__":` is fine) and keep the CI-tested `entry_point` quick.
- **No global side effects on import.** Don't print, don't open files, don't start servers when someone `import`s your module.
- **Pin reasonably.** `nengo>=3.2,<5` is a sensible default. If you need a specific Nengo version, say so.

## Step 5 — Write a good README

The README is what people see on the website. Use this rough structure:

1. **One-line tagline** right under the H1.
2. **Description** — 1–3 paragraphs. What does it do? What's the intuition? What's it useful for?
3. **Installation** — recommend a fresh venv (see existing submissions for a copy-paste-ready block).
4. **Usage** — a small code block showing how someone would use this.
5. **How it works** — the math / architecture. Be brief; link to the paper for details.
6. **Citation** — BibTeX block if there's a paper.

Don't worry about formatting perfection — the site auto-renders the markdown.

## Step 6 — Validate locally

Before opening the PR, run the validator yourself. It catches the same things CI will:

```bash
cd ../..   # back to the repo root
pip install pyyaml "jsonschema>=4.18"
python tools/validate_submission.py submissions/working-memory
```

You want to see `Result: PASS` at the end. If something fails, the message tells you what to fix (missing file, schema mismatch, name doesn't match folder, etc.).

If you want to test the example and tests too:

```bash
python -m venv .venv && source .venv/bin/activate
cd submissions/working-memory
pip install -r requirements.txt pytest
PYTHONPATH=src:. python examples/example_usage.py
PYTHONPATH=src:. pytest tests/
```

## Step 7 — Open the pull request

```bash
git add submissions/working-memory
git commit -m "add working-memory submission"
git push -u origin add-working-memory
```

Then open a PR on GitHub. CI will run automatically against multiple Nengo versions; you'll see green checkmarks once everything passes.

A maintainer (currently Terry Stewart, with delegates as the zoo grows) will review for correctness, fit, and quality. They might suggest small tweaks. Once approved, your submission is merged and appears on the site within a few minutes.

---

## After it's merged

- Your submission gets a permanent URL on the zoo site, a downloadable zip, and a "Tested on Nengo X.Y.Z" badge for each version CI verified.
- If you've enabled the Zenodo integration on your fork, tagged releases will mint a DOI you can cite.
- Future versions are easy: bump `version:` in metadata, update the code, open a follow-up PR.

## Common questions

**"What if my submission depends on something heavy like nengo_dl?"**
Set `ci_runnable: false` in metadata. CI will validate structure only, your submission gets a "CI structure-only" badge, and the README is where you explain what users need to install themselves. See `mnist-convnet` for an example.

**"Can I submit a Jupyter notebook?"**
Yes. Ship it as a file alongside your other artifacts (it'll go into the download zip). For the canonical entry_point, prefer a `.py` script — CI doesn't run notebooks currently. The LMU submission has both.

**"My submission's tests fail in CI but pass locally."**
Most often: stale Nengo decoder cache from a previous environment. Run `rm -rf ~/.cache/nengo` and try again.

**"How do I update an existing submission?"**
Bump `version:` in `metadata.yaml`, make your changes, open a PR. The old version's zip stays available; users can pin to it if they need to.

**"How do people 'star' my submission?"**
Each submission has a GitHub Discussions thread; the 👍 count on the top post is the star count. (Not enabled in the prototype yet; landing as part of the v1 launch.)

---

## License

By submitting, you agree your code can be distributed under the license you declared in `metadata.yaml`. Default is GPLv2, matching Nengo itself. Other GPL-compatible licenses are negotiable per submission.

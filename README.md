# NengoZoo

[![Tier-1 review](https://github.com/nengo/nengo-zoo/actions/workflows/review.yml/badge.svg)](https://github.com/nengo/nengo-zoo/actions/workflows/review.yml)
[![Pages deploy](https://github.com/nengo/nengo-zoo/actions/workflows/pages.yml/badge.svg)](https://github.com/nengo/nengo-zoo/actions/workflows/pages.yml)
[![License: GPL v2+](https://img.shields.io/badge/License-GPL%20v2%2B-blue.svg)](LICENSE)

A community library of [Nengo](https://www.nengo.ai/) models, components, and networks.

> **Browse the library:** [zoo.nengo.ai](https://zoo.nengo.ai/)
> **Want to contribute?** See [`CONTRIBUTING.md`](CONTRIBUTING.md).


## What's in here?

Every folder under `submissions/` is a self-contained Nengo model, network, or reusable component, with:

- A `metadata.yaml` describing it (authors, license, paper citation, tested Nengo versions, …)
- An `entry_point` script that CI runs on every PR
- A `LICENSE` (default GPLv2, matching Nengo)
- A short `README.md`
- Tests under `tests/`

Submissions are CI-validated against multiple Nengo versions; the resulting "Tested on Nengo X.Y" badges appear on each submission's page on the site.

## Repository layout

```
nengo-zoo/
├── submissions/                  # one folder per submission
│   ├── basal-ganglia/
│   ├── controlled-oscillator/
│   ├── lamprey/
│   ├── lmu/
│   ├── lorenz/
│   └── mnist-convnet/
├── template/                     # copy this when adding a new submission
├── schema/
│   └── metadata.schema.json      # JSON Schema for metadata.yaml
├── tools/
│   ├── validate_submission.py    # structural + schema validator
│   ├── build_site.py             # static-site generator
│   ├── run_matrix.py             # local Nengo-version matrix runner
│   └── site_assets/              # templates, CSS, logo
├── .github/workflows/
│   ├── review.yml                # Tier-1 CI (matrix on submission × nengo version)
│   └── pages.yml                 # builds & deploys the site to GitHub Pages
├── CONTRIBUTING.md               # how to submit
└── LICENSE                       # GPLv2
```

## Quick start (contributors)

```bash
git clone https://github.com/nengo/nengo-zoo
cd nengo-zoo
cp -r template submissions/<your-submission>
# fill in metadata.yaml, README.md, your code, tests…
pip install pyyaml "jsonschema>=4.18"
python tools/validate_submission.py submissions/<your-submission>
```

Full walkthrough: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quick start (browsing locally)

```bash
pip install pyyaml jinja2 markdown pygments nengo
python tools/build_site.py
open site/index.html
```

## Governance

NengoZoo is maintained by Terry Stewart and a rotating group of delegates from the Nengo / CNRG community. Submissions are reviewed for runnability, structural conformance, and fit; technical merit lives in the per-submission README and citation, not in gatekeeping.

See the [design doc](../nengo_hub_design.md) (in the brainstorming repo) for the full architectural reasoning.

## License

GPLv2 or later (see [`LICENSE`](LICENSE)), matching Nengo itself. Individual submissions may declare alternative GPL-compatible licenses in their `metadata.yaml`.

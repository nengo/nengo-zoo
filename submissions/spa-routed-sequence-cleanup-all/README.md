# spa-routed-sequence-cleanup-all

A spiking SPA model that cycles a `state` vector through a five-step sequence (A → B → C → D → E → A) under basal-ganglia / thalamus action selection, with a continuous-readout *cleanup* ensemble that lights up whichever vocabulary element the state currently matches.

## Description

This is example 4 of chapter 7 of *How to Build a Brain* (Eliasmith, 2013), distributed with nengo-gui as `4-spa_sequencerouted_cleanupAll.py`.

The base model is the routed-sequencing pattern from earlier in the chapter: a recurrent `state` SPA module holds a semantic pointer, and six basal-ganglia actions implement the transitions

```
dot(vision, START) --> state = vision
dot(state, A)      --> state = B
dot(state, B)      --> state = C
dot(state, C)      --> state = D
dot(state, D)      --> state = E
dot(state, E)      --> state = A
```

After a brief `vision = 0.8*START+D` pulse, the model picks up at `D` and cycles forever.

What this example adds is the *cleanup-all* read-out: a scalar `nengo.Ensemble` of 300 neurons whose `vsize`-dimensional input is the inner product of the live `state` output with every vocabulary vector. Each dimension `i` of the cleanup ensemble rises only when `state` is most similar to vocabulary item `i`. In the GUI's Value plot you see a moving bump that walks through the dimensions in sequence, providing an explicit "which symbol is loaded right now" trace alongside the SPA similarity plot.

(The previous chapter-7 example projects only the `A` direction onto a single scalar. This one generalizes to the whole vocabulary — hence "cleanup *all*".)

## A note on `nengo.spa` vs `nengo_spa`

This script uses the legacy `nengo.spa` module that still ships inside core Nengo. The newer, separately maintained `nengo_spa` package has a redesigned API. Both run today, but new submissions are generally encouraged to prefer `nengo_spa`; this submission preserves the original tutorial verbatim.

## Run it

In NengoGUI (the intended way to view it):

```bash
pip install nengo-gui
nengo spa_routed_sequence_cleanup_all.py
```

Or just at the command line:

```bash
python spa_routed_sequence_cleanup_all.py
```

## How it works

The cleanup transform is built explicitly rather than via SPA syntax. The script asks the state module for its output vocabulary in creation order, stacks each vocabulary vector into a row of a transform matrix `pd`, then attaches one `nengo.Connection(state.output, cleanup[i], transform=pd[i])` per vocabulary item. The result: `cleanup[i] ≈ state · vocab[i]`, which is exactly the cosine similarity used by the SPA cleanup memory.

Dimension 0 of the vocabulary is `START`, which is the role pointer used only by the first action — projecting it would just produce a constant signal during the kickstart pulse, so the loop deliberately starts from `i = 1`.

The `feedback=1, feedback_synapse=0.01` on the state module produces a near-perfect integrator over a 10 ms time-constant: long enough for each action to settle into the new symbol before the next BG transition fires, short enough that the cycle keeps moving.

## Citation

```bibtex
@book{eliasmith2013htb,
  author    = {Eliasmith, Chris},
  title     = {How to Build a Brain: A Neural Architecture for Biological Cognition},
  publisher = {Oxford University Press},
  year      = {2013}
}
```

## License

GPLv2 (see `LICENSE`).

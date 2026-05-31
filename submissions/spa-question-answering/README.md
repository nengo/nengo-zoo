# spa-question-answering

A spiking SPA model that binds two attributes (colour × shape) into a single semantic pointer and answers questions by unbinding with a cue. A small, visual introduction to circular-convolution binding in the Semantic Pointer Architecture.

## Description

This is example 2 of chapter 5 of *How to Build a Brain* (Eliasmith, 2013), distributed with nengo-gui as `2-spa_question.py`. It's the simplest of the chapter's question-answering models: pure cortical binding/unbinding with no memory and no action selection.

Five SPA `State`s — `A`, `B`, `C`, `D`, `E` — are connected by two cortical actions:

```
D = A * B          # bind colour (A) to shape (B)
E = D * ~C         # unbind cue (C) to recover the other attribute
```

`A` is driven with `RED`/`BLUE`, `B` with `CIRCLE`/`SQUARE`, and `C` cycles through cues (`ZERO`, `CIRCLE`, `RED`, `ZERO`, `SQUARE`, `BLUE`). When the cue is `CIRCLE` and the current bound pair is `RED * CIRCLE`, the readout `E` is most similar to `RED`; when the cue is `RED`, `E` is most similar to `CIRCLE`. `ZERO` is a null cue that produces nothing.

The model uses 32-dimensional semantic pointers — small enough to build quickly, big enough to keep similarities clean.

## A note on `nengo.spa` vs `nengo_spa`

This script uses the legacy `nengo.spa` module that still ships inside core Nengo. The newer, separately maintained `nengo_spa` package has a redesigned API. Both run today, but new submissions are generally encouraged to prefer `nengo_spa`; this submission preserves the original tutorial verbatim.

## Run it

In NengoGUI (the intended way to view it):

```bash
pip install nengo-gui
nengo spa_question_answering.py
```

Or just at the command line:

```bash
python spa_question_answering.py
```

## How it works

Semantic pointers are high-dimensional unit vectors. Two pointers are *bound* with circular convolution (`*` in SPA syntax), which produces a third pointer that is dissimilar to both operands. The binding is approximately invertible: convolving the bound pair with the *pseudoinverse* (`~`) of one operand recovers an approximation of the other. Cleanup is provided implicitly by the readout's similarity to the vocabulary, which is what the NengoGUI `SpaSimilarity` plot shows on `E`.

`spa.Cortical` compiles each action string into the equivalent set of `nengo.Connection`s between the named states — no action selection, no gating, the bindings are always running.

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

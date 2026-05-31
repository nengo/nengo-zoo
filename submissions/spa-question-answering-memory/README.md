# spa-question-answering-memory

A spiking SPA model that binds colour × shape into a bound pair, stores it in a recurrent working memory, and answers questions about the stored pair after the original inputs have gone quiet.

## Description

This is example 4 of chapter 5 of *How to Build a Brain* (Eliasmith, 2013), distributed with nengo-gui as `4-spa_question-memory.py`. It extends the [`spa-question-answering`](../spa-question-answering) model by adding a recurrent `spa.State(dim, feedback=1)` named `memory` and rewiring the cortical actions so the bound pair persists after stimulus offset.

The cortical actions are:

```
D = A * B          # bind colour (A) to shape (B)
memory = D         # write the bound pair into recurrent memory
E = memory * ~C    # answer cue questions from memory
```

Stimulation is sequential rather than alternating:

- `t ∈ [0, 0.25)` — `A=RED`, `B=CIRCLE`
- `t ∈ [0.25, 0.5)` — `A=BLUE`, `B=SQUARE`
- `t ≥ 0.5` — inputs zero; cues start cycling and memory must answer

By the end of the input phase, `memory` holds (approximately) `RED * CIRCLE + BLUE * SQUARE`. After that, asking `C = SQUARE` returns something most similar to `BLUE`; asking `C = BLUE` returns `SQUARE`; and so on.

The model uses 32-dimensional semantic pointers — small enough for fast CI builds while still keeping vocabulary similarities clean.

## A note on `nengo.spa` vs `nengo_spa`

This script uses the legacy `nengo.spa` module that still ships inside core Nengo. The newer, separately maintained `nengo_spa` package has a redesigned API. Both run today, but new submissions are generally encouraged to prefer `nengo_spa`; this submission preserves the original tutorial verbatim.

## Run it

In NengoGUI (the intended way to view it):

```bash
pip install nengo-gui
nengo spa_question_answering_memory.py
```

Or just at the command line:

```bash
python spa_question_answering_memory.py
```

## How it works

The trick is the `feedback=1` argument to `spa.State`: the state gets a recurrent connection from its own output back to its input with unit gain, which approximates a perfect integrator. Inputs from the cortical action `memory = D` accumulate; once those inputs go to zero, the state holds the most recent value (up to neural drift). The cue is then used to project back out a single attribute via the inverse-binding action `E = memory * ~C`, just as in the no-memory version.

`spa.Cortical` compiles each action string into the equivalent set of always-on `nengo.Connection`s — no action selection, no gating, the bindings and memory writes are continuous.

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

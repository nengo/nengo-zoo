# spa-question-answering-control

A spiking SPA model that decides — using a basal-ganglia/thalamus action-selection loop — whether an incoming visual input is something to *remember* (a STATEMENT) or something to *answer* (a QUESTION), and acts accordingly.

## Description

This is example 6 of chapter 5 of *How to Build a Brain* (Eliasmith, 2013), distributed with nengo-gui as `6-spa_question-control.py`. It's the most elaborate of the chapter's question-answering models: instead of always-on cortical bindings, the right thing to do is *selected* dynamically based on the structure of the visual input.

There are three states:

- `visual` — receives the current input, e.g. `STATEMENT+RED*CIRCLE` or `QUESTION+BLUE`
- `memory` — recurrent SPA state, holds the sum of seen bound pairs
- `motor` — read-out used to "answer" questions

And two competing actions, picked by `spa.BasalGanglia` / `spa.Thalamus`:

```
dot(visual, STATEMENT) --> memory = visual          # store the statement
dot(visual, QUESTION)  --> motor = memory * ~visual # answer the question
```

The utility of each action is the dot product of `visual` with one of the role pointers (`STATEMENT`, `QUESTION`). Whichever role is present wins, the basal ganglia inhibits the loser, the thalamus releases the winner, and the corresponding routing happens. Statements load memory; questions read it out.

A toy schedule of inputs runs:

- `0.1 – 0.3 s`: `STATEMENT + RED*CIRCLE`
- `0.35 – 0.5 s`: `STATEMENT + BLUE*SQUARE`
- `0.55 – 0.7 s`: `QUESTION + BLUE` → motor should read out `SQUARE`
- `0.75 – 0.9 s`: `QUESTION + CIRCLE` → motor should read out `RED`

This example is *sensitive to dimensionality*: 32 dimensions is small enough that you may need a couple of runs (or to bump `dim` higher) to see clean answers in motor. The tutorial keeps it at 32 to stay fast in the GUI.

## A note on `nengo.spa` vs `nengo_spa`

This script uses the legacy `nengo.spa` module that still ships inside core Nengo. The newer, separately maintained `nengo_spa` package has a redesigned action-selection API. Both run today, but new submissions are generally encouraged to prefer `nengo_spa`; this submission preserves the original tutorial verbatim.

## Run it

In NengoGUI (the intended way to view it):

```bash
pip install nengo-gui
nengo spa_question_answering_control.py
```

Or just at the command line:

```bash
python spa_question_answering_control.py
```

## How it works

`spa.BasalGanglia` builds a spiking neural model of the direct/indirect striatal pathway: a population per action whose firing rate scales with the action's *utility* (here, `dot(visual, STATEMENT)` or `dot(visual, QUESTION)`). The pathway competitively inhibits all but the highest-utility action's GPi output. `spa.Thalamus` reads GPi, gates the corresponding routing connection (here, either "copy `visual` into `memory`" or "compute `memory * ~visual` into `motor`"), and lets the winning action execute.

The result is a closed-loop control system: stimuli arrive in `visual`, the BG/thalamus pick which routing to enable based on the role pointer present, and `memory`/`motor` update accordingly. No clock, no explicit gating signals from outside — selection emerges from the dynamics of the action utilities.

`memory` uses `feedback=1, feedback_synapse=0.1` to integrate inputs over a 100 ms timescale, giving the statement enough time to be written in cleanly without smearing into the subsequent question.

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

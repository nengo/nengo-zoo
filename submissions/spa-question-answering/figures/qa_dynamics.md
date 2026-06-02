1.5 seconds of the SPA question-answering network, produced by running [`spa_question_answering.py`](../spa_question_answering.py) directly.

Each panel plots the similarity (dot product) between the state's decoded semantic pointer and each of the candidate vocabulary vectors `{RED, BLUE, CIRCLE, SQUARE}` over time.

**A — color input.** Alternates `RED` (0 – 0.5 s) and `BLUE` (0.5 – 1.0 s).

**B — shape input.** Tracks A, alternating `CIRCLE` and `SQUARE` on the same schedule, so during the first half-second the network observes the bound pair `RED * CIRCLE`, then `BLUE * SQUARE`, and so on.

**C — cue.** Cycles through `ZERO`, `CIRCLE`, `RED`, `ZERO`, `SQUARE`, `BLUE` over the 1.5-second window. Each cue stays high for ~250 ms.

**E — recovered answer.** The result of `D * ~C` where `D = A * B`. When the cue is `CIRCLE`, the answer is the color (RED or BLUE depending on the current binding); when the cue is `RED` or `BLUE`, the answer is the shape; when the cue is `ZERO`, the answer is near zero across all vocabulary terms. The decoded similarities follow this pattern after a short transient delay.

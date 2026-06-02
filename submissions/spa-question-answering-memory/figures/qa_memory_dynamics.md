2 seconds of the SPA question-answering-with-memory network, produced by running [`spa_question_answering_memory.py`](../spa_question_answering_memory.py) directly.

The first 0.5 s presents two colour×shape bindings to the network with no cue; the bindings accumulate in the memory subnet. After 0.5 s the colour and shape inputs go silent and only the cue varies — the network must use memory to recover the answers.

**A — colour input.** `RED` (0 – 0.25 s) then `BLUE` (0.25 – 0.5 s), then `ZERO` for the rest of the run.

**B — shape input.** `CIRCLE` then `SQUARE` on the same schedule, then `ZERO`.

**C — cue.** Stays at `ZERO` for the first 0.5 s while the bindings are presented, then cycles through `ZERO`, `CIRCLE`, `RED`, `ZERO`, `SQUARE`, `BLUE`.

**memory — bound-pair contents.** `RED * CIRCLE` builds up during 0 – 0.25 s and persists; `BLUE * SQUARE` builds up during 0.25 – 0.5 s and persists. Both pairs remain in memory throughout the question phase thanks to the State's self-feedback.

**E — recovered answer.** Computed as `memory * ~cue`. When the cue is `CIRCLE` the network recovers a mixture peaked on `RED`; when the cue is `RED` it recovers `CIRCLE`; and so on. The mixture is fuzzier than in the cortical-only variant because both bound pairs share the memory at once.

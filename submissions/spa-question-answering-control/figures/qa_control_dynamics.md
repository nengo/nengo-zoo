1 second of the SPA question-answering-with-control network, produced by running [`spa_question_answering_control.py`](../spa_question_answering_control.py) directly.

The basal-ganglia / thalamus loop selects one of two cortical actions on each timestep based on what's currently in `visual`: when `visual` is similar to `STATEMENT + …`, the bound pair is written into memory; when `visual` is similar to `QUESTION + …`, the memory is unbound by the cue and routed to `motor`.

**Top — visual input.** Each subnet receives one of four scripted inputs:
`STATEMENT + RED * CIRCLE` (0.1 – 0.3 s), `STATEMENT + BLUE * SQUARE` (0.35 – 0.5 s), `QUESTION + BLUE` (0.55 – 0.7 s), `QUESTION + CIRCLE` (0.75 – 0.9 s). Similarities to the six atomic vocab terms are plotted.

**Middle — memory contents.** Plotted against the two bound pairs the network is shown. `RED * CIRCLE` rises during the first statement, then `BLUE * SQUARE` accumulates during the second; both persist into the question phase thanks to the memory's recurrent connection.

**Bottom — motor output.** Stays near zero during the statement phase (the BG selects the "store" action, not the "answer" action). When the first question (`QUESTION + BLUE`) arrives, `SQUARE` rises sharply — the network has recalled the answer to "blue is what shape?". When the second question (`QUESTION + CIRCLE`) arrives, `RED` rises — "circle is what colour?".

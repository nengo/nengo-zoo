# NengoGUI version

`lamprey_model.py` is a **thin shim** over the library API: it calls
`lamprey.build_model()` and aliases the handles onto the short names the
`.cfg` file references. The library code is the single source of truth.

To run:

```bash
pip install nengo-gui
nengo lamprey_model.py
```

`lamprey_model.py.cfg` stores the GUI layout and visualization config (XY
plots for the CPG state, a value plot for the tensions, an HTMLView for the
body, and a slider on the kick stimulus).

For headless / scripted execution that produces static figures, use the
library API directly via `../run.py` from the submission root.

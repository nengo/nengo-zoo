"""
NengoGUI variant of the lamprey locomotion model.

This is a thin shim over the library API (`lamprey.build_model`) so that the
library code is the single source of truth. The aliases below map the
library's descriptive handle names onto the short identifiers the .cfg file
expects, so the visualization layout (lamprey_model.py.cfg) continues to
work unchanged.

To open in NengoGUI:

    pip install nengo-gui
    nengo lamprey_model.py

For headless execution producing static figures, use ../run.py instead.
"""

from lamprey import build_model

model, _h = build_model(seed=1)

# Aliases for the .cfg file's bare-name references.
a        = _h["cpg"]
u        = _h["kick"]
tensions = _h["tensions"]
lamprey  = _h["body_node"]

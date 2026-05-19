"""
Body of the lamprey: consumes 10-D tensions and stores a 9-D shape state
(differences between adjacent segment tensions).

When run inside NengoGUI, an SVG visualization is rendered via the
``_nengo_html_`` attribute (preserved from the original 2019 implementation).
For headless execution (e.g. the example script in this submission), read
``self.state`` for the segment-difference vector or call ``self.history``
after the simulation for a per-timestep record.
"""

from __future__ import annotations

import numpy as np


class Lamprey:
    """Lamprey body callable, designed to be used as a ``nengo.Node``."""

    SVG_TEMPLATE = """
        <svg width="100%" height="100%" viewbox="0 0 100 100">
            <circle cx="{x1[1]}" cy="{y1[1]}" r="1" stroke="black"/>
            <circle cx="{x1[2]}" cy="{y1[2]}" r="1" stroke="black"/>
            <circle cx="{x1[3]}" cy="{y1[3]}" r="1" stroke="black"/>
            <circle cx="{x1[4]}" cy="{y1[4]}" r="1" stroke="black"/>
            <circle cx="{x1[5]}" cy="{y1[5]}" r="1" stroke="black"/>
            <circle cx="{x1[6]}" cy="{y1[6]}" r="1" stroke="black"/>
            <circle cx="{x1[7]}" cy="{y1[7]}" r="1" stroke="black"/>
            <circle cx="{x1[8]}" cy="{y1[8]}" r="1" stroke="black"/>
            <polyline stroke="black" fill="none"
            points="{x1[0]},{y1[0]}, {x1[1]},{y1[1]}, {x1[2]},{y1[2]},
            {x1[3]},{y1[3]}, {x1[4]},{y1[4]}, {x1[5]},{y1[5]},
            {x1[6]},{y1[6]}, {x1[7]},{y1[7]}, {x1[8]},{y1[8]}"/>
            <circle cx="{x1[0]}" cy="{y1[0]}" r="3" stroke="black" fill="red"/>
        </svg>
    """

    def __init__(self, record_history: bool = False):
        self.state = np.zeros(9)
        self._nengo_html_ = ""
        self.dt = 0.001
        self.record_history = record_history
        self.history: list[np.ndarray] = []

    def __call__(self, t, x):
        # Segment-to-segment tension differences drive body curvature.
        self.state = +np.diff(x)
        if self.record_history:
            self.history.append(self.state.copy())

        # Build the SVG for NengoGUI visualization.
        y1 = 5 * self.state + 50
        x1 = np.linspace(10, 90, 10)
        self._nengo_html_ = self.SVG_TEMPLATE.format(x1=x1, y1=y1)

    def body_points(self) -> np.ndarray:
        """Return the current (x, y) coordinates of the 10 body segments."""
        y1 = 5 * self.state + 50
        x1 = np.linspace(10, 90, 10)
        return np.column_stack([x1[1:], y1])  # 9 trailing segments

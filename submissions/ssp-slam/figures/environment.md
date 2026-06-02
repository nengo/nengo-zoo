The toy 2-D environment used by [`examples/example_usage.py`](../examples/example_usage.py).

A seeded `nengo.processes.WhiteSignal` generates the agent's trajectory (gray) starting at the black star, of which the first `T = 8 s` are simulated. Three semantic objects — a blue triangle, an orange triangle, and an orange square — are placed at fixed points along the path; their positions are determined by the seed and the path period (held fixed at 10 s), so the layout stays the same regardless of `T`. A single black wall sits just off the trajectory in view of the agent without occupying the path.

Each object's semantic identity is the binding of a shape semantic pointer (`triangle` or `square`) with a color semantic pointer (`blue` or `orange`); this is what enables the memory-query figure to resolve queries like "all triangles" or "all blues" after training.

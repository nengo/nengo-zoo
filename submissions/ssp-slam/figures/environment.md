The toy 2-D environment used by [`examples/example_usage.py`](../examples/example_usage.py).

A seeded `nengo.processes.WhiteSignal` generates the 10-second trajectory (gray) for the agent, starting at the black star. Three semantic objects — a blue triangle, an orange triangle, and an orange square — are placed at points 25 %, 50 %, and 75 % along the trajectory so the agent is guaranteed to pass within `view_rad = 0.3` of each one. A single black wall sits just off the trajectory near the 62 % mark, in view of the agent without occupying the path.

Each object's semantic identity is the binding of a shape semantic pointer (`triangle` or `square`) with a color semantic pointer (`blue` or `orange`); this is what enables the memory-query figure to resolve queries like "all triangles" or "all blues" after training.

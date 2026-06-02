Three semantic queries against the associative memory `SSPSlam.assomemory` at the end of the run.

After training, the memory has learned a mapping `landmark_SP → SSP_location` for the four landmarks it saw (3 objects + 1 wall). To answer a semantic query, we compose a query semantic pointer by binding shape × color terms, feed it as the *input* to the memory, and read out the *output* — an SSP. We then compute that SSP's similarity to a 60×60 grid of position SSPs over the domain and contour-plot the result.

**Left — "Blue triangle"** (`bind(triangle_SP, blue_SP)`): a tight peak at the blue-triangle landmark in the middle-left of the domain.

**Middle — "All triangles"** (`bind(triangle_SP, blue_SP + orange_SP)`): the peak still anchors at the blue triangle, but a second region of high similarity emerges at the orange triangle in the upper-right — the memory has identified both triangular objects via SP algebra applied at query time.

**Right — "All blues"** (`bind(triangle_SP + square_SP, blue_SP)`): a single peak at the blue triangle (the only blue object in the scene).

At this CI-sized `ssp_dim = 37`, some smearing of the query response onto the agent's transit corridors between landmarks is expected — the memory associates each landmark SP with the SSPs it observed while learning it, which span a short interval of the path. The higher-fidelity setup in Dumont et al. (2023) produces sharper localisation; the compositional structure of the queries is already visible here.

This is the headline semantic capability that distinguishes SSP-SLAM from a plain path integrator — the same neural circuit that holds a self-position estimate also fields compositional spatial queries about *what kinds of things* are *where*.

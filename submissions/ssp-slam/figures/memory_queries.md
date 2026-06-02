Three semantic queries against the associative memory `SSPSlam.assomemory` at the end of the run.

After training, the memory has learned a mapping `landmark_SP → SSP_location` for the four landmarks it saw (3 objects + 1 wall). To answer a semantic query, we compose a query semantic pointer by binding shape × color terms, feed it as the *input* to the memory, and read out the *output* — an SSP. We then compute that SSP's similarity to a 60×60 grid of position SSPs over the domain and contour-plot the result.

**Left — "Blue triangle"** (`bind(triangle_SP, blue_SP)`): a single peak at the upper-right corner, exactly where the blue triangle landmark sits.

**Middle — "All triangles"** (`bind(triangle_SP, blue_SP + orange_SP)`): a strong peak at the blue triangle, with a secondary blob in the lower-left where the orange triangle is — the memory has identified both triangular objects.

**Right — "All blues"** (`bind(triangle_SP + square_SP, blue_SP)`): a single peak at the blue triangle (the only blue object in the scene).

This is the headline semantic capability that distinguishes SSP-SLAM from a plain path integrator — the same neural circuit that holds a self-position estimate also fields compositional spatial queries about *what kinds of things* are *where*.

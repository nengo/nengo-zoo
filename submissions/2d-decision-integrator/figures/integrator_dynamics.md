6 seconds of the 2-D decision integrator, produced by running [`2d_decision_integrator.py`](../2d_decision_integrator.py) directly.

The network is presented with a constant 2-D input `(-0.5, +0.5)` corrupted by independent white-noise injections on the MT, LIP, and output ensembles, modelling the noisy evidence stream a perceptual decision-making circuit would receive.

**Left — time series.** The `x` (top) and `y` (bottom) components of the decoded state in MT (light blue), LIP (orange), and the output ensemble (red), with the input target as a dotted gray line. MT jumps to the input value within milliseconds and then jitters around it; LIP integrates over the next ~2 seconds and starts to track the input value; output follows LIP after a further delay because its intercepts are clamped to `≥ 0.3`, so it only fires once LIP's magnitude has crossed that threshold.

**Right — phase portrait.** LIP's 2-D trajectory (orange) wanders from the origin (gray dot) toward the input direction (black star at `(−0.5, +0.5)`). The output trajectory (red) covers the same path but only after LIP escapes the dashed threshold circle. The wandering after t ≈ 3 s is the integrator amplifying the injected ensemble noise.

14 seconds of the controlled oscillator driven through a four-segment speed program, produced by running [`controlled_oscillator.py`](../controlled_oscillator.py) directly.

**Top-left — speed program.** The `stim_speed` input held at four levels: `0` (rest), `+0.6` (forward spin), `0` (halt), `−0.6` (reverse spin). In NengoGUI this input is a slider; here it's stepped programmatically to produce the figure.

**Bottom-left — oscillator state vs. time.** The 2-D oscillator state `(x₀, x₁)` over the run. At rest the state relaxes onto the limit cycle but doesn't rotate. With non-zero speed it traces the cycle at a rate proportional to `|s|`. When the speed is reset to zero, the state freezes at whatever phase it had reached. Flipping the sign reverses the direction of rotation.

**Right — phase portrait.** The trajectory in the `(x₀, x₁)` plane, coloured by speed regime. The transient from the origin out to the unit circle (gray) is the limit-cycle attraction during the initial rest segment; the blue and red overlapping circles are the forward and reverse rotation phases.

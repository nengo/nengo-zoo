Headline output of [`examples/example_usage.py`](../examples/example_usage.py): a downstream spiking ensemble reading the LMU state and trained with PES to compute the windowed RMS (root-mean-square over the last `theta = 1 s`) of the LMU's scalar input.

**Top — input.** A band-limited (≤ 2 Hz) white-noise carrier multiplied by a slow ( ≤ 0.2 Hz) envelope, so the input's local amplitude varies meaningfully over the 25 s run and the windowed RMS target has visible structure rather than being roughly constant.

**Middle — windowed RMS.** True windowed RMS (gray) overlaid with the spiking-ensemble decode (dashed black). The vertical dotted line marks where the PES learning rule is switched off (75 % of the way through the run). The decoded RMS stays close to the true target both during training and after learning is shut off — the downstream ensemble has learned a nonlinear function of the LMU state.

**Bottom — absolute error.** Per-timestep `|true − decoded|`. Mean over the final 1 s ≈ 0.09; the largest excursions correspond to fast envelope transitions where the network has to update its estimate quickly.

The same network architecture (LMU → spiking ensemble → PES-trained decoder) can learn any nonlinear function of the windowed input (e.g., variance, peak amplitude, threshold-crossing rate, a classifier) by swapping the target signal that PES is given.

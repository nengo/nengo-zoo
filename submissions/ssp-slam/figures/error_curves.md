Self-position estimation accuracy over the 8-second run, for SSP-SLAM (blue) versus a stand-alone SSP path integrator (orange).

The metric is **cosine error** in the SSP space: `1 − cos(estimate, true SSP)`. This is the same robust measure used in the source paper — it sidesteps the grid-decoder noise that would otherwise dominate Euclidean distance error at this CI-sized SSP dimension (`ssp_dim = 37`).

Both networks are initialised to the true SSP for the first 50 ms. SSP-SLAM stays well below the PI-only curve for essentially the entire run: each landmark sighting after t ≈ 0.5 s lets SLAM snap its position estimate back toward the corresponding memory recall, holding cosine error in the 0.1 – 0.4 range. The pure path integrator, with no correction signal available, drifts up past 0.8 by the midpoint of the run.

The headline at higher fidelity (the Dumont et al. (2023) paper's `ssp_dim ≈ 240`, ~60-second runs) is that the SSP-SLAM curve stays well below the PI-only curve for the entire trajectory; the CI-sized demo captures the same correction dynamic in compressed form.

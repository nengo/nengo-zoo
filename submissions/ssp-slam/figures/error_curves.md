Self-position estimation accuracy over the 8-second run, for SSP-SLAM (blue) versus a stand-alone SSP path integrator (orange).

The metric is **cosine error** in the SSP space: `1 − cos(estimate, true SSP)`. This is the same robust measure used in the source paper — it sidesteps the grid-decoder noise that would otherwise dominate Euclidean distance error at this CI-sized SSP dimension (`ssp_dim = 37`).

Both networks are initialised to the true SSP for the first 50 ms and drift thereafter. Through the first ~3 seconds, before the associative memory has accumulated enough Voja/PES learning to make confident landmark→location recall, the SSP-SLAM and PI-only error curves rise similarly. Starting around t ≈ 3 s, the memory is trained enough that each landmark sighting corrects the SSP estimate: SSP-SLAM oscillates in the 0.2 – 0.4 range for the rest of the run, while the pure path integrator keeps drifting up past 1.0 cosine error.

The headline at higher fidelity (the Dumont et al. (2023) paper's `ssp_dim ≈ 240`, ~60-second runs) is that SSP-SLAM stays well below the PI-only curve for the entire trajectory; the CI-sized demo captures the same correction dynamic in compressed form.

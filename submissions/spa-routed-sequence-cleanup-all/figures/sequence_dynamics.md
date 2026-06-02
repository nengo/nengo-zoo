1.5 seconds of the routed-sequence + cleanup-ensemble network, produced by running [`spa_routed_sequence_cleanup_all.py`](../spa_routed_sequence_cleanup_all.py) directly.

For the first 0.4 s the network is primed with `0.8 * START + D`; after that, the routing actions in the basal-ganglia / thalamus loop cycle the state through `A → B → C → D → E → A → …` with each element holding for roughly 60 – 80 ms before the next transition is selected.

**Top — state output.** Similarity of `model.state` to each of the five sequence elements. Each transition shows up as a near-square hand-off from one curve peaking near 1.0 to the next; the small overlap during each transition is the BG/thalamus settling onto the next selected action.

**Bottom — cleanup ensemble.** The 16-dimensional vocabulary projected through the per-vocab-element transformation matrix `pd` onto a 6-D ensemble. Each plotted dimension fires roughly in lockstep with the corresponding `state` similarity above — confirming that the cleanup ensemble is acting as an explicit categorical decoder of the rotating semantic-pointer state.

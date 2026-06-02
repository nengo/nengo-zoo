"""
SSP-SLAM toy demo: agent navigating a 2-D environment with semantic
landmarks and a wall.

The trajectory is a seeded white-signal path. Three objects (each a
shape×color binding) are placed at well-spread points *along* the path,
so the agent is guaranteed to see each within `view_rad` for several
timesteps. One wall is placed nearby. We instantiate SSPSlam alongside
a standalone PathIntegration baseline and compare position estimates.

After the sim, we read the trained associative memory and ask three
semantic queries — "blue triangle", "all triangles", "all blues" —
visualising where memory thinks each lives.

CI requirement: completes without error. Plotting is optional and
gracefully skipped if matplotlib is unavailable.
"""

import os
from pathlib import Path

import nengo
import numpy as np
import sspslam
from scipy.integrate import dblquad

from ssp_slam import SSPSlam


SEED = 11     # Hand-picked from a small sweep of white-signal seeds.
              # Trades a small reduction in per-landmark view-time for a
              # spatially well-spread item layout (blue triangle, orange
              # triangle and square at distinct corners), which makes the
              # "All triangles" compositional query in figures/
              # visibly two-peaked.
T = 10.0
DT = 0.001
TIMESTEPS = int(T / DT)
VIEW_RAD = 0.3
DOMAIN_HALF = 0.85   # path stays within (-0.85, 0.85)
PI_N_NEURONS = 200
CIRCCONV_N_NEURONS = 100
MEM_N_NEURONS_PER_DIM = 10  # mem_n_neurons = MEM_N_NEURONS_PER_DIM * ssp_dim


def make_environment(ssp_space, path):
    """3 objects placed on the path, 1 wall placed near it."""
    d = ssp_space.ssp_dim
    item_idxs = [int(TIMESTEPS * f) for f in (0.25, 0.5, 0.75)]
    item_locations = path[item_idxs]

    item_shapes = ["^", "s", "^"]
    item_colors = ["blue", "orange", "orange"]

    shape_sps = nengo.dists.UniformHypersphere(surface=True).sample(
        2, d, rng=np.random.RandomState(SEED))
    color_sps = nengo.dists.UniformHypersphere(surface=True).sample(
        2, d, rng=np.random.RandomState(SEED + 10))

    shape_to_idx = {"^": 0, "s": 1}
    color_to_idx = {"blue": 0, "orange": 1}
    item_sps = ssp_space.bind(
        shape_sps[[shape_to_idx[s] for s in item_shapes]],
        color_sps[[color_to_idx[c] for c in item_colors]],
    )

    # Place one wall just past the midpoint of the path, offset so the
    # agent passes alongside (not through) it.
    wall_center = path[int(TIMESTEPS * 0.62)]
    offset = np.array([0.18, 0.12])
    wall_w, wall_h = 0.18, 0.18
    cx, cy = wall_center[0] + offset[0], wall_center[1] + offset[1]
    wall_boundaries = np.array([[
        [cx - wall_w / 2, cx + wall_w / 2],
        [cy - wall_h / 2, cy + wall_h / 2],
    ]])
    wall_sps = nengo.dists.UniformHypersphere(surface=True).sample(
        1, d, rng=np.random.RandomState(SEED + 20))

    return {
        "item_locations": item_locations,
        "item_shapes": item_shapes,
        "item_colors": item_colors,
        "item_sps": item_sps,
        "shape_sps": shape_sps,
        "color_sps": color_sps,
        "wall_boundaries": wall_boundaries,
        "wall_sps": wall_sps,
    }


def encode_walls(ssp_space, wall_boundaries):
    """Integrate the SSP encoding over each wall's rectangle (small dims = fast)."""
    d = ssp_space.ssp_dim
    n_walls = wall_boundaries.shape[0]
    wall_ssps = np.zeros((n_walls, d))
    for j in range(n_walls):
        for i in range(d):
            wall_ssps[j, i] = dblquad(
                lambda y, x: ssp_space.encode(np.array([x, y]))[0, i],
                wall_boundaries[j, 0, 0], wall_boundaries[j, 0, 1],
                wall_boundaries[j, 1, 0], wall_boundaries[j, 1, 1],
                epsabs=1e-3,
            )[0]
    return ssp_space.normalize(wall_ssps)


def vec_to_walls(path, wall_boundaries):
    """Vector from agent to closest point on each wall rectangle, per timestep."""
    x = path[:, None, 0]
    y = path[:, None, 1]
    cx = np.clip(x, wall_boundaries[None, :, 0, 0], wall_boundaries[None, :, 0, 1])
    cy = np.clip(y, wall_boundaries[None, :, 1, 0], wall_boundaries[None, :, 1, 1])
    return np.stack([cx - x, cy - y], axis=-1)


def main():
    np.random.seed(SEED)

    # 1. SSP space — n_scales=2, n_rotates=3 gives ssp_dim=37, the
    # smallest that lets the memory learn landmark→SSP associations
    # cleanly inside ~10 s of sim.
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    ssp_space = sspslam.HexagonalSSPSpace(
        domain_dim=2, n_scales=2, n_rotates=3,
        domain_bounds=1.2 * bounds,
        length_scale=0.3, seed=SEED,
    )
    d = ssp_space.ssp_dim
    print(f"ssp_dim = {d}")

    # 2. Seeded white-signal path, shifted to (-DOMAIN_HALF, DOMAIN_HALF).
    path = np.hstack([
        nengo.processes.WhiteSignal(T, high=0.5, seed=SEED).run(T, dt=DT),
        nengo.processes.WhiteSignal(T, high=0.5, seed=SEED + 1).run(T, dt=DT),
    ])
    for i in range(2):
        lo, hi = path[:, i].min(), path[:, i].max()
        path[:, i] = (path[:, i] - lo) / (hi - lo) * (2 * DOMAIN_HALF) - DOMAIN_HALF

    real_ssp = ssp_space.encode(path)
    real_inv_ssp = ssp_space.invert(real_ssp)

    # 3. Environment.
    env = make_environment(ssp_space, path)
    item_locations = env["item_locations"]
    item_sps = env["item_sps"]
    shape_sps = env["shape_sps"]
    color_sps = env["color_sps"]
    wall_boundaries = env["wall_boundaries"]
    wall_sps = env["wall_sps"]
    n_items = item_locations.shape[0]
    n_walls = wall_boundaries.shape[0]
    n_landmarks = n_items + n_walls

    wall_ssps = encode_walls(ssp_space, wall_boundaries)

    # 4. Per-timestep landmark visibility tables.
    vec_to_items = item_locations[None, :, :] - path[:, None, :]
    vec_walls = vec_to_walls(path, wall_boundaries)
    landmark_sps = np.vstack([item_sps, wall_sps])

    # 5. Velocity from path.
    vels = (1.0 / DT) * np.diff(path, axis=0, prepend=path[0:1])
    vel_scaling_factor = 1.0 / np.max(np.abs(ssp_space.phase_matrix @ vels.T))
    vels_scaled = vels * vel_scaling_factor

    # 6. Landmark SP space.
    lm_space = sspslam.SPSpace(n_landmarks, d, seed=SEED, vectors=landmark_sps)

    # 7. Input functions (closures over the path / env).
    def velocity_func(t):
        i = int(np.minimum(np.floor(t / DT), TIMESTEPS - 2))
        return vels_scaled[i]

    def init_state_func(t):
        if t < 0.05:
            return real_ssp[max(0, int((t - DT) / DT))]
        return np.zeros(d)

    def landmark_id_func(t):
        i = int(np.clip((t - DT) / DT, 0, TIMESTEPS - 1))
        vecs = np.vstack([vec_to_items[i], vec_walls[i]])
        dists = np.linalg.norm(vecs, axis=1)
        if np.all(dists > VIEW_RAD):
            return -1
        return int(np.argmin(dists))

    def landmark_sp_func(t):
        cur = landmark_id_func(t)
        if cur < 0:
            return np.zeros(d)
        return landmark_sps[cur]

    def landmark_vecssp_func(t):
        cur = landmark_id_func(t)
        if cur < 0:
            return np.zeros(d)
        i = int(np.clip((t - DT) / DT, 0, TIMESTEPS - 1))
        if cur < n_items:
            return ssp_space.encode(vec_to_items[i, cur]).flatten()
        return ssp_space.bind(
            real_inv_ssp[i:i + 1],
            wall_ssps[cur - n_items:cur - n_items + 1],
        ).flatten()

    def is_landmark_in_view_func(t):
        return 0 if landmark_id_func(t) >= 0 else 10

    # 8. Build the network.
    with nengo.Network(seed=SEED) as model:
        vel_in = nengo.Node(velocity_func, label="velocity")
        init_in = nengo.Node(init_state_func, label="init")
        lm_vec = nengo.Node(landmark_vecssp_func, label="lm_vec")
        lm_id = nengo.Node(landmark_sp_func, label="lm_id")
        is_lm = nengo.Node(is_landmark_in_view_func, label="is_lm")

        slam = SSPSlam(
            ssp_space=ssp_space,
            lm_space=lm_space,
            n_landmarks=n_landmarks,
            view_rad=VIEW_RAD,
            pi_n_neurons=PI_N_NEURONS,
            mem_n_neurons=MEM_N_NEURONS_PER_DIM * d,
            circconv_n_neurons=CIRCCONV_N_NEURONS,
            vel_scaling_factor=vel_scaling_factor,
            seed=SEED,
        )

        nengo.Connection(vel_in, slam.velocity_input, synapse=None)
        nengo.Connection(init_in, slam.pathintegrator.input, synapse=None)
        nengo.Connection(lm_vec, slam.landmark_vec_ssp, synapse=None)
        nengo.Connection(lm_id, slam.landmark_id_input, synapse=None)
        nengo.Connection(is_lm, slam.no_landmark_in_view, synapse=None)

        # Standalone PI baseline for comparison.
        pi_baseline = sspslam.networks.PathIntegration(
            ssp_space, PI_N_NEURONS, 0.05,
            scaling_factor=vel_scaling_factor, stable=True, solver_weights=False,
        )
        nengo.Connection(vel_in, pi_baseline.velocity_input, synapse=None)
        nengo.Connection(init_in, pi_baseline.input, synapse=None)

        slam_probe = nengo.Probe(slam.pathintegrator.output, synapse=0.05)
        pi_probe = nengo.Probe(pi_baseline.output, synapse=0.05)
        mem_weights_probe = nengo.Probe(
            slam.assomemory.conn_out, "weights", sample_every=T,
        )
        mem_encoders_probe = nengo.Probe(
            slam.assomemory.conn_in.learning_rule, "scaled_encoders",
            sample_every=T,
        )

    # 9. Simulate. Capture everything we need from sim.data while the
    # simulator is still open, then close.
    with nengo.Simulator(model) as sim:
        sim.run(T)

        slam_ssps = sim.data[slam_probe][::10]
        pi_ssps = sim.data[pi_probe][::10]
        t_sub = sim.trange()[::10]
        mem_decoders = sim.data[mem_weights_probe][-1].T
        mem_encoders = sim.data[mem_encoders_probe][-1, :, :]
        mem_gain = sim.data[slam.assomemory.memory].gain
        mem_bias = sim.data[slam.assomemory.memory].bias

    # 10. Cosine similarity to the true SSP — robust signal that does not
    # depend on the (noisy at small ssp_dim) grid decoder.
    true_ssps = real_ssp[::10]

    def cos_err(estimates, truth):
        norms = np.linalg.norm(estimates, axis=1)
        norms = np.where(norms > 1e-9, norms, 1.0)
        return 1.0 - np.sum(estimates * truth, axis=1) / norms

    slam_cos_err = cos_err(slam_ssps, true_ssps)
    pi_cos_err = cos_err(pi_ssps, true_ssps)
    print(f"SLAM mean cosine err   : {slam_cos_err.mean():.3f}")
    print(f"PI-only mean cosine err: {pi_cos_err.mean():.3f}")

    # 11. Memory queries (neuron_type.rates is a pure function — no open
    # simulator required).
    memory_neuron_type = slam.assomemory.memory.neuron_type

    def query_memory(query_sp):
        # nengo's neuron_type.rates() treats x as a (n_samples, n_neurons)
        # batch; pass a 2-D row and flatten the (1, d) result back to (d,).
        x = np.dot(query_sp.reshape(1, -1), mem_encoders.T)
        activities = memory_neuron_type.rates(x, mem_gain, mem_bias)
        return (activities @ mem_decoders).flatten()

    blue_triangle = ssp_space.normalize(
        ssp_space.bind(shape_sps[0:1], color_sps[0:1])
    ).flatten()
    all_triangles = ssp_space.normalize(
        ssp_space.bind(shape_sps[0:1], np.sum(color_sps, axis=0, keepdims=True))
    ).flatten()
    all_blues = ssp_space.normalize(
        ssp_space.bind(np.sum(shape_sps, axis=0, keepdims=True), color_sps[0:1])
    ).flatten()

    blue_tri_ssp = query_memory(blue_triangle)
    triangles_ssp = query_memory(all_triangles)
    blues_ssp = query_memory(all_blues)

    # 12. Plots (optional).
    try:
        import matplotlib
        matplotlib.use("Agg")
        # sspslam.utils.figure_utils sets text.usetex=True at import time
        # via the `import sspslam` above; reset for portable CI runs.
        matplotlib.rcParams["text.usetex"] = False
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig_dir = Path(__file__).resolve().parent.parent / "figures"
        fig_dir.mkdir(exist_ok=True)
        color_map = {"blue": "#3a76c1", "orange": "#e89030"}

        # --- Figure 1: environment + ground-truth path ---
        # The grid-decoded estimated paths are too noisy at this CI-sized
        # ssp_dim to overlay informatively; we report estimate accuracy
        # numerically in Figure 2 instead.
        fig, ax = plt.subplots(figsize=(5, 5))
        for w in wall_boundaries:
            ax.add_patch(Rectangle(
                (w[0, 0], w[1, 0]), w[0, 1] - w[0, 0], w[1, 1] - w[1, 0],
                facecolor="k", alpha=0.6,
            ))
        for loc, sh, c in zip(item_locations, env["item_shapes"], env["item_colors"]):
            ax.plot(loc[0], loc[1], sh, markersize=12,
                    markerfacecolor=color_map[c], markeredgecolor="k")
        ax.plot(path[:, 0], path[:, 1], color="gray", lw=2, label="agent path")
        ax.plot(path[0, 0], path[0, 1], "*", color="k", markersize=12,
                label="start", markeredgewidth=0)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Toy environment + agent trajectory")
        fig.tight_layout()
        fig.savefig(fig_dir / "environment.png", dpi=110)
        plt.close(fig)
        print(f"Saved {fig_dir / 'environment.png'}")

        # --- Figure 2: cosine-error curves ---
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(t_sub, slam_cos_err, label="SSP-SLAM")
        ax.plot(t_sub, pi_cos_err, label="SSP-PI only", alpha=0.8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("1 − cos(estimate, true SSP)")
        ax.legend()
        ax.set_title("SLAM corrects PI drift on landmark sightings")
        fig.tight_layout()
        fig.savefig(fig_dir / "error_curves.png", dpi=110)
        plt.close(fig)
        print(f"Saved {fig_dir / 'error_curves.png'}")

        # --- Figure 3: memory queries (manual similarity grid; bypass
        # the upstream ssp_space.similarity_plot, which mis-handles the
        # 1-D query output with our small ssp_dim) ---
        n_grid = 60
        gx, gy = np.meshgrid(np.linspace(-1, 1, n_grid),
                             np.linspace(-1, 1, n_grid))
        grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
        grid_ssps = ssp_space.encode(grid_pts)  # (n_grid**2, d)

        fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
        for ax, ssp_out, title in zip(
            axs,
            [blue_tri_ssp, triangles_ssp, blues_ssp],
            ["Blue triangle", "All triangles", "All blues"],
        ):
            sims = (grid_ssps @ ssp_out).reshape(n_grid, n_grid)
            ax.contourf(gx, gy, sims, levels=20, cmap="Blues")
            for loc, sh, c in zip(
                item_locations, env["item_shapes"], env["item_colors"]
            ):
                ax.plot(loc[0], loc[1], sh, markersize=10,
                        markerfacecolor=color_map[c], markeredgecolor="k")
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
            ax.set_title(title)
            ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(fig_dir / "memory_queries.png", dpi=110)
        plt.close(fig)
        print(f"Saved {fig_dir / 'memory_queries.png'}")
    except ImportError:
        print("matplotlib not available — skipping plots.")


if __name__ == "__main__":
    main()

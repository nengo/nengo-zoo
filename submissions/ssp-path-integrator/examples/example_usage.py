"""
Demo: SSP path integration on a 2D circular trajectory.

The agent moves along a circle of radius 0.7. We drive the network with
the analytic velocity, initialize the SSP with the starting position, and
read back the network's SSP estimate of position. The SSP is then decoded
back to (x, y) and compared to ground truth.

CI requirement: completes without error. Plotting is optional — if
matplotlib is available, a path-comparison figure is saved to figures/.
"""

import os

import nengo
import numpy as np

import sspslam
from ssp_path_integrator import PathIntegrator


T = 0.5             # seconds
DT = 0.001
RADIUS = 0.7
OMEGA = 2 * np.pi   # 1 revolution / second
N_NEURONS = 200


def velocity_fn(t):
    """Analytic velocity for a circle of radius RADIUS."""
    return np.array([
        -RADIUS * OMEGA * np.sin(OMEGA * t),
         RADIUS * OMEGA * np.cos(OMEGA * t),
    ])


def true_position(t):
    return np.array([RADIUS * np.cos(OMEGA * t), RADIUS * np.sin(OMEGA * t)])


def build_model(ssp_space, scale_fac):
    d = ssp_space.ssp_dim
    init_ssp = ssp_space.encode(true_position(0.0).reshape(1, -1)).flatten()

    with nengo.Network(seed=0) as model:
        vel = nengo.Node(lambda t: velocity_fn(t) * scale_fac, label="velocity")
        init = nengo.Node(
            lambda t: init_ssp if t < 0.05 else np.zeros(d),
            label="init_ssp",
        )

        pi = PathIntegrator(ssp_space, N_NEURONS, scaling_factor=scale_fac)
        nengo.Connection(vel, pi.velocity_input, synapse=None)
        nengo.Connection(init, pi.input, synapse=None)

        model.out_probe = nengo.Probe(pi.output, synapse=0.05)
    return model


def main():
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    ssp_space = sspslam.HexagonalSSPSpace(
        domain_dim=2,
        n_scales=2,
        n_rotates=3,
        domain_bounds=bounds,
        length_scale=0.2,
        seed=0,
    )

    # Normalize the velocity input against the largest phase-projected speed
    # the trajectory will produce, so the VCOs operate in their stable range.
    sample_t = np.linspace(0, T, 100)
    sample_v = np.column_stack([velocity_fn(t) for t in sample_t]).T
    scale_fac = 1.0 / np.max(np.abs(ssp_space.phase_matrix @ sample_v.T))

    model = build_model(ssp_space, scale_fac)
    with nengo.Simulator(model) as sim:
        sim.run(T)

    # Decode position estimates.
    skip = 20
    estimated = ssp_space.decode(
        sim.data[model.out_probe][::skip], "from-set", "grid", 50,
    )
    t_decoded = sim.trange()[::skip]
    true_path = np.array([true_position(t) for t in t_decoded])

    final_err = np.linalg.norm(estimated[-1] - true_path[-1])
    mean_err = np.mean(np.linalg.norm(estimated - true_path, axis=1))
    print(f"True final position : ({true_path[-1, 0]: .3f}, {true_path[-1, 1]: .3f})")
    print(f"Estimated position  : ({estimated[-1, 0]: .3f}, {estimated[-1, 1]: .3f})")
    print(f"Final error         : {final_err: .3f}")
    print(f"Mean error          : {mean_err: .3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, "path_integration.png")

        plt.figure(figsize=(5, 5))
        plt.plot(true_path[:, 0], true_path[:, 1], color="gray", label="true")
        plt.plot(estimated[:, 0], estimated[:, 1], "--k", label="SSP-PI estimate")
        plt.axis("equal")
        plt.legend(loc="upper right")
        plt.title("SSP path integration: circular trajectory")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=110)
        print(f"Saved figure: {fig_path}")
    except ImportError:
        print("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()

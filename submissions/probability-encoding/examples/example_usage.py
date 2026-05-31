"""
Demo: estimate a Beta(2, 5) probability density with spiking SSP encoders.

We draw samples from Beta(2, 5), construct a `ProbabilityEncoder` in
both of its modes, and read out the density estimate from each:

  Population mode — one neuron per query point on (0, 1). Drive with
  `mu`; per-neuron firing rate approximates p(query_x).

  Single-neuron mode — one neuron with encoder = mu. Sweep the input
  SSP over (0, 1); the neuron's spike rate over time traces the PDF.

CI requirement: this script must complete without error. Plotting is
optional — if matplotlib is available, a comparison plot is saved to
figures/.
"""

import os

import nengo
import numpy as np
from scipy.stats import beta as beta_dist

from ssp_bayes_opt import sspspace

from probability_encoding import ProbabilityEncoder


N_SAMPLES = 500
SSP_DIM = 128
MAX_RATE = 50
DOMAIN = (0.0, 1.0)
N_QUERY = 50
SWEEP_DURATION = 2.0  # seconds — long enough that each query bin gets ~40 samples
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)  # for sspspace (no rng knob exposed in older versions)

    # 1. Training samples.
    samples = beta_dist.rvs(2, 5, size=N_SAMPLES, random_state=rng).reshape(-1, 1)
    query_xs = np.linspace(DOMAIN[0], DOMAIN[1], N_QUERY).reshape(-1, 1)
    true_pdf = beta_dist.pdf(query_xs.flatten(), 2, 5)

    # 2. SSP spaces — one per encoder, since each constructor updates the
    # space's lengthscale in place from its own training samples.
    ssp_space_pop = sspspace.RandomSSPSpace(ssp_dim=SSP_DIM, domain_dim=1)
    ssp_space_single = sspspace.RandomSSPSpace(ssp_dim=SSP_DIM, domain_dim=1)

    def sweep_input(t):
        x = (t / SWEEP_DURATION) * (DOMAIN[1] - DOMAIN[0]) + DOMAIN[0]
        x = min(max(x, DOMAIN[0]), DOMAIN[1])
        return ssp_space_single.encode(np.array([[x]])).flatten()

    # 3. Build both encoders inside the parent model so Nengo can see them.
    with nengo.Network(seed=SEED) as model:
        pop = ProbabilityEncoder(
            ssp_space=ssp_space_pop,
            training_samples=samples,
            domain_bounds=DOMAIN,
            query_points=query_xs,
            max_rate=MAX_RATE,
        )
        single = ProbabilityEncoder(
            ssp_space=ssp_space_single,
            training_samples=samples,
            domain_bounds=DOMAIN,
            max_rate=MAX_RATE,
        )

        mu_stim = nengo.Node(lambda t: pop.mu, label="mu_stim")
        nengo.Connection(mu_stim, pop.input, synapse=None)
        model.pop_probe = nengo.Probe(pop.ensemble.neurons)

        sweep_stim = nengo.Node(sweep_input, label="sweep_stim")
        nengo.Connection(sweep_stim, single.input, synapse=None)
        model.single_probe = nengo.Probe(single.ensemble.neurons)

    print(f"Bandwidth: {pop.bandwidth:.4f}, xi: {pop.xi:.4f}")

    with nengo.Simulator(model) as sim:
        sim.run(SWEEP_DURATION)

    # 5. Analyze.
    pop_spikes = sim.data[model.pop_probe]
    pop_rate = np.mean(pop_spikes, axis=0) / MAX_RATE

    pop_err = np.mean(np.abs(pop_rate - true_pdf))
    print(f"Population mode mean abs err: {pop_err:.4f}")

    # Map the swept single-neuron output back onto the query domain.
    t = sim.trange()
    sweep_x = (t / SWEEP_DURATION) * (DOMAIN[1] - DOMAIN[0]) + DOMAIN[0]
    single_spikes = sim.data[model.single_probe][:, 0]
    # Smooth the spike train into a per-x rate estimate.
    bin_edges = np.linspace(DOMAIN[0], DOMAIN[1], N_QUERY + 1)
    bin_idx = np.minimum(
        np.searchsorted(bin_edges, sweep_x) - 1, N_QUERY - 1,
    ).clip(0, N_QUERY - 1)
    single_rate = np.zeros(N_QUERY)
    counts = np.zeros(N_QUERY)
    for i, b in enumerate(bin_idx):
        single_rate[b] += single_spikes[i]
        counts[b] += 1
    counts[counts == 0] = 1
    single_rate = (single_rate / counts) / MAX_RATE

    single_err = np.mean(np.abs(single_rate - true_pdf))
    print(f"Single-neuron sweep mean abs err: {single_err:.4f}")

    # 6. Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        # The vendored utility code does not touch rcParams, but in case a
        # downstream package does, keep TeX off for portability.
        matplotlib.rcParams["text.usetex"] = False
        import matplotlib.pyplot as plt

        fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, "density_estimation.png")

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
        axes[0].plot(query_xs, true_pdf, color="gray", label="Beta(2,5)")
        axes[0].plot(query_xs, pop_rate, "--k", label="population estimate")
        axes[0].set_title("Population mode")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("p(x)")
        axes[0].legend(loc="upper right")

        axes[1].plot(query_xs, true_pdf, color="gray", label="Beta(2,5)")
        axes[1].plot(query_xs, single_rate, "--k", label="single-neuron sweep")
        axes[1].set_title("Single-neuron mode (swept)")
        axes[1].set_xlabel("x")
        axes[1].legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(fig_path, dpi=110)
        print(f"Saved figure: {fig_path}")
    except ImportError:
        print("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()

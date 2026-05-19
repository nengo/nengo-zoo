"""
Long-form LMU training run. Reproduces the headline plot from the notebook:
band-limited white-noise input, an LMU layer feeding a downstream spiking
ensemble trained with PES to compute a fixed time delay, with learning shut
off after 80% of the run to assess generalization.

Runs ~100 simulated seconds — a couple of minutes wall-clock on CPU.
"""

import os
import sys
from pathlib import Path

# Import the model defined in the parent directory's lmu.py.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nengo

import lmu as L


def main():
    with nengo.Simulator(L.model) as sim:
        sim.run(L.sim_t)

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    t_per_plot = 10
    for i in range(L.sim_t // t_per_plot):
        mask = (sim.trange() >= t_per_plot * i) & (sim.trange() < t_per_plot * (i + 1))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(sim.trange()[mask], sim.data[L.p_stim][mask],  label="input")
        ax.plot(sim.trange()[mask], sim.data[L.p_ideal][mask], label="ideal")
        ax.plot(sim.trange()[mask], sim.data[L.p_out][mask],   label="output")
        ax.set_title("Learning ON" if i * t_per_plot < L.sim_t * 0.8 else "Learning OFF")
        ax.set_ylim([-1, 1])
        ax.legend(loc="upper right")
        fig.tight_layout()
        out_path = fig_dir / f"lmu_segment_{i:02d}.png"
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

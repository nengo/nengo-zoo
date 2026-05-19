"""
Run the lamprey locomotion model and save the headline figures.

The CPG is kicked at t=0, settles into a stable limit cycle by ~t=1s, and
drives a 10-segment muscle-tension traveling wave. This script simulates
3 seconds and renders:

  - the 3D CPG state (limit-cycle attractor)
  - the 10D tension waveforms (traveling wave)
  - body snapshots at evenly spaced times (swimming motion)

CI requirement (Tier 1): completes without error. Total runtime ~10s.
"""

from __future__ import annotations

import os

import nengo
import numpy as np

from lamprey import build_model


SIM_DURATION = 3.0


def main():
    model, h = build_model(seed=1, record_body_history=True)
    with nengo.Simulator(model) as sim:
        sim.run(SIM_DURATION)

    t = sim.trange()
    cpg = sim.data[h["p_cpg"]]
    tensions = sim.data[h["p_tensions"]]
    body_history = np.array(h["body"].history) if h["body"].history else np.empty((0, 9))

    print(f"Simulated {t[-1]:.2f}s of swimming.")
    print(f"  CPG state at t=2.5s:      {cpg[int(2.5 / 0.001)]}")
    print(f"  tension at t=2.5s:        {tensions[int(2.5 / 0.001)]}")
    print(f"  body history shape:       {body_history.shape}")

    # Save plots if matplotlib is available.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. CPG limit cycle (top) + tension traveling wave (bottom).
    fig, axes = plt.subplots(3, 1, figsize=(9, 7))

    axes[0].plot(t, cpg)
    axes[0].set_ylabel("CPG state")
    axes[0].set_title("Central Pattern Generator (3D damped oscillator)")
    axes[0].legend(["x_0", "x_1", "x_2"], loc="upper right")

    for seg in range(10):
        axes[1].plot(t, tensions[:, seg] + seg * 0.5, lw=0.6)
    axes[1].set_ylabel("segment tension (offset)")
    axes[1].set_title("10 body segments — traveling wave")

    if body_history.size:
        axes[2].imshow(
            body_history.T,
            aspect="auto",
            origin="lower",
            extent=(0, SIM_DURATION, 0, 9),
            cmap="seismic",
            vmin=-np.abs(body_history).max(),
            vmax=+np.abs(body_history).max(),
        )
        axes[2].set_xlabel("time (s)")
        axes[2].set_ylabel("body segment")
        axes[2].set_title("Body curvature over time (∂ tension)")
    fig.tight_layout()
    waves_path = os.path.join(fig_dir, "waves.png")
    fig.savefig(waves_path, dpi=110)
    plt.close(fig)
    print(f"Saved: {waves_path}")

    # 2. Body snapshots at evenly spaced times (after the CPG has settled).
    if body_history.size:
        snapshot_times = np.linspace(1.5, SIM_DURATION - 0.05, 6)
        fig, axs = plt.subplots(1, len(snapshot_times), figsize=(12, 2.2), sharey=True)
        for ax, t_target in zip(axs, snapshot_times):
            idx = int(t_target / 0.001) - 1
            idx = min(max(idx, 0), body_history.shape[0] - 1)
            xs = np.linspace(0, 1, 9)
            ys = body_history[idx]
            ax.plot(xs, ys, "-o", color="steelblue", markersize=3)
            ax.set_title(f"t={t_target:.2f}s", fontsize=9)
            ax.set_ylim(-0.4, 0.4)
            ax.axhline(0, color="grey", lw=0.4)
        axs[0].set_ylabel("body curvature")
        fig.suptitle("Lamprey body snapshots — swimming motion")
        fig.tight_layout()
        snap_path = os.path.join(fig_dir, "body_snapshots.png")
        fig.savefig(snap_path, dpi=110)
        plt.close(fig)
        print(f"Saved: {snap_path}")


if __name__ == "__main__":
    main()

"""
Train an ActorCritic on a tiny 5x5 gridworld with one-hot states and TD(0).

The agent starts at (0, 0), needs to reach (4, 4), gets +1 at the goal and
-0.01 per step otherwise, and chooses actions by softmax over the
network's learned action values. After ~80 episodes the policy reliably
finds the goal in <30 steps.

This is small on purpose — CI runs it on every PR, so it has to stay under
~60 s wall-clock. To scale it up locally just bump `n_episodes` and
`max_steps_per_episode`.
"""

import numpy as np
import nengo

from nengo_a2c import ActorCritic, TD0, OneHotRepCP, softmax


GRID = 5
N_ACTIONS = 4
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
GOAL = (GRID - 1, GRID - 1)
STEP_PENALTY = -0.01
GOAL_REWARD = 1.0


def step_env(pos, action):
    dx, dy = ACTIONS[action]
    new_pos = (max(0, min(GRID - 1, pos[0] + dx)),
               max(0, min(GRID - 1, pos[1] + dy)))
    if new_pos == GOAL:
        return new_pos, GOAL_REWARD, True
    return new_pos, STEP_PENALTY, False


def train(n_episodes=80, max_steps_per_episode=30, temperature=0.3, seed=0):
    rng = np.random.default_rng(seed)
    rep = OneHotRepCP(ranges=(GRID, GRID))
    rule = TD0(n_actions=N_ACTIONS, lr=0.3, act_dis=0.9, state_dis=0.95)
    ac = ActorCritic(rep, rule, neuron_type=nengo.RectifiedLinear())

    episode_rewards = []
    for ep in range(n_episodes):
        pos = (0, 0)
        action = 0
        # Tell the network the episode just started; the action passed in
        # here is arbitrary because the reset flag suppresses the update.
        ac.step(pos, action, reward=0.0, reset=True)

        total_reward = 0.0
        for _ in range(max_steps_per_episode):
            # Pick action from current action values (softmax with temperature).
            _, action_values = ac.step(pos, action, reward=0.0, reset=False)
            probs = softmax(np.asarray(action_values) / temperature)
            action = int(rng.choice(N_ACTIONS, p=probs))

            # Step the env, then tell the network what happened.
            pos, r, done = step_env(pos, action)
            total_reward += r
            ac.step(pos, action, reward=r, reset=False)
            if done:
                break

        episode_rewards.append(total_reward)

    return ac, episode_rewards


def main():
    ac, rewards = train()

    n = len(rewards)
    early = np.mean(rewards[: max(1, n // 4)])
    late = np.mean(rewards[-max(1, n // 4):])
    print(f"Average episode reward — first quarter: {early:+.3f}")
    print(f"Average episode reward — last  quarter: {late:+.3f}")
    print(f"Improvement: {late - early:+.3f}")
    print(f"State value at start (0,0): {ac.step((0, 0), 0, 0.0)[0][0]:+.3f}")

    try:
        from pathlib import Path
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path(__file__).resolve().parent.parent / "figures"
        fig_dir.mkdir(exist_ok=True)

        episodes = np.arange(1, n + 1)
        # 10-episode rolling mean to make the trend legible.
        window = 10
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        x_smoothed = np.arange(window, n + 1)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(episodes, rewards, color="C0", alpha=0.35, lw=0.9, label="per-episode reward")
        ax.plot(x_smoothed, smoothed, color="C0", lw=2.0, label=f"{window}-episode rolling mean")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("episode")
        ax.set_ylabel("total return")
        ax.set_title("ActorCritic + TD(0) on a 5×5 gridworld")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "learning_curve.png", dpi=110)
        plt.close(fig)
        print(f"Saved {fig_dir / 'learning_curve.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()

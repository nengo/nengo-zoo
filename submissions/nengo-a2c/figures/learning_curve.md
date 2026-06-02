Learning curve from the `examples/example_usage.py` script — an `ActorCritic` with one-hot state representation and a `TD(0)` learning rule trained for 80 episodes on a 5×5 gridworld (goal at (4, 4), -0.01 per step, +1 at the goal). Action selection is softmax with temperature 0.3.

Light line: per-episode total return. Bold line: 10-episode rolling mean.

The agent starts with random exploration in the first ~15 episodes (per-episode return ≈ -0.3, the floor when the goal isn't reached within the 30-step budget), starts hitting the goal intermittently around episode 15 – 20, and converges to a near-optimal policy by episode 50 (return ≈ +0.93, which corresponds to roughly 8 steps to the goal). The same code with `n_episodes` and `max_steps_per_episode` scaled up reproduces the curve at higher resolution on local hardware.

# nengo-a2c

A spiking Actor-Critic (A2C) reinforcement learning network in Nengo, with an optional Legendre-Delay-Network (LDN) memory of recent rewards and values for n-step / continuous-time TD updates.

## Description

This is a vendored, focused cut of `rlnet` from [Bartlett et al. (2022)](https://mathpsych.org/presentation/1221), *"Improving Reinforcement Learning with Biologically Motivated Continuous State Representations"* (ICCM). The upstream repo contains the full RatBox/CartPole experiment suite and several state-representation variants (SSP, VSA, normalized, one-hot, ...). This submission packages the parts most reusable as a standalone Zoo entry:

- **Networks**
  - `ActorCritic` — basic single-layer A2C, no memory of past rewards. Optional state ensemble.
  - `ActorCriticLDN` — A2C whose reward and state-value signals are passed through LDN delay-line memories, enabling TD(n) (discrete-time) or TD(θ) (continuous-time) updates from a multi-step return.
  - `LDN` — a `nengo.Process` implementing Aaron Voelker's Legendre Delay Network. Can be dropped into a `nengo.Node` to provide a continuous memory of the last θ seconds of input.
- **Learning rules**
  - `TD0` — single-step TD(0) update.
  - `TDt` (TD-theta) — n-step / continuous-time TD update that consumes the LDN-stored reward and value traces.
- **State representations**
  - `NormalRep` — normalize the env's observation into `[-1, 1]`.
  - `OneHotRepCP`, `OneHotRepRB` — discretize the observation and one-hot encode it.

> ⚠️ **License.** Upstream is distributed under Applied Brain Research's academic-use-only license, which is **not** GPL-compatible. This submission preserves that license verbatim — see `LICENSE`. Academic research, teaching, and personal learning are permitted without payment; commercial use requires a license from ABR (sales@appliedbrainresearch.com).

## Installation

We recommend a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

The A2C classes are *trainer-style* — they build their own `nengo.Network`, instantiate their own `nengo.Simulator`, and expose a `step(state, action, reward, reset=False)` method that advances the simulation one timestep and returns the updated state-value and action-values. They're **not** drop-in `nengo.Network` subclasses to put inside a parent `with model:` block.

```python
import nengo
from nengo_a2c import ActorCritic, TD0, OneHotRepCP, softmax

rep = OneHotRepCP(ranges=(5, 5))                 # discrete 5×5 state space
rule = TD0(n_actions=4, lr=0.3, act_dis=0.9, state_dis=0.95)
ac = ActorCritic(rep, rule, neuron_type=nengo.RectifiedLinear())

state = (0, 0)
ac.step(state, 0, reward=0.0, reset=True)        # episode start
state_value, action_values = ac.step(state, action=0, reward=0.0)
probs = softmax(action_values / 0.3)             # softmax action selection
# ... pick action, step env, call ac.step(new_state, action_taken, reward), repeat
```

`examples/example_usage.py` is a runnable, CI-tested 5×5 gridworld demo that builds an `ActorCritic`, trains it for ~100 episodes, and reports the improvement in average episode reward.

## The LDN variant

`ActorCriticLDN` reads the same `(state, action, reward, reset)` API but routes reward and value through `LDN(theta, q)` memory nodes — a `q`-dimensional Legendre basis of the last `theta` seconds. The `TDt` rule then computes an n-step return by integrating the LDN's stored reward trace against pre-computed decoder weights:

```python
from nengo_a2c import ActorCriticLDN, TDt, OneHotRepCP

rep = OneHotRepCP(ranges=(5, 5))
rule = TDt(n_actions=4, lr=0.1, act_dis=0.85, state_dis=0.9, n=3, env_dt=0.01)
ac = ActorCriticLDN(rep, rule,
                    state_neurons=200, active_prop=0.1,
                    theta=0.03, q_r=6, q_v=6,
                    continuous=False)
```

The `continuous=True` path computes the decoder by integrating against a continuous discount factor instead of summing over n discrete time-steps, which is the paper's TD(θ) variant.

## How it works

The actor-critic is implemented as a single linear "weight matrix" rule operating on the state representation (or on neural-ensemble activities of that representation), updating both a scalar state value and a vector of action values from the TD error. There's no separate critic network; the same weight matrix carries both heads.

```
state ──► [rep.map] ──► [optional Ensemble] ──┐
                                              ▼
action ─────────────────────────────────► [rule Node] ──► state value
reward ─────────────────────────────────►              └► action values
reset  ─────────────────────────────────►
```

The LDN variant adds:

```
                                              ┌──► [LDN reward] ──► [decoder] ──┐
reward ────────────────────────────────►──────┘                                  │
                                                                                 ▼
state value (fed back) ────► [LDN value] ──► [decoder] ──► [rule with multi-step return]
```

The decoder weights for the reward LDN are computed once at network construction, from the Legendre coefficients of the discount kernel — either a discrete sum (`continuous=False`) or a continuous integral (`continuous=True`).

## What's not vendored

To keep this submission focused, the following parts of upstream `rlnet/` are **not** included:

- `sspspace.py` (Spatial Semantic Pointer construction) — large; warrants its own submission. See the existing `ssp-path-integrator` zoo entry for an SSP-based network.
- `representations/ssp.py`, `representations/vsa.py`, `representations/onehottransform.py` — depend on SSP/VSA machinery or RatBox-specific transforms.
- `rules/td0iG.py`, `rules/tdlambda.py`, `rules/tdn.py` — TD variants that overlap heavily with `TD0` and `TDt`.
- The `cartpoleExperiments/`, `ratboxExperiments/`, `figures/`, NNI hyperparameter configs, and the upstream `utils.py` plotting helpers (`plot_policy`, `get_ac_output`, `save_gifs`, etc.) — environment- and analysis-specific.

If you want any of these as a separate submission, the upstream paths above are the starting point.

## Citation

```bibtex
@inproceedings{bartlett2022rl,
  author    = {Bartlett, Maddy and Simone, Kathryn and Dumont, Nicole Sandra-Yaffa and
               Furlong, P. Michael and Eliasmith, Chris and Orchard, Jeff and Stewart, Terry},
  title     = {Improving Reinforcement Learning with Biologically Motivated
               Continuous State Representations},
  booktitle = {Proceedings of the 20th International Conference on Cognitive Modeling
               (ICCM 2022)},
  year      = {2022},
  url       = {https://mathpsych.org/presentation/1221},
}
```

## License

Applied Brain Research academic-use-only — see `LICENSE` for the full upstream terms. **This is not a free-software license**; commercial use requires a paid ABR license.

"""Tier-1 sanity tests for the nengo-a2c submission."""

import numpy as np

from nengo_a2c import (
    ActorCritic,
    ActorCriticLDN,
    LDN,
    NormalRep,
    OneHotRepCP,
    OneHotRepRB,
    TD0,
    TDt,
    softmax,
    sparsity_to_x_intercept,
)


def test_imports():
    assert ActorCritic is not None
    assert ActorCriticLDN is not None
    assert LDN is not None
    assert TD0 is not None
    assert TDt is not None
    assert NormalRep is not None
    assert OneHotRepCP is not None
    assert OneHotRepRB is not None
    assert callable(softmax)
    assert callable(sparsity_to_x_intercept)


def test_actor_critic_builds_and_steps():
    """Basic A2C builds, accepts a few steps, and emits correctly-shaped outputs."""
    rep = OneHotRepCP(ranges=(3, 3))
    rule = TD0(n_actions=2, lr=0.1, act_dis=0.9, state_dis=0.95)
    ac = ActorCritic(rep, rule)

    sv, av = ac.step((0, 0), 0, reward=0.0, reset=True)
    assert sv.shape == (1,)
    assert av.shape == (2,)

    for _ in range(5):
        sv, av = ac.step((0, 0), 0, reward=1.0, reset=False)
    # After several positive-reward updates the weight row for the visited
    # state must be non-zero (otherwise the rule never fired).
    assert not np.allclose(av, 0), "action values still zero after reward updates"


def test_actor_critic_with_state_ensemble_builds():
    """A2C with a state ensemble in front of the rule node."""
    rep = OneHotRepCP(ranges=(3, 3))
    rule = TD0(n_actions=2, lr=0.1)
    ac = ActorCritic(rep, rule, state_neurons=30, active_prop=0.2)
    sv, av = ac.step((0, 0), 0, reward=0.0, reset=True)
    sv, av = ac.step((0, 0), 0, reward=0.0)
    assert sv.shape == (1,)
    assert av.shape == (2,)


def test_actor_critic_ldn_builds_and_steps():
    """LDN variant builds with the TDt rule and emits correctly-shaped outputs."""
    rep = OneHotRepCP(ranges=(3, 3))
    rule = TDt(n_actions=2, lr=0.1, act_dis=0.85, state_dis=0.9, n=2, env_dt=0.001)
    ac = ActorCriticLDN(
        rep, rule,
        state_neurons=30, active_prop=0.2,
        theta=0.005, q_r=3, q_v=3,
        continuous=False,
    )
    sv, av = ac.step((0, 0), 0, reward=0.0, reset=True)
    sv, av = ac.step((0, 0), 0, reward=0.5)
    assert sv.shape == (1,)
    assert av.shape == (2,)


def test_ldn_process_runs_standalone():
    """The LDN nengo.Process builds and produces q*size_in-shaped output per step."""
    import nengo
    ldn = LDN(theta=0.05, q=4, size_in=1)
    with nengo.Network() as net:
        u = nengo.Node(lambda t: np.sin(2 * np.pi * 5 * t))
        memory = nengo.Node(ldn, size_in=1)
        nengo.Connection(u, memory, synapse=None)
        probe = nengo.Probe(memory, synapse=None)
    with nengo.Simulator(net) as sim:
        sim.run(0.05)
    # 50 ms / 1 ms timestep = 50 samples; q=4 dims.
    assert sim.data[probe].shape == (50, 4)

"""
Vendored from https://github.com/maddybartlett/ImprovedRLContinuousStateReps
(network/rlnet/networks/acBasic.py), commit fc75ae0 (2024-02-01).

This file is © Maddy Bartlett et al. and distributed under the Applied Brain
Research academic-use-only license — see LICENSE at the submission root.

Local edits relative to upstream:
  - Adjusted the rlnet.utils import to a relative import (..utils).
  - Removed unused top-level `import matplotlib.pyplot as plt`.
  - Removed `get_tuning()` and `get_policy()` methods. The former is a
    one-liner plotting helper; the latter hardcodes the RatBox 8×8×4 grid
    geometry and isn't generically reusable.
"""

import nengo
import numpy as np

from ..utils import sparsity_to_x_intercept


## Actor-Critic without LDNs ##
class ActorCritic(object):
    ''' Nengo model implementing an Actor-Critic network.
    Single-layer network
    Inputs: state, action, reward and reset
    Outputs: updated state value, action values for actions available in current state

    Example of Usage:
        >> rep = NormalRep((8,8,4))
        >> ac = ActorCritic(rep,
                 ActorCriticTD0(n_actions=3, alpha=0.1, beta=0.9, gamma=0.95),state_neurons
                 state_neurons=1000,
                 neuron_type=nengo.RectifiedLinear()
                 intercepts=nengo.dists.Uniform(0.01, 0.5)
                )
    '''
    def __init__(self, representation, rule, state_neurons=None, active_prop=None, neuron_type=nengo.RectifiedLinear(),
                 **ensemble_args):

        self.representation = representation
        ## set dim = size of state representation
        dim = representation.size_out
        ## create empty array for action values being updated
        self.update_action = np.zeros(rule.n_actions)
        ## ensemble
        self.state_neurons = state_neurons
        self.active_prop = active_prop
        self.neuron_type = neuron_type

        ## empty arrays for state value and action values
        self.state_value = np.zeros(1)
        self.action_values = np.zeros(rule.n_actions)

        ## Create nengo model
        self.model = nengo.Network()
        with self.model:

            ## empty array for state
            ## size = size of state representation + number of actions + reward + whether env was reset
            self.state = np.zeros(dim+rule.n_actions+2)
            ## create nengo node for containing state
            self.state_node = nengo.Node(lambda t: self.state)

            ## if we're not using a neuron ensemble to contain the state representation
            if state_neurons is None:
                ## create nengo node for containing the learning rule
                ## input size in is dim (state) + n_actions + reward + env.reset
                self.rule = nengo.Node(rule, size_in=dim+rule.n_actions+2)
                ## connect the state node to the rule node
                nengo.Connection(self.state_node, self.rule, synapse=None)

            ## if we are using a neuron ensemble
            else:
                ## create nengo node for containing the learning rule
                ## input size in is state_neurons (state) + n_actions + reward + env.reset
                self.rule = nengo.Node(rule, size_in=self.state_neurons+rule.n_actions+2)
                ## create ensemble for containing the state representation
                self.ensemble = nengo.Ensemble(n_neurons=state_neurons, dimensions=dim,
                                               neuron_type=self.neuron_type,
                                               intercepts=nengo.dists.Choice([sparsity_to_x_intercept(dim, self.active_prop)]),
                                               **ensemble_args
                                              )
                ##connect the state representation to the ensemble
                nengo.Connection(self.state_node[:dim], self.ensemble, synapse=None)
                ##connect the state ensemble to the rule node
                nengo.Connection(self.ensemble.neurons, self.rule[:state_neurons], synapse=None)
                ##connect the state representation to the rule node
                nengo.Connection(self.state_node[dim:], self.rule[state_neurons:], synapse=None)

            ## create node for containing the updated state value
            self.value_node = nengo.Node(self.value_node_func, size_in=1)

            ## create node for containing the updated action values
            self.action_values_node = nengo.Node(self.action_values_node_func, size_in=rule.n_actions)

            ## send first output from rule node (updated state value) to the state value node
            nengo.Connection(self.rule[0], self.value_node, synapse=None)
            ## send updated action values from rule node to action value node
            nengo.Connection(self.rule[1:], self.action_values_node, synapse=None)

        ## run model
        self.sim = nengo.Simulator(self.model)

    def step(self, state, update_action, reward, reset=False):
        '''Function for running the model for one time step.

        Inputs: agent's state, chosen action, reward
        Outputs: state value, action values'''

        if type(update_action) == int or type(update_action) == np.int64 or len(update_action) == 1:
            ## set update_action to an array of 0's with one value for each action
            self.update_action[:] = 0
            ## set the update_action value at the position of the chosen action to 1
            self.update_action[update_action] = 1
        else:
            self.update_action = update_action

        ## create state variable containing state representation,
        ## update_action array, reward, and whether or not the env was reset
        self.state[:] = np.concatenate([
            self.representation.map(state),
            self.update_action,
            [reward, reset],]
            )

        ## run model for one step
        self.sim.step()

        ## return the updated state and action values from the model
        ## these are the values returned by the learning rule
        return self.state_value, self.action_values

    ## function for state value node
    def value_node_func(self, t, x):
        ## identity function
        self.state_value[:] = x

    ## function for action value node
    def action_values_node_func(self, t, x):
        ## identity function
        self.action_values[:] = x

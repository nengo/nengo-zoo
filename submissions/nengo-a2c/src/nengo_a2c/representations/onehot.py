"""
Vendored from https://github.com/maddybartlett/ImprovedRLContinuousStateReps
(network/rlnet/representations/onehot.py), commit fc75ae0 (2024-02-01).

This file is © Maddy Bartlett et al. and distributed under the Applied Brain
Research academic-use-only license — see LICENSE at the submission root.

Local edits relative to upstream:
  - OneHotRepRB.get_state: `np.int` → `int`. `np.int` was deprecated in
    NumPy 1.20 and removed in NumPy 1.24; bare `int` is the long-standing
    NumPy-recommended replacement and works on every NumPy release the
    rest of this submission supports.
"""

import numpy as np


class OneHotRepRB(object):
    '''Create one-hot representation. I.e. the state is represented as a list of 0's and a 1.
    This method works with RatBox.
    '''
    def __init__(self, ranges):
        ##step 1
        self.ranges = ranges
        self.factors = np.array([np.prod(ranges[x+1:]) for x in range(len(ranges))], dtype=int)
        ##step 2
        self.result = np.zeros(np.prod(ranges))
        self.size_out = len(self.result)

    def map(self, state):
        index = 0
        ##step 3
        for i, v in enumerate(state):
            index += v*self.factors[i]
        self.result[:] = 0
        ##step 4
        if index % 1 == 0:
            index = int(index)
        self.result[index] = 1
        return self.result

    def get_state(self, state, env=None):
        discrete_state = state

        state_space_high = np.asarray([env.width, env.height, 360])
        state_space_low = np.asarray([0, 0, 0])

        discrete_obs_size = np.array(self.ranges)

        discrete_obs_size = np.array(discrete_obs_size, dtype='int')
        discrete_obs_win_size = (state_space_high - state_space_low)/discrete_obs_size

        discrete_state = (state - state_space_low)/discrete_obs_win_size
        discrete_state = tuple(discrete_state.astype(int))

        return discrete_state


class OneHotRepCP(object):
    '''Create one-hot representation. I.e. the state is represented as a list of 0's and a 1.
    This method works with CartPole v1.
    '''
    def __init__(self, ranges):
        ##step 1
        self.ranges = ranges
        self.factors = np.array([np.prod(ranges[x+1:]) for x in range(len(ranges))], dtype=int)
        ##step 2
        self.result = np.zeros(np.prod(ranges))
        self.size_out = len(self.result)

    def map(self, state):
        index = 0
        ##step 3
        for i, v in enumerate(state):
            index += v*self.factors[i]
        self.result[:] = 0
        ##step 4
        if index % 1 == 0:
            index = int(index)
        self.result[index] = 1
        return self.result

    def get_state(self, state, env=None):

        discrete_state = state

        state_space_low = env.observation_space.low
        state_space_high = env.observation_space.high

        discrete_obs_size = np.array(self.ranges)

        discrete_obs_size = np.array(discrete_obs_size, dtype='int')
        discrete_obs_win_size = (state_space_high - state_space_low) / discrete_obs_size

        discrete_state = (state - state_space_low)/discrete_obs_win_size
        discrete_state = tuple(discrete_state.astype(int))

        return discrete_state

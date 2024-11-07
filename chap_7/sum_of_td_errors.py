# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:44:46 2024

@author: ccw
"""
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from env.randomWalk import RandomWalk


class Memory:
    """Get the value in memory base on the time"""

    def __init__(self, length, dtype=float):
        self._ar = np.empty(length, dtype=dtype)

    def _t2i(self, t):
        return t % (len(self._ar))

    def __setitem__(self, t, value):
        self._ar[self._t2i(t)] = value

    def __getitem__(self, t):
        """Get the value from array at time t"""
        return self._ar[self._t2i(t)]

    def __repr__(self):
        return self._ar.__repr__()


class NStepTD:

    def __init__(self, env, n, step_size=.5, discount_rate=1, seed=None,
                 approximate=False):
        """When approximate equals True, N step TD error is replaced by
        sum of 1 step TD error"""
        self._env = env
        self._n = n  # n step TD
        self._V = np.zeros_like(self._env.observation_space, dtype=float)
        self._alpha = step_size
        self._gamma = discount_rate

        self._rng = np.random.default_rng(seed)
        self._approximate = approximate

    def episode_prediction(self):

        def delta(t):
            return R[t + 1] + self._gamma * self._V[S[t + 1]] - self._V[S[t]]

        n = self._n
        S = Memory(n + 1, dtype=int)
        R = Memory(n + 1, dtype=float)

        s, terminated = self._env.reset(self._rng.integers(100000))
        T = np.inf
        t = 0
        S[t] = s
        while True:
            # save state and reward for value updating
            if t < T:
                s, r, terminated = self._env.step()
                S[t + 1] = s
                R[t + 1] = r
                if terminated:
                    T = t + 1
            # value updating
            tou = t - n + 1
            if tou >= 0:  # can update
                if self._approximate:
                    n_td_error = sum(
                        delta(i) for i in range(tou, min(tou + n, T)))
                else:
                    G = sum(self._gamma ** (i - tou - 1) * R[i]
                            for i in range(tou + 1, min(tou + n, T) + 1))
                    if tou + n < T:
                        G += (self._gamma ** n * self._V[S[tou + n]])
                    n_td_error = G - self._V[S[tou]]

                self._V[S[tou]] += self._alpha * n_td_error
            t += 1
            if tou == T - 1:
                break

    @property
    def state_values(self):
        return self._V[1: -1]


def validate():

    def cal_rmse(v_0: np.ndarray, v_1: np.ndarray):
        return np.sqrt(np.power(v_0 - v_1, 2).mean())

    env = RandomWalk(19)

    agents = dict(
        exact=NStepTD(env, n=8, step_size=.01, seed=12345),
        approximate=NStepTD(env, n=8, step_size=.01, seed=12345,
                            approximate=True))
    rmse = {name: list() for name in agents.keys()}

    episodes = range(1, 700)
    for episode in tqdm(episodes):
        for name, agent in agents.items():
            agent.episode_prediction()
            v = cal_rmse(agent.state_values, env.true_value)
            rmse[name].append(v)
    # plot
    fig, ax = plt.subplots()
    for name, data in rmse.items():
        ax.plot(episodes, data, label=name)
    ax.legend()


if __name__ == '__main__':
    m = Memory(3)
    for t in range(6):
        m[t] = t
        print(m[t])

    validate()

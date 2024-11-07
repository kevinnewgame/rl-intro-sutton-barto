# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:27:26 2024

@author: ccw
"""
import numpy as np
from itertools import product
from multiprocessing import Pool
from tqdm import tqdm

import matplotlib.pyplot as plt

from env.randomWalk import RandomWalk


class MRP:

    def __init__(self):
        self._mrp = list()
        self._t = -1

    def append(self, s, r):
        self._t += 1
        self._mrp.append((self._t, s, r))

    def __getitem__(self, i):
        return self._mrp[i]

    @property
    def T(self):
        return self._t


def generate_mrp(env: RandomWalk, n_episode, seed=None) -> list[MRP]:
    rng = np.random.default_rng(seed)

    mrps = list()
    for episode in range(n_episode):
        mrp = MRP()
        s, terminated = env.reset(rng.integers(100000))
        mrp.append(s, np.nan)
        while not terminated:
            s, r, terminated = env.step()
            mrp.append(s, r)
        mrps.append(mrp)
    return mrps


class NStepTD:

    def __init__(self, n, step_size, values: np.ndarray, discount_rate=1):
        self._V = values.copy()
        self._n = n
        self._alpha = step_size
        self._gamma = discount_rate

    def prediction(self, mrps: list[MRP]):
        for mrp in mrps:
            T = mrp.T
            for t, s, _ in mrp:
                mrp_use = mrp[t + 1: min(t + self._n, T) + 1]
                rs = [r for _, _, r in mrp_use]  # rewards
                if t + self._n < T:
                    last_s = mrp_use[-1][1]
                    rs.append(self._V[last_s])
                rs = np.array(rs)
                gammas = np.array([self._gamma ** x for x in range(len(rs))])  # gammas
                G = (rs * gammas).sum()
                td_error = G - self._V[s]
                self._V[s] += self._alpha * td_error
            yield self.state_values

    @property
    def state_values(self):
        return self._V[1: -1]


def cal_rmse(n_step, alpha, values, mrps, true_value):

    def rmse(v_0: np.ndarray, v_1: np.ndarray):
        return np.sqrt(np.power(v_0 - v_1, 2).mean())

    agent = NStepTD(n=n_step, step_size=alpha, values=values)
    episode_rmse = [rmse(v, true_value) for v in agent.prediction(mrps)]
    return sum(episode_rmse) / len(mrps)


def figure_7_2(seed=None, nr_parallel_processes=4):
    env = RandomWalk(19)
    init_state_values = np.zeros_like(env.observation_space, dtype=float)
    n_run = 100
    alphas = np.power(np.linspace(.1, 1, 20), 2)
    n_steps = [2 ** i for i in range(10)]  # n step TD
    rng = np.random.default_rng(seed)
    result = np.zeros((len(alphas), len(n_steps)))

    for run in tqdm(range(1, n_run + 1)):
        seed = rng.integers(100000)
        # Markov reward processes
        mrps = generate_mrp(env, n_episode=10, seed=seed)

        with Pool(processes=nr_parallel_processes) as pool:
            params = [(n_step, alpha, init_state_values, mrps, env.true_value)
                      for alpha, n_step in product(alphas, n_steps)]
            res = pool.starmap(cal_rmse, params)
            res = np.array(res).reshape(result.shape)

        result += (res - result) / run

    # plot
    fig, ax = plt.subplots()
    ax.plot(alphas, result)
    ax.set_ylim(.25, .55)

    return result


if __name__ == '__main__':
    # env = RandomWalk(19)
    # agent = NStepTD(3, .1, np.zeros_like(env.observation_space, dtype=float))

    # # markov reward processes
    # mrps = generate_mrp(env, 10, seed=12345)
    # print("terminal time:", mrps[0].T)
    # print("t, s, r in time zero", mrps[0][0])
    # for t, s, r in mrps[0][:10]:
    #     print(t, s, r)

    # for v in agent.prediction(mrps):
    #     print(v[:5])

    result = figure_7_2()

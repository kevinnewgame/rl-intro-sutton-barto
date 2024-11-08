# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:47:43 2024

@author: ccw
"""
import numpy as np
import gymnasium as gym

from multiprocessing import Pool
from functools import partial

import matplotlib.pyplot as plt

from algorithm import simple_bandit_algorithm


gym.envs.registration.register(
    id="k-armed-testbed-v0",
    entry_point="rl_intro_env.gymnasium_env.envs:KArmedTestbed",
    )


def get_result(env, params, seed=12345, **kwargs):
    res = {name: simple_bandit_algorithm(env, seed=seed, **param, **kwargs)
           for name, param in params.items()}
    return res


def experiment(env, params, n_run=2000, seed=None, nr_parallel_processes=4,
               **kwargs):
    f = partial(get_result, env, params, **kwargs)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(100000, size=n_run)
    with Pool(processes=nr_parallel_processes) as pool:
        res = pool.map(f, seeds)

    # organize data
    data = dict()
    for name, d in res[0].items():
        data[name] = dict()
        for stat, ar in d.items():
            data[name][stat] = np.zeros_like(ar, dtype=float)

    for i, d0 in enumerate(res, 1):
        for name, d1 in d0.items():
            for stat, d in d1.items():
                data[name][stat] += (np.array(d) - data[name][stat]) / i
    return data


def plot(data, out_path, dpi=65):
    n_stat = len(data[list(data.keys())[0]])
    fig, axs = plt.subplots(nrows=n_stat, ncols=1)

    for name, d0 in data.items():
        for i, (stat, d) in enumerate(d0.items()):
            axs[i].plot(d, label=name)
    axs[0].legend()
    plt.savefig(out_path, dpi=dpi)


def figure_2_2(seed=None):
    env = gym.make("k-armed-testbed-v0", k=10)
    params = {str(eps): {"eps": eps} for eps in (0, 0.01, 0.1)}
    data = experiment(env, params, seed=seed)
    plot(data, "plots/figure_2_2.png")


def exercise_2_5(seed=None):
    env = gym.make("k-armed-testbed-v0", k=10, stationary=False)
    params = {"mean": {"alpha": None},
              "$\\alpha=0.1$": {"alpha": 0.1}}
    data = experiment(env, params, seed=seed, n_run=1000, n_steps=10000,
                      nr_parallel_processes=6)
    plot(data, "plots/exercise_2_5.png")


def figure_2_3(seed=None):
    env = gym.make('k-armed-testbed-v0', k=10, stationary=True)
    params = {"0": {"init_value": 0},
              "5": {"init_value": 5}}
    data = experiment(env, params, seed=seed, n_run=2000,
                      n_steps=1000, eps=0.1, alpha=0.1,)
    plot(data, "plots/figure_2_3.png")


def figure_2_4(seed=None):
    env = gym.make('k-armed-testbed-v0', k=10, stationary=True)
    params = {"$\\epsilon$=0.1": {"ucb": False, "eps": 0.1},
              "ubc2": {"ucb": True, "c": 2}, }
    data = experiment(env, params, seed=seed, n_run=2000,
                      n_steps=1000, alpha=None, init_value=0)
    plot(data, "plots/figure_2_4.png")


if __name__ == '__main__':
    figure_2_2(12345)
    exercise_2_5(12345)
    figure_2_3(12345)
    figure_2_4(12345)

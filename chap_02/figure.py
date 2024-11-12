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

from algorithm import simple_bandit_algorithm, gradient_bandit_algorithm


gym.envs.registration.register(
    id="k-armed-testbed-v0",
    entry_point="rl_intro_env.gymnasium_env.envs:KArmedTestbed",)


def get_result(env, alg, params, seed=12345, **kwargs):
    res = {name: alg(env, seed=seed, **param, **kwargs)
           for name, param in params.items()}
    return res


def experiment(env, alg, params, n_run=2000, seed=None,
               nr_parallel_processes=4, **kwargs):
    f = partial(get_result, env, alg, params, **kwargs)
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
    data = experiment(env, simple_bandit_algorithm, params, seed=seed)
    plot(data, "plots/figure_2_2.png")


def exercise_2_5(seed=None):
    env = gym.make("k-armed-testbed-v0", k=10, stationary=False)
    params = {"mean": {"alpha": None},
              "$\\alpha=0.1$": {"alpha": 0.1}}
    data = experiment(env, simple_bandit_algorithm, params, seed=seed,
                      n_run=1000, n_steps=10000, nr_parallel_processes=6)
    plot(data, "plots/exercise_2_05.png")


def figure_2_3(seed=None):
    env = gym.make('k-armed-testbed-v0', k=10, stationary=True)
    params = {"0": {"init_value": 0},
              "5": {"init_value": 5}}
    data = experiment(env, simple_bandit_algorithm, params, seed=seed,
                      n_run=2000, n_steps=1000, eps=0.1, alpha=0.1,)
    plot(data, "plots/figure_2_3.png")


def figure_2_4(seed=None):
    env = gym.make('k-armed-testbed-v0', k=10, stationary=True)
    params = {"$\\epsilon$=0.1": {"ucb": False, "eps": 0.1},
              "ubc2": {"ucb": True, "c": 2}, }
    data = experiment(env, simple_bandit_algorithm, params, seed=seed,
                      n_run=2000, n_steps=1000, alpha=None, init_value=0)
    plot(data, "plots/figure_2_4.png")


def figure_2_5(seed=None):
    env = gym.make('k-armed-testbed-v0', k=10, stationary=True, loc=4)
    params = {
        "b $\\alpha=0.1$": {"baseline": True, "alpha_h": 0.1},
        "b $\\alpha=0.4$": {"baseline": True, "alpha_h": 0.4},
        "nb $\\alpha=0.1$": {"baseline": False, "alpha_h": 0.1},
        "nb $\\alpha=0.4$": {"baseline": False, "alpha_h": 0.4},
        }
    data = experiment(env, gradient_bandit_algorithm, params, n_run=2000,
                      seed=seed, stationary=True)
    plot(data, "plots/figure_2_5.png")


def figure_2_6(out_path, seed=None, n_run=2000, nr_parallel_processes=6,
               dpi=65):

    def first_n_steps_average(res):
        return np.mean(res["reward"])

    rng = np.random.default_rng(seed)
    env = gym.make('k-armed-testbed-v0', k=10, stationary=True)
    n_steps = 1000

    fig, ax = plt.subplots()

    # 1. epsilon-greedy
    epsilons = [2 ** p for p in range(-7, -1)]
    mean_rewards = np.empty(len(epsilons))

    for i, eps in enumerate(epsilons):
        seeds = rng.integers(100000, size=n_run)
        # func params: env, n_steps, alpha, init_value, eps, ucb, c, seed
        params = [(env, n_steps, None, 0, eps, False, 1, seed) for seed in seeds]
        with Pool(processes=nr_parallel_processes) as pool:
            results = pool.starmap(simple_bandit_algorithm, params)

        mr = np.mean([first_n_steps_average(res) for res in results])
        mean_rewards[i] = mr

    ax.plot(epsilons, mean_rewards, c="red", lw=1)

    # 2. greedy with optimistic initialization
    init_values = [2 ** p for p in range(-2, 3)]
    mean_rewards = np.empty(len(init_values))

    for i, v in enumerate(init_values):
        seeds = rng.integers(100000, size=n_run)
        # func params: env, n_steps, alpha, init_value, eps, ucb, c, seed
        params = [(env, n_steps, 0.1, v, 0, False, 1, seed) for seed in seeds]
        with Pool(processes=nr_parallel_processes) as pool:
            results = pool.starmap(simple_bandit_algorithm, params)

        mr = np.mean([first_n_steps_average(res) for res in results])
        mean_rewards[i] = mr

    ax.plot(init_values, mean_rewards, c="black", lw=1)

    # 3. UCB
    cs = [2 ** p for p in range(-4, 5)]
    mean_rewards = np.empty(len(cs))

    for i, c in enumerate(cs):
        seeds = rng.integers(100000, size=n_run)
        # func params: env, n_steps, alpha, init_value, eps, ucb, c, seed
        params = [(env, n_steps, None, 0, 0, True, c, seed) for seed in seeds]
        with Pool(processes=nr_parallel_processes) as pool:
            results = pool.starmap(simple_bandit_algorithm, params)

        mr = np.mean([first_n_steps_average(res) for res in results])
        mean_rewards[i] = mr

    ax.plot(cs, mean_rewards, c="blue", lw=1)

    # 4. gradient bandit
    alphas = [2 ** p for p in range(-5, 3)]
    mean_rewards = np.empty(len(alphas))

    for i, alpha in enumerate(alphas):
        seeds = rng.integers(100000, size=n_run)
        # func params: env, n_steps, alpha_h, baseline, stationary, alpha_r, seed
        params = [(env, n_steps, alpha, True, True, None, seed) for seed in seeds]
        with Pool(processes=nr_parallel_processes) as pool:
            results = pool.starmap(gradient_bandit_algorithm, params)

        mr = np.mean([first_n_steps_average(res) for res in results])
        mean_rewards[i] = mr

    ax.plot(alphas, mean_rewards, c="green", lw=1)

    xticks = [2 ** p for p in range(-7, 3)]
    ax.set_ylim(1 * 0.98, 1.5 * 1.02)
    ax.set_xlim(xticks[0] * 1.02, xticks[-1] * 1.02)
    ax.set_xticks(xticks)
    ax.set_xscale('log', base=2)
    ax.grid()

    plt.savefig(out_path, dpi=dpi)


def exercise_2_11(
        out_path, seed=None, n_run=1, nr_parallel_processes=4, dpi=65):

    def get_stat(res):
        return np.mean(res["reward"][int(n_steps/2):])

    # nonstationary environment
    env = gym.make('k-armed-testbed-v0', k=10, stationary=False)
    n_steps = 200000

    fig, ax = plt.subplots()

    # 1. epsilon-greedy
    epsilons = [2 ** p for p in range(-7, -1)]
    mean_rewards = np.empty(len(epsilons))

    for i, eps in enumerate(epsilons):
        # func params: env, n_steps, alpha, init_value, eps, ucb, c, seed
        res = simple_bandit_algorithm(env, n_steps, None, 0, eps, False, 1, seed)
        mean_rewards[i] = get_stat(res)

    ax.plot(epsilons, mean_rewards, c="red", lw=1)

    # 2. greedy with optimistic initialization
    init_values = [2 ** p for p in range(-2, 3)]
    mean_rewards = np.empty(len(init_values))

    for i, v in enumerate(init_values):
        # func params: env, n_steps, alpha, init_value, eps, ucb, c, seed
        res = simple_bandit_algorithm(env, n_steps, 0.1, v, 0, False, 1, seed)
        mean_rewards[i] = get_stat(res)

    ax.plot(init_values, mean_rewards, c="black", lw=1)

    # 3. UCB
    cs = [2 ** p for p in range(-4, 5)]
    mean_rewards = np.empty(len(cs))

    for i, c in enumerate(cs):
        # func params: env, n_steps, alpha, init_value, eps, ucb, c, seed
        res = simple_bandit_algorithm(env, n_steps, None, 0, 0, True, c, seed)
        mean_rewards[i] = get_stat(res)

    ax.plot(cs, mean_rewards, c="blue", lw=1)

    # 4. gradient bandit
    alphas = [2 ** p for p in range(-5, 3)]
    mean_rewards = np.empty(len(alphas))

    for i, alpha in enumerate(alphas):
        # func params: env, n_steps, alpha_h, baseline, stationary, alpha_r, seed
        res = gradient_bandit_algorithm(env, n_steps, alpha, True, True, None, seed)
        mean_rewards[i] = get_stat(res)

    ax.plot(alphas, mean_rewards, c="green", lw=1)

    # 5. constant step (alpha = 0.1) epsilon-greedy
    epsilons = [2 ** p for p in range(-7, -1)]
    mean_rewards = np.empty(len(epsilons))

    for i, eps in enumerate(epsilons):
        # func params: env, n_steps, alpha, init_value, eps, ucb, c, seed
        res = simple_bandit_algorithm(env, n_steps, 0.1, 0, eps, False, 1, seed)
        mean_rewards[i] = get_stat(res)

    ax.plot(epsilons, mean_rewards, c="orange", lw=1)

    xticks = [2 ** p for p in range(-7, 3)]
    ax.set_xticks(xticks)
    ax.set_xscale('log', base=2)
    ax.grid()
    plt.savefig(out_path, dpi=dpi)


if __name__ == '__main__':
    figure_2_2(12345)
    exercise_2_5(12345)
    figure_2_3(12345)
    figure_2_4(12345)
    figure_2_5(12345)
    figure_2_6("plots/figure_2_6.png", seed=12345)
    exercise_2_11("plots/exercise_2_11.png", seed=12345)

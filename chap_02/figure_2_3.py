# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:47:43 2024

@author: ccw
"""
import numpy as np
import gymnasium as gym

from multiprocessing import Pool

import matplotlib.pyplot as plt

from agent import AgentSimpleBandit


gym.envs.registration.register(
    id="k-armed-testbed-v0",
    entry_point="rl_intro_env.gymnasium_env.envs:KArmedTestbed",
    )
ENV = gym.make('k-armed-testbed-v0', k=10, stationary=True)


def simple_bandit_algorithm(
        n_steps=1000, eps=0.1, alpha=0.1, init_value=0, seed=12345):
    res = {"reward": list(), "opt_act": list()}

    agent = AgentSimpleBandit(
        ENV, seed=seed * 2, eps=eps, alpha=alpha, init_value=init_value)
    ENV.reset(seed=seed)
    for step in range(n_steps):
        a = agent.get_action()
        _, r, _, _, info = ENV.step(a)
        agent.update(a, r)

        res["reward"].append(r)
        res["opt_act"].append(a == info["best_arm"])
    return res


def experiment(seed=12345):
    init_values = [0, 5]
    res = {v: simple_bandit_algorithm(init_value=v, seed=seed) for v in init_values}
    # take out rewards
    rewards = np.stack([res["reward"] for alpha, res in res.items()], axis=1)
    # take out is optimal action
    opt_acts = np.stack([res["opt_act"] for alpha, res in res.items()], axis=1)
    return {"reward": rewards, "opt_act": opt_acts}


def plot(out_path, n_run=2000, seed=None, nr_parallel_processes=4):

    def average(attr):
        return np.stack([d[attr] for d in res]).mean(axis=0)

    # get data
    rng = np.random.default_rng(seed)
    seeds = rng.integers(100000, size=n_run)
    with Pool(processes=nr_parallel_processes) as pool:
        res = pool.map(experiment, seeds)

    reward = average("reward")
    opt_act = average("opt_act")

    # plot
    fig, axs = plt.subplots(nrows=2, ncols=1)
    for label, col in zip(["q*0", "q*5"], range(reward.shape[1])):
        axs[0].plot(reward[:, col], label=label)
        axs[1].plot(opt_act[:, col], label=label)
    axs[0].legend()
    plt.savefig(out_path, dpi=65)
    return reward, opt_act


if __name__ == '__main__':
    # res = simple_bandit_algorithm()
    # res_1 = simple_bandit_algorithm(init_value=5)

    # res = experiment()

    data = plot("plots/figure_2_3.png", seed=12345, nr_parallel_processes=4)
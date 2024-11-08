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
ENV = gym.make('k-armed-testbed-v0', k=10)


def simple_bandit_algorithm(n_steps=1000, seed=12345, eps=0.1):
    info_ = {"reward": list(), "action": list(), "best_arm": int()}

    agent = AgentSimpleBandit(ENV, seed=seed * 2, eps=eps)
    ENV.reset(seed=seed)
    for step in range(n_steps):
        a = agent.get_action()
        _, r, _, _, info = ENV.step(a)
        agent.update(a, r)

        info_["reward"].append(r)
        info_["action"].append(a)
    info_["best_arm"] = info["best_arm"]
    return agent, info_


def experiment(seed=12345):
    epss = [0, 0.01, 0.1]
    res = {e: simple_bandit_algorithm(eps=e, seed=seed) for e in epss}
    # take out rewards
    rewards = np.stack(
        [info["reward"] for e, (agent, info) in res.items()],
        axis=1)
    # take out is optimal action
    opt_acts = np.stack(
        [np.array(info["action"]) == info["best_arm"]
         for e, (agent, info) in res.items()],
        axis=1)
    return {"reward": rewards, "opt_act": opt_acts}


def figure_2_2(n_run=2000, seed=None, nr_parallel_processes=4):

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
    axs[0].plot(reward)
    axs[1].plot(opt_act)
    plt.savefig('plots/figure_2_2.png', dpi=65)
    return reward, opt_act


if __name__ == '__main__':
    # agent, info = simple_bandit_algorithm()
    # res = experiment()
    data = figure_2_2(seed=12345)

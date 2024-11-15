# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:25:45 2024

@author: ccw
"""
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from multiprocessing import Pool
from functools import partial

import gymnasium as gym

from algorithm import AgentMC, generate_episode

import matplotlib.pyplot as plt
# from matplotlib import colors


gym.envs.registration.register(
    id="blackjack",
    entry_point="rl_intro_env.gymnasium_env.envs:Blackjack",)


def set_ticks(ax):
    # dealer's flip up
    ax.set_xticks(range(10))
    labels = [str(i) if i != 1 else "A" for i in range(1, 11)]
    ax.set_xticklabels(labels)

    # agent's points
    ax.set_yticks(range(10))
    labels = [str(i) for i in range(12, 22)]
    ax.set_yticklabels(labels)


def figure_5_1(out_path, seed=None, dpi=65):

    def policy(s):
        agent_point = s[0]
        return 1 if agent_point < 20 else 0

    def v2ar(d: dict) -> np.ndarray:
        array = np.zeros((2, 10, 10))
        for (point, usable_ace, flip), v in d.items():
            array[usable_ace, point - 12, flip - 1] = v
        return array

    fig, axs = plt.subplots(2, 2, figsize=(8, 7))

    env = gym.make("blackjack")
    agent = AgentMC(env)
    rng = np.random.default_rng(seed)
    vs = list()

    for _ in tqdm(range(10000)):
        episode = generate_episode(env, policy, rng.integers(100000))
        agent.policy_evaluation(episode)

    v = v2ar(agent.V)
    vs.append(v)
    for usable_ace in range(2):
        axs[1 - usable_ace, 0].imshow(v[usable_ace], origin='lower')

    for _ in tqdm(range(490000)):
        episode = generate_episode(env, policy, rng.integers(100000))
        agent.policy_evaluation(episode)

    v = v2ar(agent.V)
    vs.append(v)
    for usable_ace in range(2):
        axs[1 - usable_ace, 1].imshow(v[usable_ace], origin='lower')

    for row in axs:
        for ax in row:
            set_ticks(ax)

    plt.savefig(out_path, dpi=dpi)
    return vs


def figure_5_2(out_path, seed=None, agent=None, dpi=65, n_episode=3000000):

    def policy(s):
        return agent.Q[s].argmax()

    def q2pi(q: dict) -> np.ndarray:
        """return policy and optimal value in array"""
        ar = np.zeros((2, 2, 10, 10))  # usable_ace, pi | v, ...
        for (point, usable_ace, flip), av in q.items():
            ar[usable_ace, 0, point - 12, flip - 1] = av.argmax()  # policy
            ar[usable_ace, 1, point - 12, flip - 1] = av.max()  # optimal value
        return ar

    env = gym.make("blackjack", es=True)
    rng = np.random.default_rng(seed)
    agent = AgentMC(env)

    for _ in tqdm(range(n_episode)):
        episode = generate_episode(env, policy, rng.integers(100000), es=True)
        agent.policy_iteration(episode)

        # for changing monitoring
        if not _ % 10000:
            for usable_ace in range(2):
                print("\n", q2pi(agent.Q)[usable_ace, 0])

    data = q2pi(agent.Q)

    # plot
    fig, axs = plt.subplots(2, 2, figsize=(8, 7))
    for usable_ace in range(2):
        for pi_v in range(2):
            axs[1 - usable_ace, pi_v].imshow(
                data[usable_ace, pi_v], origin="lower")

    for row in axs:
        for ax in row:
            set_ticks(ax)

    plt.savefig(out_path, dpi=dpi)
    return agent


if __name__ == '__main__':
    data = figure_5_1("plots/figure_5_1.png", seed=123)
    agent = figure_5_2("plots/figure_5_2.png", seed=123, n_episode=500000)

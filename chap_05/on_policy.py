# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:41:28 2024

@author: ccw
"""
import numpy as np
from tqdm import tqdm

import gymnasium as gym

from algorithm import AgentMC, generate_episode

import matplotlib.pyplot as plt


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


def on_policy_mc(
        out_path, eps=0.1, seed=None, agent=None, dpi=65, n_episode=3000000):

    def policy(s):
        if rng.random() < eps:
            return env.action_space.sample()
        return agent.Q[s].argmax()

    def q2pi(q: dict) -> np.ndarray:
        """return policy and optimal value in array"""
        ar = np.zeros((2, 2, 10, 10))  # usable_ace, pi | v, ...
        for (point, usable_ace, flip), av in q.items():
            ar[usable_ace, 0, point - 12, flip - 1] = av.argmax()  # policy
            ar[usable_ace, 1, point - 12, flip - 1] = av.max()  # optimal value
        return ar

    env = gym.make("blackjack", es=False)
    rng = np.random.default_rng(seed)
    agent = AgentMC(env)

    for _ in tqdm(range(n_episode)):
        episode = generate_episode(env, policy, rng.integers(100000), es=False)
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
    on_policy_mc("plots/on_policy_blackjack.png", n_episode=500000)

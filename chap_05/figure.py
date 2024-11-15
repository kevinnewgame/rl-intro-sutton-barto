# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:25:45 2024

@author: ccw
"""
import numpy as np
from tqdm import tqdm

import gymnasium as gym

from algorithm import FirstVisitMC, MC_ES

import matplotlib.pyplot as plt
# from matplotlib import colors


gym.envs.registration.register(
    id="blackjack",
    entry_point="rl_intro_env.gymnasium_env.envs:Blackjack",)


# def dict2array(d: dict, func: callable) -> np.ndarray:
#     array = np.empty((10, 10))
#     for (point, _, flip), v in d.items():
#         array[point - 12, flip - 1] = func(v)
#     return array


# def to_array_value(v: dict, func: callable) -> tuple[np.ndarray]:
#     # split into usable_ace and non-usable_ace
#     t = {s: value for s, value in v.items() if s[1]}
#     f = {s: value for s, value in v.items() if not s[1]}
#     # transform to array
#     u, uu = [dict2array(d, func) for d in (t, f)]  # usable, unusable
#     return u, uu


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
    alg = FirstVisitMC(env, seed=seed)
    vs = list()

    for _ in tqdm(range(10000)):
        alg.policy_evaluation(policy)

    v = v2ar(alg.V)
    vs.append(v)
    for usable_ace in range(2):
        axs[1 - usable_ace, 0].imshow(v[usable_ace], origin='lower')

    for _ in tqdm(range(490000)):
        alg.policy_evaluation(policy)

    v = v2ar(alg.V)
    vs.append(v)
    for usable_ace in range(2):
        axs[1 - usable_ace, 1].imshow(v[usable_ace], origin='lower')

    for row in axs:
        for ax in row:
            set_ticks(ax)

    plt.savefig(out_path, dpi=dpi)
    return vs


def figure_5_2(out_path, seed=None, alg=None, dpi=65, n_episode=3000000):

    def q2pi(q: dict) -> np.ndarray:
        """return policy and optimal value in array"""
        ar = np.zeros((2, 2, 10, 10))  # usable_ace, pi | v, ...
        for (point, usable_ace, flip), av in q.items():
            ar[usable_ace, 0, point - 12, flip - 1] = av.argmax()  # policy
            ar[usable_ace, 1, point - 12, flip - 1] = av.max()  # optimal value
        return ar

    env = gym.make("blackjack", es=True)
    alg = alg if alg else MC_ES(env, seed=seed)
    for _ in tqdm(range(n_episode)):
        alg.policy_iteration()

        # for changing monitoring
        if not _ % 10000:
            for usable_ace in range(2):
                print("\n", q2pi(alg.Q)[usable_ace, 0])

    data = q2pi(alg.Q)
    fig, axs = plt.subplots(2, 2, figsize=(8, 7))
    for usable_ace in range(2):
        for pi_v in range(2):
            axs[1 - usable_ace, pi_v].imshow(
                data[usable_ace, pi_v], origin="lower")

    for row in axs:
        for ax in row:
            set_ticks(ax)

    plt.savefig(out_path, dpi=dpi)
    return alg


if __name__ == '__main__':
    data = figure_5_1("plots/figure_5_1.png", seed=123)
    alg = figure_5_2("plots/figure_5_2.png", seed=123)
    # for _ in range(10):
    #     alg = figure_5_2(seed=123, alg=alg)

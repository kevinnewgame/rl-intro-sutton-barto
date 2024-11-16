# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:17:40 2024

@author: ccw
"""
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from multiprocessing import Pool
from functools import partial

import gymnasium as gym
from rl_intro_env.gymnasium_env.envs.blackjack import State

from algorithm import generate_episode, AgentMC

import matplotlib.pyplot as plt


gym.envs.registration.register(
    id="blackjack",
    entry_point="rl_intro_env.gymnasium_env.envs:Blackjack",)


def bj_state_prediction(state: State, seed=None, n_episode=100000000):

    def policy(s):
        agent_point = s[0]
        return 1 if agent_point < 20 else 0

    env = gym.make("blackjack", start=state)
    rng = np.random.default_rng(seed)
    agent = AgentMC(env)

    for _ in tqdm(range(n_episode)):
        episode = generate_episode(env, policy, rng.integers(100000))
        agent.policy_evaluation(episode)

        if not _ % 10000:
            print(agent.V[state])


class MC_Off:
    """Assumption: target policy is deterministic.
    Every visit"""

    def __init__(self, env: gym.Env, weighted=False, gamma=1, seed=None):
        self.env = env
        self.V = defaultdict(lambda: 0)
        self.N = defaultdict(lambda: 0)
        self.C = defaultdict(lambda: 0)
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.weighted = weighted

    def behavior_policy(self, s):
        return self.env.action_space.sample()

    def target_policy(self, s):
        return 1 if s[0] < 20 else 0

    def policy_evaluation(self):
        """the episode is generated from behavior policy"""
        episode = generate_episode(
            self.env, self.behavior_policy, self.rng.integers(100000))
        G = 0
        W = 1
        for t in reversed(range(len(episode) - 1)):  # T - 1, T - 2, ..., 0
            G = self.gamma * G + episode.r[t + 1]
            s, a = episode.s[t], episode.a[t]
            W = W * 2 if a == self.target_policy(s) else 0  # 1 / (1/2)
            if not self.weighted:
                self.N[s] += 1
                self.V[s] += (W * G - self.V[s]) / self.N[s]
            else:  # weighted update
                if W != 0:
                    self.C[s] += W
                    self.V[s] += W / self.C[s] * (G - self.V[s])
                else:
                    break
        return self.V


def experiment(env, seed=None):
    n_episode = 10000
    state = (13, 1, 2)
    algs = [MC_Off(env, weighted=w, seed=seed) for w in (True, False)]
    vs = np.empty((n_episode, len(algs)))
    for i in range(n_episode):
        vs[i, :] = [alg.policy_evaluation()[state] for alg in algs]
    return vs


def plot(out_path, seed=None, dpi=65, nr_parallel_processes=4):
    n_run = 100
    v = -0.27726
    env = gym.make("blackjack", start=(13, 1, 2))
    rng = np.random.default_rng(seed)
    seeds = rng.integers(100000, size=n_run)
    f = partial(experiment, env)
    with Pool(processes=nr_parallel_processes) as pool:
        results = pool.map(f, seeds)
    mse = np.power(np.stack(results) - v, 2).mean(axis=0)

    # plot
    fig, ax = plt.subplots()
    x = np.arange(1, mse.shape[0] + 1)
    ax.plot(x, mse, lw=1, alpha=0.9)
    ax.set_xscale('log')
    ax.set_ylim(-1e-1, 5)

    plt.savefig(out_path, dpi=dpi)
    return mse


if __name__ == '__main__':
    # bj_state_prediction((13, 1, 2))
    data = plot("plots/figure_5_3.png", seed=123)

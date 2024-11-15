# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:10:30 2024

@author: ccw
"""
import numpy as np
from copy import deepcopy
from collections import defaultdict
import gymnasium as gym


class Episode:

    def __init__(self):
        self.s = list()
        self.a = list()
        self.r = list()

    def __len__(self):
        return len(self.s)

    def __repr__(self):
        print("state, action, and reward")
        return "\n".join([str(x) for x in zip(self.s, self.a, self.r)])


def generate_episode(
        env: gym.Env, policy: callable, seed: int, es=False) -> dict[list]:
    # setup episode
    episode = Episode()
    episode.r.append(None)
    # setup environment
    s, _ = env.reset(seed=int(seed))
    a = env.action_space.sample() if es else policy(s)
    episode.s.append(s)
    episode.a.append(a)
    while True:
        s, r, terminated, _, _ = env.step(a)
        episode.s.append(s)
        episode.r.append(r)
        if terminated:
            episode.a.append(None)
            break
        a = policy(s)
        episode.a.append(a)
    return episode


class FirstVisitMC:

    def __init__(self, env: gym.Env, gamma=1, seed=None):
        self.env = env
        self.V = defaultdict(lambda: 0)
        self.N = defaultdict(lambda: 0)
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma

    def policy_evaluation(self, policy: callable):
        seed = self.rng.integers(100000)
        episode = generate_episode(self.env, policy, seed)
        G = 0
        for t in reversed(range(len(episode) - 1)):  # T - 1, T - 2, ..., 0
            G = self.gamma * G + episode.r[t + 1]
            s = episode.s[t]
            if s not in episode.s[:t]:
                self.N[s] += 1
                self.V[s] += (G - self.V[s]) / self.N[s]


class MC_ES():
    """Monte Carlo exploring start for policy iteration.
    Should use environment with exploring start setting"""

    def __init__(self, env: gym.Env, gamma=1, seed=None):
        self.env = env
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(env.action_space.n))
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma

    def policy_iteration(self):
        def policy(s):
            return self.Q[s].argmax()

        seed = self.rng.integers(100000)
        episode = generate_episode(self.env, policy, seed, es=True)
        G = 0
        for t in reversed(range(len(episode) - 1)):  # T - 1, T - 2, ..., 0
            G = self.gamma * G + episode.r[t + 1]
            (s, a) = (episode.s[t], episode.a[t])
            if (s, a) not in zip(episode.s[:t], episode.a[:t]):
                self.N[s][a] += 1
                self.Q[s][a] += (G - self.Q[s][a]) / self.N[s][a]


class OffPiMC


if __name__ == '__main__':
    def policy(s):
        agent_point = s[0]
        return 1 if agent_point < 20 else 0

    gym.envs.registration.register(
        id="blackjack",
        entry_point="rl_intro_env.gymnasium_env.envs:Blackjack",)

    env = gym.make("blackjack")

    rng = np.random.default_rng(123)
    episodes = list()
    for _ in range(10):
        episode = generate_episode(env, policy, seed=int(rng.integers(100000)))
        episodes.append(episode)

    alg = FirstVisitMC(env, seed=123)
    for _ in range(10):
        alg.policy_evaluation(policy)
        print(alg.V)

    env = gym.make("blackjack", es=True)
    alg = MC_ES(env, seed=123)
    for _ in range(10):
        alg.policy_iteration()
        print(alg.Q)

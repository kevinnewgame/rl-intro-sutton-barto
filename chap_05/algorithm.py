# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:10:30 2024

@author: ccw
"""
import numpy as np
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


class AgentMC:

    def __init__(self, env: gym.Env, gamma=1):
        self.env = env
        self.V = defaultdict(lambda: 0)
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N = None
        self.gamma = gamma

    def policy_evaluation(self, episode: Episode):
        if not self.N:
            self.N = defaultdict(lambda: 0)
        G = 0
        for t in reversed(range(len(episode) - 1)):  # T - 1, T - 2, ..., 0
            G = self.gamma * G + episode.r[t + 1]
            s = episode.s[t]
            if s not in episode.s[:t]:
                self.N[s] += 1
                self.V[s] += (G - self.V[s]) / self.N[s]

    def policy_iteration(self, episode: Episode):
        if not self.N:
            self.N = defaultdict(lambda: np.zeros(self.env.action_space.n))
        G = 0
        for t in reversed(range(len(episode) - 1)):
            G = self.gamma * G + episode.r[t + 1]
            (s, a) = (episode.s[t], episode.a[t])
            if (s, a) not in zip(episode.s[:t], episode.a[:t]):
                self.N[s][a] += 1
                self.Q[s][a] += (G - self.Q[s][a]) / self.N[s][a]


if __name__ == '__main__':
    ...

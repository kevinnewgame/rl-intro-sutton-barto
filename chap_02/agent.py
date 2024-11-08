# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:02:16 2024

@author: ccw
"""
import gymnasium as gym
import numpy as np


class AgentSimpleBandit:

    def __init__(self, env: gym.Env, seed=None, eps=0.1, alpha=None,
                 init_value=0, ucb=False, c=1):
        self.env = env
        self.Q = np.ones(env.action_space.n) * init_value
        self.N = np.zeros(env.action_space.n)  # number of action being visited
        self.rng = np.random.default_rng(seed)
        self.eps = eps
        self.alpha = alpha
        self.ucb = ucb
        self.c = c

    def _epsilon_greedy(self):
        if self.rng.random() < self.eps:
            return self.rng.integers(self.env.action_space.n)
        else:
            max_value = self.Q.max()
            actions = [a for a, v in enumerate(self.Q) if v == max_value]
            return actions[self.rng.choice(len(actions))]

    def _ucb(self, t):
        ucb = self.c * np.sqrt(np.log(t) / self.N)
        return (self.Q + ucb).argmax()

    def get_action(self, t=None):
        """t is the current time step for UCB"""
        a = self._ucb(t) if self.ucb else self._epsilon_greedy()
        self.N[a] += 1
        return a

    def update(self, a, r):
        if self.alpha:
            self.Q[a] += self.alpha * (r - self.Q[a])
        else:
            self.Q[a] += (r - self.Q[a]) / self.N[a]


if __name__ == '__main__':

    def run_steps(env, agent, n_step=1000):
        env.reset(seed=seed)
        for step in range(1, n_step + 1):
            a = agent.get_action(step)
            _, r, _, _, info = env.step(a)
            agent.update(a, r)

    gym.envs.registration.register(
        id="k-armed-testbed-v0",
        entry_point="rl_intro_env.gymnasium_env.envs:KArmedTestbed",
        )
    env = gym.make('k-armed-testbed-v0', k=10)

    seed = 12345

    agent = AgentSimpleBandit(env, seed=seed, eps=0.1)
    run_steps(env, agent)

    agent = AgentSimpleBandit(env, seed=seed, eps=0.1, alpha=0.1)
    run_steps(env, agent)

    agent = AgentSimpleBandit(env, seed=seed, ucb=True, c=2)
    run_steps(env, agent)

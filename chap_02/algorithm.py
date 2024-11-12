# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:37:43 2024

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


def simple_bandit_algorithm(
        env: gym.Env,
        n_steps=1000,
        alpha=None,
        init_value=0,
        eps=0.1,
        ucb=False, c=1,
        seed=12345,
        ):
    res = {"reward": list(), "opt_act": list()}

    rng = np.random.default_rng(seed)
    agent = AgentSimpleBandit(
        env, seed=rng.integers(100000), eps=eps, alpha=alpha,
        init_value=init_value, ucb=ucb, c=c)
    env.reset(seed=rng.integers(100000))
    for step in range(1, n_steps + 1):
        a = agent.get_action(step)
        _, r, _, _, info = env.step(a)
        agent.update(a, r)

        res["reward"].append(r)
        res["opt_act"].append(a == info["best_arm"])
    return res


class AgentGradientBandit:

    def __init__(self, env: gym.Env, seed=None, alpha_h: float | None = None,
                 baseline=True, stationary=True, alpha_r=0.1):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.alpha = alpha_h
        self.baseline = baseline
        self.stationary = stationary
        self.alpha_r = alpha_r

        self.R = 0  # baseline: mean of rewards
        self.N = 0  # number of action being visited
        self.H = np.zeros(env.action_space.n)

    def _soft_max_dist(self):
        exph = np.exp(self.H)
        return exph / exph.sum()

    def get_action(self):
        pi = self._soft_max_dist()
        return int(self.rng.choice(len(pi), p=pi))

    def update(self, a, r):
        self.N += 1
        if self.N == 1:
            self.R = r

        # preference update
        pi = self._soft_max_dist()
        pi[a] -= 1
        self.H += self.alpha * (r - (self.R if self.baseline else 0)) * -pi

        # baseline update
        if self.stationary:
            self.R += (r - self.R) / self.N
        else:
            self.R += self.alpha_r * (r - self.R)


def gradient_bandit_algorithm(
        env: gym.Env, n_steps=1000, alpha_h=0.1, baseline=True,
        stationary=True, alpha_r=0.1, seed=12345,):
    res = {"reward": list(), "opt_act": list()}

    rng = np.random.default_rng(seed)
    agent = AgentGradientBandit(
        env, seed=rng.integers(100000), alpha_h=alpha_h, baseline=baseline,
        stationary=stationary, alpha_r=alpha_r)
    _, info = env.reset(seed=rng.integers(100000))
    for step in range(1, n_steps + 1):
        a = agent.get_action()
        _, r, _, _, info = env.step(a)
        agent.update(a, r)

        res["reward"].append(r)
        res["opt_act"].append(a == info["best_arm"])
    return res


if __name__ == '__main__':
    seed = 1233

    gym.envs.registration.register(
        id="k-armed-testbed-v0",
        entry_point="rl_intro_env.gymnasium_env.envs:KArmedTestbed",
        )

    env = gym.make('k-armed-testbed-v0', k=10)
    res_1 = simple_bandit_algorithm(env, eps=0.1, seed=seed)
    res_2 = simple_bandit_algorithm(env, eps=0.1, alpha=0.1, seed=seed)
    res_3 = simple_bandit_algorithm(
        env, eps=0.1, alpha=0.1, ucb=True, c=2, seed=seed)

    env = gym.make('k-armed-testbed-v0', k=4, loc=4, stationary=True)
    res_4 = gradient_bandit_algorithm(
        env, seed=seed, alpha_h=0.4, stationary=True, baseline=True)

    # !!! This setting has problem and not solved yet
    env = gym.make('k-armed-testbed-v0', k=4, loc=4, stationary=False)
    res_5 = gradient_bandit_algorithm(
        env, seed=seed, alpha_h=0.4, stationary=False, baseline=True,
        alpha_r=0.1)

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:47:17 2024

@author: ccw
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from env.windyGridWorld import WindyGridWorld


class Memory:
    """Get the value in memory base on the time"""

    def __init__(self, length):
        self._ar = [np.nan for _ in range(length)]

    def _t2i(self, t):
        return t % (len(self._ar))

    def __setitem__(self, t, value):
        self._ar[self._t2i(t)] = value

    def __getitem__(self, t):
        """Get the value from array at time t"""
        return self._ar[self._t2i(t)]

    def __repr__(self):
        return self._ar.__repr__()


def memory(length):
    return Memory(length)


class NStepTreeBackup:

    def __init__(self, env: WindyGridWorld, n, step_size=0.5, eps=0.1,
                 discount_rate=1, seed=None):
        self._env = env
        self._rng = np.random.default_rng(seed)

        self._n = n
        self._alpha = step_size
        self._gamma = discount_rate
        self._eps = eps

        self._Q = self._make_action_value_function()

    def _make_action_value_function(self):
        Q = dict()
        for s in self._env.observation_space:
            Q[s] = dict()
            for a in self._env.action_space:
                Q[s][a] = 0
        return Q

    def _make_policy(self):
        pi = dict()
        for s, avs in self._Q.items():
            actions = list(avs.keys())
            values = list(avs.values())
            pi[s] = actions[np.array(values).argmax()]
        return pi

    def greedy_policy(self, s):
        avs = self._Q[s]
        actions = list(avs.keys())
        values = list(avs.values())
        return actions[np.array(values).argmax()]

    def epsilon_greedy(self, s):
        if self._rng.random() < self._eps:
            actions = list(self._Q[s].keys())
            return actions[self._rng.choice(len(actions))]
        return self.greedy_policy(s)

    def episode_train(self):

        def expected_value(s):
            avs = self._Q[s]
            values = np.array(list(avs.values()))
            idx_max = values.argmax()

            n_a = len(avs)
            p = np.ones(n_a) * eps / n_a
            p[idx_max] += 1 - eps
            return (p * values).sum()

        def expected_value_except(s):
            """Return
            1. the expected approximate value in the state, s, but not include
            the term of the greedy policy
            2. the probability of greedy policy"""
            avs = self._Q[s]
            values = np.array(list(avs.values()))
            idx_max = values.argmax()

            n_a = len(avs)
            p = np.ones(n_a) * eps / n_a
            p[idx_max] += 1 - eps
            greedy_prob = p[idx_max]
            p[idx_max] = 0
            return (p * values).sum(), greedy_prob

        n = self._n
        gamma = self._gamma
        alpha = self._alpha
        Q = self._Q
        eps = self._eps

        S, A, R = [memory(n + 1) for _ in range(3)]  # setup memory
        s, terminated = self._env.reset()  # episode start
        a = self.epsilon_greedy(s)
        t = 0
        S[t] = s
        A[t] = a
        T = np.inf
        while True:
            # save information
            if t < T:
                s, r, terminated = self._env.step(a)
                S[t + 1] = s
                R[t + 1] = r
                if terminated:
                    T = t + 1
                else:
                    a = self.epsilon_greedy(s)
                    A[t + 1] = a
            # value update
            tou = t - n + 1
            if tou >= 0:
                # calculate the return from tou, G
                if t + 1 >= T:
                    G = R[T]
                    end_time = T - 1
                else:
                    G = R[t + 1] + gamma * expected_value(S[t + 1])
                    end_time = t
                for k in reversed(range(tou + 1, end_time + 1)):
                    ev, p = expected_value_except(S[k])
                    G = R[k] + gamma * ev + gamma * p * G
                # action value, Q update
                n_step_td_error = G - Q[S[tou]][A[tou]]
                Q[S[tou]][A[tou]] += alpha * n_step_td_error
            if tou == T - 1:
                break
            t += 1

        return T

    @property
    def policy(self):
        return self._make_policy()


def plot_grid(ax, shape: (int, int)):

    # Setup grid world
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # background color
    rgb = np.ones(3)
    colors = [rgb]
    custom_cmap = ListedColormap(colors)

    # plot
    grid = np.zeros(shape)
    _ = ax.pcolormesh(
        grid, edgecolors="k", linewidth=0.5, cmap=custom_cmap)

    # adjust ticks
    # position
    nrow, ncol = shape
    ax.set_xticks(np.arange(ncol) + 0.5, minor=False)
    ax.set_yticks(np.arange(nrow) + 0.5, minor=False)
    # hide the tick line
    ax.tick_params(axis='both', which='both', length=0)

    # set label
    ax.set_xticklabels(np.arange(ncol))
    ax.set_yticklabels(np.arange(nrow))

    return ax


def example_6_5():

    def run_policy(policy, max_step=100):
        s, terminated = env.reset()
        traj = [s]
        step = 0
        while not terminated and step < max_step:
            a = policy[s]
            s, _, terminated = env.step(a)
            traj.append(s)
            step += 1
        return traj

    def plot_trajectory(ax, traj, color):
        coors = np.array(traj) + 0.5  # adjust coordinate
        x = coors[:, 1]
        y = coors[:, 0]
        ax.plot(x, y, c=color)

    def learning_plot(agent, n_episode, title):
        # learning curve
        steps = [agent.episode_train() for i in range(n_episode)]

        # trajectory
        traj = run_policy(agent.policy)

        # plots
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(title, fontsize=14)

        axs[0].plot(range(1, len(steps) + 1), steps)
        axs[0].set_ylim(0, 100)
        axs[0].axhline(y=15, color='r', linestyle='--', linewidth=1)

        plot_grid(axs[1], env.shape)
        plot_trajectory(axs[1], traj, "blue")
        axs[1].set_title("{} steps".format(str(len(traj) - 1)))

    env = WindyGridWorld()
    agent = NStepTreeBackup(env, n=4, step_size=.1, seed=12345)

    learning_plot(agent, 700, "Tree Backup")


if __name__ == '__main__':
    agent = example_6_5()

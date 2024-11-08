# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:23:24 2024

@author: ccw
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from kArmBandit import KArmBandit


class Exercise:

    def fig_2_2():
        # Simulation settings
        eps = (0, 0.01, 0.1)  # 3 types of bandit setting through epsilon
        bandits = [KArmBandit(k=10, eps=e) for e in eps]
        # simulating
        res = np.array([bandit.simulate(time=1000) for bandit in bandits])

        # Plot
        def plot(ax, data, eps, ylabel):
            for e, d in zip(eps, data):
                ax.plot(d, label=r'$\epsilon = {0:.02f}$'.format(e),
                        alpha=0.9, linewidth=1)
            ax.legend()
            ax.set_ylabel(ylabel)

        fig, (up, lo) = plt.subplots(
            2, 1, layout='constrained', sharex=True,
            )
        # 1. upper plot: average reward
        plot(up, res[:, 0, :], eps, ylabel='Average reward')

        # 2. lower plot: optimal action pickness %
        plot(lo, res[:, 1, :], eps, ylabel='% Optimal action')
        lo.set_xlabel('Steps')
        lo.set_ylim(0, 1)
        lo.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))


    def fig_2_3():
        # Simulation settings
        bandits = [
            KArmBandit(k=10, init_Q=5, eps=0, step_size=0.1),  # optimistic initial value
            KArmBandit(k=10, init_Q=0, eps=0.1, step_size=0.1),
            ]
        # simulating
        res = np.array(
            [bandit.simulate(time=1000, run=2000) for bandit in bandits]
            )
        # Plot
        def plot(ax, data, labels, ylabel):
            for l, d in zip(labels, data):
                ax.plot(d, label=l,
                        alpha=0.9, linewidth=1)
            ax.legend()
            ax.set_ylabel(ylabel)

        fig, ax = plt.subplots(layout='constrained')
        labels = (r'$Q_1=5, \epsilon=0$', r'$Q_1=0, \epsilon=0.1$')
        plot(ax, res[:, 1, :], labels, ylabel='% Optimal action')
        ax.set_xlabel('Steps')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))


    def exer_2_5():
        # Simulation settings
        bandits = [
            KArmBandit(k=10, eps=0.1, stationary=False, step_size=0.1),  # constant step size
            KArmBandit(k=10, eps=0.1, stationary=False)  # sample average
            ]
        # simulating
        res = np.array(
            [bandit.simulate(time=10000, run=2000) for bandit in bandits]
            )

        # Plot
        def plot(ax, data, ylabel):
            label_const = r'$\alpha=0.1$'
            for l, d in zip(('sample average', label_const), data):
                ax.plot(d, label=l, alpha=0.9, linewidth=1)
            ax.legend()
            ax.set_ylabel(ylabel)

        fig, (up, lo) = plt.subplots(
            2, 1, layout='constrained', sharex=True,
            )
        # 1. upper plot: average reward
        plot(up, res[:, 0, :], ylabel='Average reward')

        # 2. lower plot: optimal action pickness %
        plot(lo, res[:, 1, :], ylabel='% Optimal action')
        lo.set_xlabel('Steps')
        lo.set_ylim(0, 1)
        lo.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))


    def fig_2_4():
        # Simulation settings
        bandits = [
            KArmBandit(k=10, ucb=True, c=2),  # optimistic initial value
            KArmBandit(k=10, eps=0.1),
            ]
        # simulating
        res = np.array(
            [bandit.simulate(time=1000, run=2000) for bandit in bandits]
            )
        # Plot
        def plot(ax, data, labels, ylabel):
            for l, d in zip(labels, data):
                ax.plot(d, label=l,
                        alpha=0.9, linewidth=1)
            ax.legend()
            ax.set_ylabel(ylabel)

        fig, ax = plt.subplots(layout='constrained')
        labels = ('UCB c=2', r'$\epsilon$-greedy $\epsilon$=0.1',)
        plot(ax, res[:, 0, :], labels, ylabel='Average reward')
        ax.set_xlabel('Steps')


    def fig_2_5():
        # Simulation settings
        bandits = [
            KArmBandit(k=10, step_size=0.1, gradient=True, init_q=4),
            KArmBandit(k=10, step_size=0.4, gradient=True, init_q=4),
            ]
        # simulating
        res = np.array(
            [bandit.simulate(time=250, run=2000) for bandit in bandits]
            )
        # Plot
        def plot(ax, data, labels, ylabel):
            for l, d in zip(labels, data):
                ax.plot(d, label=l,
                        alpha=0.9, linewidth=1)
            ax.legend()
            ax.set_ylabel(ylabel)

        fig, ax = plt.subplots(layout='constrained')
        labels = (r'$\alpha$=0.1', r'$\alpha$=0.4')
        plot(ax, res[:, 1, :], labels, ylabel='% Optimal action')
        ax.set_xlabel('Steps')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))


if __name__ == '__main__':

    # Exercise.fig_2_2()
    # Exercise.fig_2_3()
    # Exercise.exer_2_5()
    # Exercise.fig_2_4()
    Exercise.fig_2_5()


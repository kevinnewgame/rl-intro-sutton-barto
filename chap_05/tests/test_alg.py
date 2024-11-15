# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:58:19 2024

@author: ccw
"""
import gymnasium as gym
import numpy as np
from chap_05.algorithm import generate_episode, AgentMC


gym.envs.registration.register(
    id="blackjack",
    entry_point="rl_intro_env.gymnasium_env.envs:Blackjack",)


def policy(s):
    agent_point = s[0]
    return 1 if agent_point < 20 else 0


class TestAlg:

    def test_gen_episode(self):
        env = gym.make("blackjack")
        rng = np.random.default_rng(123)
        episodes = list()
        for _ in range(10):
            episode = generate_episode(
                env, policy, seed=int(rng.integers(100000)))
            episodes.append(episode)

    def test_policy_evalution(self):
        env = gym.make("blackjack")
        rng = np.random.default_rng(123)
        agent = AgentMC(env)
        for _ in range(10):
            episode = generate_episode(
                env, policy, seed=int(rng.integers(100000)))
            agent.policy_evaluation(episode)
            print(agent.V)

    def test_policy_iteration(self):
        env = gym.make("blackjack", es=True)
        rng = np.random.default_rng(123)
        agent = AgentMC(env)

        def policy(s):
            return agent.Q[s].argmax()

        for _ in range(10):
            episode = generate_episode(
                env, policy, seed=int(rng.integers(100000)), es=True)
            agent.policy_iteration(episode)
            print(agent.Q)


if __name__ == '__main__':
    do = TestAlg()
    do.test_policy_evalution()
    do.test_policy_iteration()

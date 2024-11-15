# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:53:42 2024

@author: ccw
"""
import numpy as np
from tqdm import tqdm

import gymnasium as gym
from rl_intro_env.gymnasium_env.envs.blackjack import State

from algorithm import AgentMC, generate_episode


gym.envs.registration.register(
    id="blackjack",
    entry_point="rl_intro_env.gymnasium_env.envs:Blackjack",)





if __name__ == '__main__':

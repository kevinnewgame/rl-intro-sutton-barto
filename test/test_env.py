# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:20:31 2024

@author: ccw
"""
import sys
from gymnasium.envs.registration import register
import gymnasium as gym


print(sys.path)

register(
    id="GridWorld-v0",
    entry_point="rl_intro_env.gymnasium_env:GridWorldEnv",
)

gym.pprint_registry()
# gym.spec("GridWorld-v0")
env = gym.make("GridWorld-v0", shape=(6, 9), start=(3, 0), goal=(0, 8))

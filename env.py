import gym
import numpy as np


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    
    def _action(self, action):
        
        
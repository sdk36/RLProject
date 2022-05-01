from cpprb import ReplayBuffer, PrioritizedReplayBuffer
from DRL.env import env
import numpy as np


buffer_size = 1e+6
env_dict = {"obs":{"shape": env.observation_space.shape},
            "act":{"shape": 1,"dtype": np.ubyte},
            "rew": {},
            "next_obs": {"shape": env.observation_space.shape},
            "done": {}}


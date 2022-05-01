import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)

from cpprb import ReplayBuffer, PrioritizedReplayBuffer

buffer_size = 1e+6
env_dict = {"obs":{"shape": env.observation_space.shape},
            "act":{"shape": 1,"dtype": np.ubyte},
            "rew": {},
            "next_obs": {"shape": env.observation_space.shape},
            "done": {}}

prioritized = True

if prioritized:
    rb = PrioritizedReplayBuffer(buffer_size,env_dict,Nstep=Nstep)

    # Beta linear annealing
    beta = 0.4
    beta_step = (1 - beta)/N_iteration
else:
    rb = ReplayBuffer(buffer_size,env_dict,Nstep=Nstep)



class DDPG()
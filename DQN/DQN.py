#import gym library
import gym

#import math library
import math

#import numpy library
import numpy as np

#import matplotlib library
import matplotlib
import matplotlib.pyplot as plt

#import the double ended queue and named typle collections
from collections import namedtuple, deque

#import count froim itertools
from itertools import count

# from pillow import image for image management
from PIL import Image

#import pytorch library
import torch

#import pytorch neural network tools
import torch.nn as nn

#import pytorch optimiser tools
import torch.optim as optim

# import pytorch functional tools 
import torch.nn.functional as F

#import pytorch vision transformations
import torchvision.transforms as T

#import replay memory methods from rpm.py
from rpm import *


########################################

#create the environment
env = gym.make('CartPole-v0').unwrapped

#set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

#setup using gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from statistics import mode
import numpy as np
import gym
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from memory import ReplayMemory, Transition
from PIL import Image
from cpprb import ReplayBuffer, PrioritizedReplayBuffer

from model import Agent
from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(env):
    screen = env.render(mode='rgb_array')
    print(screen)
    screen = np.array(channels, screen_height, screen_width)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)

class DQN(object):
    def __init__(self, actions, env, env_dict, N_step, args):
        
        #hyper-parameters 
        self.batch_size = args.batch
        self.gamma = args.discount
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10
        self.actions = actions.shape[0]
        self.act_sp = actions
        self.steps_done = 0

        # setup agent models
        self.policy_net = Agent(args.hidden1, args.hidden2, args.init_w).to(device) 
        self.target_net = Agent(args.hidden1, args.hidden2, args.init_w).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        #optimizer
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        #replay memory
        self.memory = ReplayMemory(args.rmsize)

        #
        self.s_t = None # most resent state
        self.a_t = None # most recent action
        self.is_training = True

        #

    def update_policy(self): 
        
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)),
                                              device = device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action) 
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch) 

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        #computer loss 
        crierion = nn.SmoothL1Loss()
        loss = crierion(state_action_values, expected_state_action_values.unsqueeze(1))

        #optimize model 
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()


    def eval(self):
        self.policy_net.eval()
        self.target_net.eval()

    def cuda(self):
        self.policy_net.cuda()
        self.target_net.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.push(self.s_t,
                            self.a_t,
                            s_t1,
                            r_t)

    def random_action(self):
        action = self.act_sp.sample()
        self.a_t = action
        return action
    
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action = self.policy_net(to_tensor(np.array([state]))).max(1)[1].view(1,1)
                self.a_t = action 
                return action
        else: 
            action = self.act_sp.sample()
            self.a_t = action
            return action



    def reset(self, obs):
        self.s_t = obs

    def load_weights(self, output): 
        if output is None: return 

        self.policy_net.load_state_dict(
            torch.load('{}/policy_net.pkl'.format(output))
        )
    
    def save_model(self, output):
        torch.save(
            self.policy_net.state_dict(),
            '{}/policy_net.pkl'.format(output)
        )
    

    




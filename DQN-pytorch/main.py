import gym
import math
import random
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

from copy import deepcopy
from model import Agent
from dqn import DQN,get_screen
from evaluator import Evaluator
from util import *
from cpprb import ReplayBuffer, PrioritizedReplayBuffer
#
episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def train(num_iterations, agent, env, evaluate, validate_steps, output, max_episode_length=None, debug=False):
    
    agent.is_training = True
    step = episode = episode_steps = 0 
    episode_reward = 0.
    observation = None
    while step < num_iterations:
        # reset if start of episode
        if observation is None: 
            observation = deepcopy(env.reset())
            agent.reset(observation) 

        # agent pick action 
        if step <= args.warmup: 
            action = agent.random_action()
        else: 
            action = agent.select_action(observation) 

        #env response 
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1: 
            done = True
        
        #agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup: 
            agent.update_policy() 
        
        # evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0: 
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=True, visualize=True)
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
        
        # save intermediate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)

        # update
        step += 1 
        episode_steps += 1 
        episode_reward += reward
        observation = deepcopy(observation2) 

        if done: #end of episode 
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))
            episode_durations.append(t+1)

            agent.memory.on_episode_end()

            #reset 
            observation = None
            episode_steps = 0 
            episode_reward = 0.
            episode += 1


def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pytorch")

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden number of first fully connected layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connected layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--warmup', default=100, type=int, help='time without trainign but filling the replay memory')
    parser.add_argument('--discount', default=0.999, type=float, help='discount rate gamma')
    parser.add_argument('--rmsize', default=600000, type=int, help='memory size')
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--output', default='output', type=str, help='output folder location')
    parser.add_argument('--batch', default=64, type=int, help='minibatch size')
    parser.add_argument('--init_w', default=3e-3, type=float, help='initial model weights')
    parser.add_argument('--train_iter', default=200000, type=int, help='train iterations')
    parser.add_argument('--validation_episodes', default=20, type=int, help='how many steps to perform a validation exp')
    parser.add_argument('--nstep', default=3, type=int, help='set the nstep for the prioritized replay memory')
    parser.add_argument('--max_episode_length', default=2500, type=int, help='the maximum steps any episode can be')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')

    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    args = parser.parse_args()
    env = gym.make('CarRacing-v1')
    actions = env.action_space
    args.output = get_output_folder(args.output, 'CarRacing-v1')
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format('CarRacing-v1')
    Nstep = {"size": args.nstep, "rew": "rew", "next": "next_obs"}
    env_dict = {"obs":{"shape": env.observation_space.shape},
            "act":{"shape": 1,"dtype": np.ubyte},
            "rew": {},
            "next_obs": {"shape": env.observation_space.shape},
            "done": {}}
    agent = DQN(actions, env, env_dict, Nstep, args)
    evaluate = Evaluator(args.validation_episodes, 2000, args.output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate,args.validation_episodes, args.output, args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validation_episodes, agent, env, evaluate, args.resume, visualize=True, debug=args.debug)
    
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))


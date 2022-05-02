# README


https://github.com/ghliu/pytorch-ddpg - DDPG example for gym environments

https://arxiv.org/abs/1509.02971 - 
DDPG paper


https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py - 
DQN cartpole tutorial example

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
DQN Cartpole website

https://ymd_h.gitlab.io/cpprb/examples/dqn_per/
DQN with PrioritizedReplayBuffer cpprb tutorial

---

## Execution 

The main script has been equipt with optional parameters to help with testing and execution

    py main.py --parmeter option

Options

    --mode Options: train/test Default: train
    determines whether to run training or test

    --hidden1 Options: Int Default: 400
    determines the FCN first layer

    --hidden2 Options: Int Default: 300
    deteremines the FCN second layer

    --rate Options: Float Default: 0.001
    sets the learning rate

    --warmup Options: Int Default: 100 
    sets the amount of time without training to fill the replay memory

    --discount Options: Float Default: 0.999
    sets the discount rate gamma

    --rmsize Options: Int Default: 600000
    sets the memory size

    --visualize actions: store_true
    sets visualise to true if included

    --debug actions: store_true
    sets debug to true if included

    --output Options: String Default: 'output'
    sets the output folder that we want to use

    --batch Options: Int Default: 64 
    sets the batch size for replay memory

    --init_w Options: Float Default: 3e-3 
    sets the init weights of the models

    --train_iter Options: Int Default: 200000
    sets the number of training cycles 

    --validation_episodes Options: Int Default: 20 
    sets how many steps to perform a validation exp



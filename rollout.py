import gym
import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from utils.make_soccer_env import make_soccer_env_rollout as make_env
from algorithms.maddpg_soccer import MADDPG

def run(config):
    model_dir = Path('./finish') / config.model_name

    env = make_env(config.num_agents)

    maddpg = MADDPG.init_from_save(model_dir)

    t = 0
    for ep_i in range(0, config.n_episodes):
        print('Episodes %i-%i of %i' % (ep_i + 1,
                                        ep_i + 2,
                                        config.n_episodes))

        obs = env.reset()
        #print('main_obs.shape: ', obs.shape)
        maddpg.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            torch_obs = [Variable(torch.Tensor(obs[i]).flatten().reshape(1, -1), requires_grad=False)
                            for i in range(maddpg.num_agents)]
            print(ep_i, et_i) 
            #print('main_torch_obs[0].shape: ', torch_obs[0].shape)

            torch_agent_actions = maddpg.step(torch_obs, explore=False)

            actions = [np.argmax(ac.data.numpy(), axis=1)[0] for ac in torch_agent_actions]
            

            #print('torch_agent_actions: ', torch_agent_actions)
            #print('main_agent_actions: ', agent_actions)
            print('main_actions: ', actions)

            next_obs, rewards, dones, infos = env.step(actions)

            #print(rewards)


            if dones:
                break
            obs = next_obs
            t += 1

    env.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--num_agents', type=int, default=3)
    parser.add_argument('--episode_length', default=1000, type=int)

    config = parser.parse_args()
    run(config)

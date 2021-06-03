import gym
import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
#from utils.make_env import make_env
from utils.make_soccer_env import make_soccer_env as make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
#from algorithms.maddpg import MADDPG
from algorithms.maddpg_soccer import MADDPG

USE_CUDA = False  # torch.cuda.is_available()

#def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
#    def get_env_fn(rank):
#        def init_env():
#            env = make_env(env_id, discrete_action=discrete_action)
#            env.seed(seed + rank * 1000)
#            np.random.seed(seed + rank * 1000)
#            return env
#        return init_env
#    if n_rollout_threads == 1:
#        return DummyVecEnv([get_env_fn(0)])
#    else:
#        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def make_parallel_env(n_rollout_threads, num_agents):
    def get_env_fn(rank):
        def init_env():
            env = make_env(num_agents)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def one_hot_encoding(target, nb_classes):
    ret = np.zeros(nb_classes)
    ret[target] = 1
    return ret

def run(config):

    if torch.cuda.is_available():
        print('cuda available')
        USE_CUDA=True

    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    
    # make parallel env
    #env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
    #                        config.discrete_action)
    
    # make parallel env
    env = make_parallel_env(config.n_rollout_threads, config.num_agents)


    maddpg = MADDPG.init_from_env(env, num_agents=config.num_agents,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    

    action_spaces = [ gym.spaces.Discrete(env.action_space.nvec[idx]) for idx in range(config.num_agents) ]
    observation_spaces = [ gym.spaces.Box(low=env.observation_space.low[idx],
                                          high=env.observation_space.high[idx],
                                          dtype=env.observation_space.dtype) for idx in range(config.num_agents) ]


    replay_buffer = ReplayBuffer(config.buffer_length, config.num_agents,
                                 [ np.product(obsp.shape) for obsp in observation_spaces],
                                 [ np.product(acsp.shape) if isinstance(acsp, Box) else acsp.n
                                  for acsp in action_spaces])
    
                                
    while True:

        if replay_buffer.filled_i == config.buffer_length:
            break

        obs = env.reset()
        maddpg.prep_rollouts(device='cpu')
        
        for et_i in range(config.episode_length):
            act = env.action_space_sample()
            actions = [ ac.tolist() for ac in act ]

            act = [np.array([one_hot_encoding(ac[i], 19) for ac in act]) for i in range(config.num_agents)]
            
            #print(act)
            #print(len(act))
            #print(act[0].shape)

            n_obs, rews, dones, info = env.step(actions)

            obs = np.concatenate([obs[:, i].flatten().reshape(config.n_rollout_threads, 1, -1) for i in range(config.num_agents)], axis=1)
            n_obs = np.concatenate([n_obs[:, i].flatten().reshape(config.n_rollout_threads, 1, -1) for i in range(config.num_agents)], axis=1)
            dones = [[dones[0] for _ in range(config.num_agents)] for _2 in range(config.n_rollout_threads)]
            dones = np.array(dones)

            replay_buffer.push(obs, act, rews, n_obs, dones)


    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()

        ###
        #print('main_obs.type: ', type(obs))
        #print('main_obs.shape: ', obs.shape)
        #print('main_obs[:, 0].shape: ', obs[:, 0].shape)

        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')


        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()




        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[:, i]).flatten().reshape(config.n_rollout_threads, -1),
                                  requires_grad=False)
                         for i in range(maddpg.num_agents)]

            

            ###
            #print('main_torch_obs.type: ', type(torch_obs))
            #print('main_torch_obs.len: ', len(torch_obs))
            #print('main_torch_obs.shape[0]: ', torch_obs[0].shape)

            # get actions as torch Variables
            # torch_agent_actions = (num_agents, n_rollout_threads)
            torch_agent_actions = maddpg.step(torch_obs, explore=True)


            ###
            #print('main_torch_agent_actions: ', torch_agent_actions)

            # convert actions to numpy arrays
            agent_actions = [np.argmax(ac.data.numpy(), axis=1) for ac in torch_agent_actions]

            ###
            #print('main_agent_actions: ', agent_actions)

            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]

            


            # return to one-hot-encoding for replay buffer
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]



            ###
            #print(agent_actions)
            #print(len(agent_actions))
            #print(agent_actions[0].shape)
            #print('main_actions: ', actions)
            #print('main_actions: ', actions[0])

            ###
            #print(env)
            #print(actions)
            #print(len(actions))
            #print(type(actions[0]))
            
            next_obs, rewards, dones, infos = env.step(actions)



            obs = np.concatenate([ obs[:, i].flatten().reshape(config.n_rollout_threads, 1, -1) for i in range(config.num_agents) ], axis=1)
            next_obs = np.concatenate([ next_obs[:, i].flatten().reshape(config.n_rollout_threads, 1, -1) for i in range(config.num_agents) ], axis=1)

            dones = [[dones[0] for _ in range(config.num_agents)] for _2 in range(config.n_rollout_threads)]
            dones = np.array(dones)


            ###
            #print('main_type_nobs[0], obs[0], rewards, dones', type(next_obs[0]), type(obs[0]), type(rewards), type(dones))
            #print('main_env_step_next_obs.shape: ', next_obs.shape)
            #print('~rewards.shape: ', rewards.shape)
            #print('~dones: ', dones)
            #print('~infos: ', infos)
            #print('~obs.shape: ', obs.shape)
            #print('~agent_actions: ', agent_actions)
            
         
            
            

            # (n_rollout_threads, num_agents, ~) except actions
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    ###
                    maddpg.prep_training(device='gpu')
                else:
                    ###
                    maddpg.prep_training(device='cpu')
                
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.num_agents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        
        
        
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=2, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1.5e5), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=125, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=8192, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true', default=True)

    # plus
    parser.add_argument('--num_agents', type=int, default=3)


    config = parser.parse_args()


    run(config)

import numpy as np
import gym
import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent

from torch import Tensor
from torch.autograd import Variable

MSELoss = torch.nn.MSELoss()


class MADDPG(object):

    def __init__(self, agent_init_params, num_agents,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=True):
        
        self.num_agents = num_agents
        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'
        self.critic_dev = 'cpu'
        self.trgt_pol_dev = 'cpu'
        self.trgt_critic_dev = 'cpu'
        self.niter = 0


    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        # scale noise for each agent

        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()


    def step(self, observations, explore=False):
        # take a step forward in env with all agents
        # return list of actions for each agent
        
        ###
        #print('maddpg_soccer_step@@@@@@@@@@@@@@@@@@@@@')


        return [a.step(obs, explore=explore) for a, obs in zip(self.agents, observations)]

    def update(self, sample, agent_i, parallel=False, logger=None):

        ###
        #print('update_agent_i: ', agent_i)
        
        obs, acs, rews, next_obs, dones = sample
        
        ###
        #print('update_len: ', len(obs), len(next_obs), len(acs), len(rews), len(dones))
        #print('update_nobs.len: ', len(next_obs))
        #print('update_obs[0].shape: ', obs[0].shape)
        #print('update_nobs[0].shape: ', next_obs[0].shape)
        #print('update_acs[0].shape: ', acs[0].shape)
        #print('update_rews[0].shape: ', rews[0].shape)
        #print('update_dones[0].shape: ', dones[0].shape)

        
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        if self.discrete_action: # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(self.target_policies, next_obs)]
        else:
            all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]


        ###
        #print('update_all_trgt_acs.len: ', len(all_trgt_acs))
        #print('update_all_trgt_acs[0].shape: ', all_trgt_acs[0].shape)

            
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=-1)

        ###
        #print('update_trgt_vf_in.len: ', len(trgt_vf_in))
        #print('update_trgt_vf_in[0].shape: ', trgt_vf_in[0].shape)

        # y' = r + gamma*Q'()
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))
    
        #print('update_acs[0]: ', acs[0])
        #print('update_acs[0]: ', acs[0].to(torch.int64))
        #print('update_one_hot_act.shape: ', torch.nn.functional.one_hot(acs[0], num_classes=all_trgt_acs[0].shape[1]).shape)
        

        # value_function input
        vf_in = torch.cat((*obs, *acs), dim=1)

        actual_value = curr_agent.critic(vf_in)

        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()

        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        
        curr_agent.policy_optimizer.zero_grad()
        if self.discrete_action:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True, flag=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        

        all_pol_acs = []

        ###
        #print('update_num_agents; ', self.num_agents)
        #print('update_num_agents.type: ', type(self.num_agents))
        #print('update_self.policies.type: ', type(self.policies))
        #print('update_obs.type: ', type(obs))

        for i, pi, ob in zip(range(self.num_agents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
            else:
                all_pol_acs.append(pi(ob))

        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()

        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                                {'vf_loss': vf_loss,
                                 'pol_loss': pol_loss},
                                self.niter)

    

    def update_all_targets(self):

        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1


    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()

        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device


    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device


    def save(self, filename):
        # move parameters to CPU before saving
        self.prep_training(device='cpu')
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)



    @classmethod
    def init_from_env(cls, env, num_agents, 
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):

        agent_init_params = []
       
        ###
        #print('policy_env.action_space: ', env.action_space)
        #print('policy_env.observation_space: ', env.observation_space)
        
        action_spaces = [ gym.spaces.Discrete(env.action_space.nvec[idx]) for idx in range(num_agents) ]
        observation_spaces = [ gym.spaces.Box(low=env.observation_space.low[idx],
                                              high=env.observation_space.high[idx],
                                              dtype=env.observation_space.dtype) for idx in range(num_agents) ]

        #print(action_spaces)
        #print(observation_spaces)



        for acsp, obsp in zip(action_spaces, observation_spaces):

            ###
            #print('policy_num_in_pol: ', np.product(obsp.shape))
            num_in_pol = np.product(obsp.shape)
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: np.product(x.shape)
            else: # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)

            num_in_critic = 0
            for oobsp in observation_spaces:
                num_in_critic += np.product(oobsp.shape)
            for oacsp in action_spaces:
                num_in_critic += get_shape(oacsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        ###
        #print('policy_num_in_pol: ', agent_init_params[0]['num_in_pol'])
        #print('~num_out_pol: ', agent_init_params[0]['num_out_pol'])
        #print('~num_in_critic: ', agent_init_params[0]['num_in_critic'])


        
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'num_agents': num_agents,
                     'hidden_dim': hidden_dim,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action}

        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance


    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance
        

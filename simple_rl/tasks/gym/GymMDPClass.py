'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import random
import sys
import os
import random
from collections import defaultdict

# Other imports.
import gymnasium as gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.gym.GymStateClass import GymState

class GymMDP(MDP):
    ''' Class for Gym MDPs '''

    def __init__(self, env_name='CartPole-v0', render=False, render_every_n_episodes=0, wrapper=None, seed=None, **kwargs):
        '''
        Args:
            env_name (str)
            render (bool): If True, renders the screen every time step.
            render_every_n_epsiodes (int): @render must be True, then renders the screen every n episodes.
        '''
        # self.render_every_n_steps = render_every_n_steps
        self.render_every_n_episodes = render_every_n_episodes
        self.episode = 0
        self.env_name = env_name
        self.env = gym.make(env_name, **kwargs)
        if wrapper:
            self.env = wrapper(self.env)
        self.render = render
        if seed:
            init_state = GymState(self.env.reset(seed=seed)[0])
            self.env_seed = seed
        else:
            init_state = GymState(self.env.reset()[0])
            self.env_seed = None
        MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func, init_state=init_state)
    
    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["env_name"] = self.env_name
   
        return param_dict

    def _reward_func(self, state, action, next_state):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        return self.prev_reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        obs, reward, is_terminal, truncated, info = self.env.step(action)

        if self.render and (self.render_every_n_episodes == 0 or self.episode % self.render_every_n_episodes == 0):
            self.env.render()

        self.prev_reward = reward
        self.next_state = GymState(obs, is_terminal=is_terminal)

        return self.next_state

    def reset(self, seed=None):
        if seed:
            if self.env_seed:
                print("Got a reset seed but the env already has a seed. Going with reset seed.")
            self.init_state = GymState(self.env.reset(seed=seed)[0])
        elif self.env_seed:
            self.init_state = GymState(self.env.reset(seed=self.env_seed)[0])
        else:
            self.init_state = GymState(self.env.reset()[0])
        # print(obs)
        self.episode += 1
        return self.init_state

    def __str__(self):
        return "gym-" + str(self.env_name)

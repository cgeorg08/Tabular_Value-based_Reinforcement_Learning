#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import itertools
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class MonteCarloAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        self.Q_sa_means = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            # TO DO: Add own code
            if np.random.uniform() < epsilon:
                a = np.random.randint(0,self.n_actions)
            else:
                a = argmax(self.Q_sa_means[s])
            
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
                
            # TO DO: Add own code
            a = np.random.choice([0,1,2,3],p=softmax(self.Q_sa_means[s],temp))
            
        return a
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        pass

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    total_rewards = []

    # TO DO: Write your Monte Carlo RL algorithm here!
    outersteps = 0
    while outersteps<n_timesteps:
        done = False
        state = env.reset()

        states_list = list()
        states_list.append(state)

        isterminal_list = list()
        isterminal_list.append(False)

        actions_list = list()
        rewards = list()

        # phase 1: collect episode
        for steps in range(min(max_episode_length,n_timesteps-outersteps)):
            action = pi.select_action(state,policy,epsilon,temp)    # select action         
            s_next,r,done = env.step(action)    # simulate environment

            rewards.append(r)
            state = s_next  # update state

            # keep backups for update
            states_list.append(s_next)
            isterminal_list.append(done)
            actions_list.append(action)

            if done:
                break
        
        # print('t step to find the solution = ',steps)
        # print('len of states list = ',len(states_list))
        # print('len of isterminal list = ',len(isterminal_list))
        # print('len of actions list = ',len(actions_list))
        # print('len of rewards list = ',len(rewards))

        outersteps = outersteps + steps+1

        # phase 2: update
        backup_estimate_next = 0
        for i in range(steps,-1,-1):
            backup_estimate = rewards[i] + gamma*backup_estimate_next
            current_Q = pi.Q_sa_means[states_list[i]][actions_list[i]]
            pi.Q_sa_means[states_list[i]][actions_list[i]] = current_Q + learning_rate*(backup_estimate-current_Q)
            backup_estimate_next = backup_estimate

        total_rewards.append(rewards)
    
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

    return list(itertools.chain(*total_rewards)) 
    
def test():
    n_timesteps = 10000
    max_episode_length = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    # print("Obtained rewards: {}".format(rewards))
    print('number of times found the goal state = ',rewards.count(40))   
            
if __name__ == '__main__':
    test()

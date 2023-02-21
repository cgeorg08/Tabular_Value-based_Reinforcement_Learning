#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
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
            a = np.argmax(softmax(self.Q_sa_means[s],temp))
            
        return a
        
    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        pass

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
    rewards = []

    # TO DO: Write your n-step Q-learning algorithm here!
    states_list = list()
    actions_list = list()

    done = False
    state = env._location_to_state(env.start_location)
    states_list.append(state)

    for steps in range(max_episode_length):
        action = pi.select_action(state,policy,epsilon,temp)    # select action         
        s_next,r,done = env.step(action)    # simulate environment

        rewards.append(r)
        state = s_next  # update state

        # keep backups for update
        states_list.append(s_next)
        actions_list.append(action)

        if done:
            break
    
    print('total steps to find the solution = ',steps)
    print('len of states list = ',len(states_list))
    print('len of actions list = ',len(actions_list))

    T_ep = steps+1

    for t in range(T_ep):
        backup_estimate = 0
        m = min(n, T_ep-t)
        if rewards[t+m] != -1:          # if state t+m is terminal
            for i in range(m):          # n-step target without bootstrap
                backup_estimate = backup_estimate + (gamma**i)*rewards[t+i]          
        else:
            for i in range(m):          # n-step target
                backup_estimate = backup_estimate + (gamma**i)*rewards[t+i] + (gamma**m)*np.max(pi.Q_sa_means,axis=1)[t+m]
        current_Q = pi.Q_sa_means[states_list[t]][actions_list[t]]
        pi.Q_sa_means[states_list[t]][actions_list[t]] = current_Q + learning_rate*(backup_estimate-current_Q) # update Q-table

        
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution

    return rewards 

def test():
    n_timesteps = 10000
    max_episode_length = 10000
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    print("Obtained rewards: {}".format(rewards))    
    
if __name__ == '__main__':
    test()

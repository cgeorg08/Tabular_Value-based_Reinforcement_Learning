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

class QLearningAgent:

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
        
    def update(self,s,a,r,s_next,done):
        # TO DO: Add own code
        backup_estimate = r + self.gamma * np.max(self.Q_sa_means,axis=1)[s_next]
        new_Q_sa_mean = self.Q_sa_means[s][a] + self.learning_rate * (backup_estimate - self.Q_sa_means[s][a])
        return new_Q_sa_mean

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    # TO DO: Write your Q-learning algorithm here!
    done = False
    steps = 0
    state = env.reset()
    while steps<n_timesteps:
        steps+=1
        action = pi.select_action(state,policy,epsilon,temp)    # select action         
        s_next,r,done = env.step(action)    # simulate environment
        pi.Q_sa_means[state][action] = pi.update(state,action,r,s_next,done)    # Q update

        rewards.append(r)

        if done == True:
            state = env.reset()  # reset environment
            done = False
        else:
            state = s_next  # update state
        
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

    return rewards 

def q_learning_annealing(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon_list=None, temp=None, plot=True):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    # TO DO: Write your Q-learning algorithm here!
    done = False
    steps = 0
    state = env.reset()
    while steps<n_timesteps:
        steps+=1
        action = pi.select_action(state,policy,epsilon_list[steps-1],temp)    # select action         
        s_next,r,done = env.step(action)    # simulate environment
        pi.Q_sa_means[state][action] = pi.update(state,action,r,s_next,done)    # Q update

        rewards.append(r)

        if done == True:
            state = env.reset()  # reset environment
            done = False
        else:
            state = s_next  # update state
        
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

    return rewards 

def test():
    
    n_timesteps = 5000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'softmax' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))
    print('number of times found the goal state = ',rewards.count(40))

if __name__ == '__main__':
    test()

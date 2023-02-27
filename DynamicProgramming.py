#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        self.Q_sa_means = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: Add own code
        a = np.argmax(self.Q_sa,axis=1)[s]
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''

        # TO DO: Add own code
        sum = 0
        for next_s in range(p_sas.shape[2]):
            sum = sum + (p_sas[s][a][next_s] * (r_sas[s][a][next_s] + self.gamma * np.max(self.Q_sa,axis=1)[next_s]))
        return sum
    
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)    
        
    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    i=0     # loops
    while True:
        i+=1
        max_error = 0 
        for state in range(QIagent.n_states):
            for action in range(QIagent.n_actions):
                x = QIagent.Q_sa[state][action]     # store current estimate
                QIagent.Q_sa_means[state][action] = QIagent.update(state,action,env.p_sas,env.r_sas)    # Q-iteration update rule
                max_error = max(max_error,abs(x-QIagent.Q_sa_means[state][action]))     # update max error
                QIagent.Q_sa[state][action] = QIagent.Q_sa_means[state][action]
        print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=1)

        if max_error < threshold:
            break
    
    QIagent.Q_sa = QIagent.Q_sa_means.copy()
        
    # Plot current Q-value estimates & print max error
    # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.8)
    # print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
     
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)

    print('I found the optimal policy')
    
    # View optimal policy
    done = False
    s = env.reset()
    steps = 0
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        steps += 1
        print('step: {} , reward: {}'.format(steps,r))
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=2)
        s = s_next

    # TO DO: Compute mean reward per timestep under the optimal policy
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))

if __name__ == '__main__':
    experiment()

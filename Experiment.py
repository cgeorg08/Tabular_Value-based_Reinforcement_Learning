#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

from tqdm import tqdm
import numpy as np
import time
from itertools import zip_longest
from sklearn.preprocessing import MinMaxScaler


from Q_learning import q_learning, q_learning_annealing
from SARSA import sarsa
from MonteCarlo import monte_carlo
from Nstep import n_step_Q
from Helper import LearningCurvePlot, smooth, linear_anneal, exponential_anneal

def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy', 
                    epsilon=None, temp=None, smoothing_window=51, plot=False, n=5):

    reward_results = np.empty([n_repetitions,n_timesteps]) # Result array
    now = time.time()
    
    for rep in tqdm(range(n_repetitions)): # Loop over repetitions
        if backup == 'q':
            rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
        elif backup == 'sarsa':
            rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
        elif backup == 'mc':
            rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
        elif backup == 'nstep':
            rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
        elif backup == 'qa':
            rewards = q_learning_annealing(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)

        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve  

def prepare_exp_anneal(scaler,values_list):
    scaled_num_list = scaler.fit_transform([[num] for num in values_list])
    final_list = [num[0] for num in scaled_num_list]
    final_list = sorted(final_list, reverse=True)
    return final_list


def experiment(option):
    ####### Settings
    # Experiment    
    n_repetitions = 50
    smoothing_window = 1001
    plot = False # Plotting is very slow, switch it off when we run repetitions
    
    # MDP    
    n_timesteps = 50000
    max_episode_length = 150
    gamma = 1.0

    # Parameters we will vary in the experiments, set them to some initial values: 
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.05
    temp = 1.0
    # Back-up & update
    backup = 'q' # 'q' or 'sarsa' or 'mc' or 'nstep'
    learning_rate = 0.25
    n = 5 # only used when backup = 'nstep'
        
    # Nice labels for plotting
    backup_labels = {'q': 'Q-learning',
                  'sarsa': 'SARSA',
                  'mc': 'Monte Carlo',
                  'nstep': 'n-step Q-learning'}
    
    ####### Experiments
    
    #### Assignment 1: Dynamic Programming
    # Execute this assignment in DynamicProgramming.py
    optimal_average_reward_per_timestep = 1.3 # set the optimal average reward per timestep you found in the DP assignment here
    
    #### Assignment 2: Effect of exploration
    if option == 2:
        policy = 'egreedy'
        epsilons = [0.02,0.1,0.3]
        learning_rate = 0.25
        backup = 'q'
        Plot = LearningCurvePlot(title = 'Exploration: $\epsilon$-greedy versus softmax exploration')    
        for epsilon in epsilons:        
            learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                gamma, policy, epsilon, temp, smoothing_window, plot, n)
            Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))    
        policy = 'softmax'
        temps = [0.01,0.1,1.0]
        for temp in temps:
            learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                gamma, policy, epsilon, temp, smoothing_window, plot, n)
            Plot.add_curve(learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))
        Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
        Plot.save('exploration.png')
    
    ###### Assignment 3: Q-learning versus SARSA
    if option == 3:
        policy = 'egreedy'
        epsilon = 0.1 # set epsilon back to original value 
        learning_rates = [0.02,0.1,0.4]
        backups = ['q','sarsa']
        Plot = LearningCurvePlot(title = 'Back-up: on-policy versus off-policy')    
        for backup in backups:
            for learning_rate in learning_rates:
                learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                    gamma, policy, epsilon, temp, smoothing_window, plot, n)
                Plot.add_curve(learning_curve,label=r'{}, $\alpha$ = {} '.format(backup_labels[backup],learning_rate))
        Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
        Plot.save('on_off_policy.png')
    
    ####### Assignment 4: Back-up depth
    if option == 4:
        policy = 'egreedy'
        epsilon = 0.1 # set epsilon back to original value
        learning_rate = 0.25
        backup = 'nstep'
        ns = [1,3,10,30]
        Plot = LearningCurvePlot(title = 'Back-up: depth')    
        for n in ns:
            learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                gamma, policy, epsilon, temp, smoothing_window, plot, n)
            Plot.add_curve(learning_curve,label=r'{}-step Q-learning'.format(n))
        backup = 'mc'
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                            gamma, policy, epsilon, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label='Monte Carlo')        
        Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
        Plot.save('depth.png')


    # new section
    if option == 5:
        print('Enabling option 5 - effect of annealing . . .')
        policy = 'egreedy'
        learning_rate = 0.4
        backup = 'qa'

        epsilons1 = []
        epsilons2 = []
        epsilons3 = []
        for i in range(n_timesteps):
            epsilons1.append(linear_anneal(i,n_timesteps,1,0.01,0.8))
            epsilons2.append(linear_anneal(i,n_timesteps,1,0.01,0.5))
            epsilons3.append(linear_anneal(i,n_timesteps,1,0.01,0.2))
        

        Plot = LearningCurvePlot(title = 'Exploration: annealing egreedys')  
        i=1  
        for epsilon_list in [epsilons1,epsilons2,epsilons3]:        
            learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                gamma, policy, epsilon_list, temp, smoothing_window, plot, n)
            Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(i))    
            i+=1
        Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
        plot_name = 'anneal'+'_l='+str(learning_rate)+'.png'
        Plot.save(plot_name)

    if option == 6:
        print('Enabling option 6 - effect of exponential annealing . . .')
        policy = 'egreedy'
        learning_rate = 0.25
        backup = 'qa'

        scaler = MinMaxScaler(feature_range=(0, 1))
        epsilons1 = []
        epsilons2 = []
        epsilons3 = []
        for i in range(n_timesteps):
            epsilons1.append(exponential_anneal(i,0.8))
            epsilons2.append(exponential_anneal(i,0.5))
            epsilons3.append(exponential_anneal(i,0.2))

        epsilons1_new = prepare_exp_anneal(scaler,epsilons1)
        epsilons2_new = prepare_exp_anneal(scaler,epsilons2)
        epsilons3_new = prepare_exp_anneal(scaler,epsilons3)    

        Plot = LearningCurvePlot(title = 'Exploration: annealing egreedys')  
        i=1  
        for epsilon_list in [epsilons1_new,epsilons2_new,epsilons3_new]:        
            learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                gamma, policy, epsilon_list, temp, smoothing_window, plot, n)
            Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(i))    
            i+=1
        Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
        plot_name = 'anneal_exp'+'_l='+str(learning_rate)+'.png'
        Plot.save(plot_name)

    if option == 7:
        print('Enabling option 7 - final graph . . .')
        policy = 'egreedy'
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        epsilons1 = []
        epsilons2 = []

        for i in range(n_timesteps):
            epsilons1.append(exponential_anneal(i,0.5))
            epsilons2.append(linear_anneal(i,n_timesteps,1,0.01,0.8))

        epsilons_exp = prepare_exp_anneal(scaler,epsilons1)

        # title
        Plot = LearningCurvePlot(title = 'Q-learning')  

        # linear annealing
        backup = 'qa'
        learning_rate = 0.4
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                            gamma, policy, epsilons2, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'lin-annealing, factor = {}'.format(0.8)) 

        # exponential annealing
        backup = 'qa'
        learning_rate = 0.25
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                            gamma, policy, epsilons_exp, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'exp-annealing, factor = {}'.format(0.5))    

        # best params
        backup = 'q'
        learning_rate = 0.4
        epsilon = 0.01
        gamma = 0.5
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                            gamma, policy, epsilon, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'fine-tuning')   

        # optimal reward
        Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")

        plot_name = 'investigation_plot.png'
        Plot.save(plot_name)

if __name__ == '__main__':
    experiment(2)
    experiment(3)
    experiment(4)
    experiment(7)
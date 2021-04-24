import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import time

from hiive.mdptoolbox import mdp, example
from openai import OpenAI_MDPToolbox
import gym

RANDOM_SEED = 1994540101
np.random.seed(RANDOM_SEED) # keep results consistent

lake = OpenAI_MDPToolbox('FrozenLake-v0')

def convert_action(a):
    if a == 0:
        return '<'
    elif a == 1:
        return 'V'
    elif a == 2:
        return '>'
    else:
        return '^'

def illustrate_policy(policy, problem_name="Forest"):
    if problem_name == "Forest":
        cuts = 0
        waits = 0
        for i in policy:
            if i == 1:
                cuts += 1
            else:
                waits += 1
        return f'Number of states: {len(policy)}, Number of cuts: {cuts}, Number of waits: {waits}'
    else:
        policy_desc = [convert_action(i) for i in policy]
        policy_grid = np.reshape(policy_desc, (4, 4))
        for i in range(lake.env.desc.shape[0]):
            for j in range(lake.env.desc.shape[1]):
                s = lake.env.desc[i][j].decode('UTF-8')
                if s == 'H' or s == 'G':
                    policy_grid[i][j] = s
        return policy_grid

def run_iterations(P, R, gammas=[0.99, 0.9, 0.85, 0.8], problem_name="Forest", value_iter=True, output="output", show=False):
    policies = {}
    rewards = {}
    time = []
    Iteration = mdp.ValueIteration if value_iter else mdp.PolicyIteration
    desc = "Value_Iteration" if value_iter else "Policy_Iteration"
    for gamma in gammas:
        it = Iteration(P, R, gamma)
        results = it.run()
        col = f'Gamma: {gamma}'
        iterations = [it['Iteration'] for it in results]
        rewards[col] = pd.Series([it['Mean V'] for it in results], index=iterations)
        time.append(it.time)
        policies[col] = it.policy
    rewards = pd.DataFrame(rewards)
    rewards.index.name = 'Iterations'
    time = pd.DataFrame(time, index=gammas)
    time.index.name = 'Gamma'
    
    
    # plot and log results
    rewards.plot()
    plt.title(f"{problem_name}: {desc}: Mean Utility over Iterations")
    plt.ylabel("Mean Utility")
    plt.tight_layout()
    plt.savefig(f"{output}/{problem_name}-{desc}-utility.png")
    if show:
        plt.plot()
    
    time.plot(legend=False)
    plt.title(f"{problem_name}: {desc}: Time for different gammas")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.savefig(f"{output}/{problem_name}-{desc}-time.png")
    if show:
        plt.plot()
    else:
        plt.close()
    
    policy_desc = {key: illustrate_policy(policies[key], problem_name) for key in policies}
    print(f"Problem: {problem_name}\nFunction: {desc}\nPolicies for different gamma values:\n{policy_desc}")
    return rewards, time, policies


def q_learning(P, R, gamma=0.99 ,alpha=0.1, alpha_decay=0.99, alpha_min=0.001, epsilon=1.0, e_min=0.1, e_decay=0.999, n_iter=10000, plot=False, show=False, output="output", problem_name="Forest", callback=None):
    args = {
        "alpha": alpha,
        "alpha_decay": alpha_decay,
        "alpha_min": alpha_min,
        "epsilon": epsilon,
        "epsilon_min": e_min,
        "epsilon_decay": e_decay,
        "n_iter": n_iter,
        "iter_callback": callback
    }
    ql = mdp.QLearning(P, R, gamma, **args) 
    ql_results = ql.run()

    if plot:
        rewards = [i['Mean V'] for i in ql_results]
        iterations = [i['Iteration'] for i in ql_results]
        desc = 'Q-Learning'

        # plot and log results
        plt.clf()
        plt.plot(iterations, rewards)
        plt.title(f"{problem_name}: {desc}: Mean Utility over Iterations")
        plt.ylabel("Mean Utility")
        plt.xlabel("Iterations")
        plt.tight_layout()
        plt.savefig(f"{output}/{problem_name}-{desc}-utility.png")
        if show:
            plt.plot()
        else:
            plt.close()
        print(f'Q Learning time: {ql.time}\npolicy: {illustrate_policy(ql.policy, problem_name)}')
    return ql, ql_results

def run_qlearnings(P, R, params=[0.1, 0.25], problem_name="Forest", value_iter=True, output="output", param_alpha=True, show=False, n_iter=10000):
    policies = {}
    rewards = {}
    time = {}
    param_name = "alpha" if param_alpha else "epsilon"
    desc = f"Q-Learning_different_{param_name}s"
    
    for param in params:
        q, results = q_learning(P, R, n_iter=n_iter, **{param_name: param})
        col = f'{param_name}: {param}'
        iterations = [it['Iteration'] for it in results]
        rewards[col] = pd.Series([it['Mean V'] for it in results], index=iterations)
        time[col] = pd.Series([it['Time'] for it in results], index=iterations)
        policies[col] = q.policy
        
    rewards = pd.DataFrame(rewards)
    rewards.index.name = 'Iterations'
    time = pd.DataFrame(time)
    time.index.name = 'Iterations'
    
    # plot and log results
    rewards.plot()
    plt.title(f"{problem_name}: {desc}: Mean Utility over Iterations")
    plt.ylabel("Mean Utility")
    plt.tight_layout()
    plt.savefig(f"{output}/{problem_name}-{desc}-utility.png")
    if show:
        plt.plot()
    else:
        plt.close()
    
    time.plot()
    plt.title(f"{problem_name}: {desc}: Time over Iterations")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.savefig(f"{output}/{problem_name}-{desc}-time.png")
    if show:
        plt.plot()
    else:
        plt.close()
        
    policy_desc = {key: illustrate_policy(policies[key], problem_name) for key in policies}
    print(f"Problem: {problem_name}\nFunction: {desc}\nPolicies for different {param_name} values:\n{policy_desc}")
    return rewards, time, policies

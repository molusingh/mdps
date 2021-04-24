import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import time

from hiive.mdptoolbox import mdp, example

RANDOM_SEED = 1994540101
np.random.seed(RANDOM_SEED) # keep results consistent

def run_iterations(P, R, gammas=[0.8, 0.99], problem_name="Forest", value_iter=True, output="output", show=False):
    policies = {}
    rewards = {}
    time = {}
    Iteration = mdp.ValueIteration if value_iter else mdp.PolicyIteration
    desc = "Value_Iteration" if value_iter else "Policy_Iteration"
    for gamma in gammas:
        it = Iteration(P, R, gamma)
        results = it.run()
        col = f'Gamma: {gamma}'
        iterations = list(range(1, len(results) + 1))
        rewards[col] = pd.Series([it['Reward'] for it in results], index=iterations)
        time[col] = pd.Series([it['Time'] for it in results], index=iterations)
        policies[col] = it.policy
    rewards = pd.DataFrame(rewards)
    rewards.index.name = 'Iterations'
    time = pd.DataFrame(time)
    time.index.name = 'Iterations'
    
    
    # plot and log results
    rewards.plot()
    plt.title(f"{problem_name}: {desc}: Max Utility over Iterations")
    plt.ylabel("Max Utility")
    plt.tight_layout()
    plt.savefig(f"{output}/{problem_name}-{desc}-utility.png")
    if show:
        plt.plot()
    
    time.plot()
    plt.title(f"{problem_name}: {desc}: Time over Iterations")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.savefig(f"{output}/{problem_name}-{desc}-time.png")
    if show:
        plt.plot()
    
    print(f"Problem: {problem_name}\nFunction: {desc}\nPolicies for different gamma values:\n{policies}")
    return rewards, time, policies


def q_learning(P, R, gamma=0.99 ,alpha=0.1, alpha_decay=0.99, alpha_min=0.001, e=1.0, e_min=0.1, e_decay=0.99, n_iter=10000):
    args = {
        "alpha": alpha,
        "alpha_decay": alpha_decay,
        "alpha_min": alpha_min,
        "epsilon": e,
        "epsilon_min": e_min,
        "epsilon_decay": e_decay,
        "n_iter": n_iter
    }
    ql = mdp.QLearning(P, R, gamma, **args) 
    ql_results = ql.run()
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
        iterations = list(range(1, len(results) + 1))
        rewards[col] = pd.Series([it['Max V'] for it in results], index=iterations)
        time[col] = pd.Series([it['Time'] for it in results], index=iterations)
        policies[col] = q.policy
        
    rewards = pd.DataFrame(rewards)
    rewards.index.name = 'Iterations'
    time = pd.DataFrame(time)
    time.index.name = 'Iterations'
    
    # plot and log results
    rewards.plot()
    plt.title(f"{problem_name}: {desc}: Max Utility over Iterations")
    plt.ylabel("Max Utility")
    plt.tight_layout()
    plt.savefig(f"{output}/{problem_name}-{desc}-utility.png")
    if show:
        plt.plot()
    
    time.plot()
    plt.title(f"{problem_name}: {desc}: Time over Iterations")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.savefig(f"{output}/{problem_name}-{desc}-time.png")
    if show:
        plt.plot()
        
    print(f"Problem: {problem_name}\nFunction: {desc}\nPolicies for different alphas values:\n{policies}")
    return rewards, time, policies
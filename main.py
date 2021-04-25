#!/usr/bin/env python

import os
import sys
import argparse
import pandas as pd

from mdp import *

def main(args):
    print(f'\nBeginning experiments, provided arguments: {args}\n')

    problem_name = "Forest" 
    P, R = example.forest(S=500, r1=4, r2=2, p=0.25, is_sparse=False) # Forest 
    alphas = [0.1, 0.01, 0.001]
    epsilons = [0.95, 0.5, 0.25, 0.1]
    gammas = [0.5, 0.75, 0.9, 0.99]
    n_iter = 1000000

    if args.lake:
        P = lake.P 
        R = lake.R
        problem_name = "FrozenLake"
        n_iter = 200000
        gammas = [0.99, 0.75, 0.5, 0.25]
        alphas = [0.2, 0.1, 0.01, 0.0001]
        epsilons = [0.95, 0.5, 0.25, 0.1, 0.01]

    kargs = {
        "P": P, 
        "R": R,
        "problem_name": problem_name,
        "output": args.output
    }

    run_iterations(gammas=gammas, value_iter=True, **kargs) # value iteration
    print()
    run_iterations(gammas=gammas, value_iter=False, **kargs) # policy iteration
    print()
    run_qlearnings(params=alphas, n_iter=n_iter, param_alpha=True, **kargs) # qlearning alphas
    print()
    run_qlearnings(params=epsilons, n_iter=n_iter, param_alpha=False, **kargs) # qlearnings 
    print()
    q_learning(plot=True, **kargs, n_iter=n_iter, callback=callback)

    print(f'\nCompleted!\narguments: {args}\n')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute Markov Decision Processes experiments')
    parser.add_argument('--lake', dest='lake', action='store_true', help="run on frozen lake problem, else forest")
    parser.add_argument('--output',type=str,help='output directory, default value is "output"')
    parser.set_defaults(output='output', lake=False)
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    main(args)

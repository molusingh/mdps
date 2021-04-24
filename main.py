#!/usr/bin/env python

import os
import sys
import argparse
import pandas as pd

from mdp import *

def main(args):
    print(f'\nBeginning experiments, provided arguments: {args}\n')
    problem_name = "Forest" if not args.lake else "Frozen_Lake"
    P, R = example.forest(S=10, r1=4, r2=2, p=0.25, is_sparse=False) # Forest 
    alphas = [0.1, 0.001]
    epsilons = [0.95, 0.9]

    gammas = [0.8, 0.9, 0.95, 0.99]
    kargs = {
        "P": P, 
        "R": R,
        "problem_name": problem_name,
        "output": args.output
    }
    run_iterations(gammas=gammas, value_iter=True, **kargs) # value iteration
    run_iterations(gammas=gammas, value_iter=False, **kargs) # policy iteration
    run_qlearnings(params=alphas, n_iter=1000000, param_alpha=True, **kargs) # qlearning alphas
    run_qlearnings(params=epsilons, n_iter=1000000, param_alpha=False, **kargs) # qlearnings 
    q_learning(plot=True, **kargs)

    print(f'\nCompleted!\narguments: {args}\n')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute Markov Decision Processes experiments')
    parser.add_argument('--lake', dest='lake', action='store_false', help="run on frozen lake problem, else forest")
    parser.add_argument('--output',type=str,help='output directory, default value is "output"')
    parser.set_defaults(output='output', lake=False)
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    main(args)

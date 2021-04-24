#!/usr/bin/env python

import os
import sys
import argparse
import pandas as pd

from mdp import *

def main(args):
    print(f'\nBeginning experiments, provided arguments: {args}\n')
    print(f'\nCompleted!\narguments: {args}\n')
    P, R = example.forest(S=10, r1=40, r2=20, p=0.25, is_sparse=False)
    run_iterations(P, R, problem_name="Forest")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute Markov Decision Processes experiments')
    parser.add_argument('--lake', dest='lake', action='store_false', help="run on frozen lake problem, else forest")
    parser.add_argument('--output',type=str,help='output directory, default value is "output"')
    parser.set_defaults(output='output', lake=False)
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    main(args)

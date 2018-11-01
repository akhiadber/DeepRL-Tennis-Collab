#!/usr/bin/env python

"""
Main program.
"""

import argparse
import env
import agent
import run

parser = argparse.ArgumentParser()
parser.add_argument("--load", help="path to saved model in checkpoints dir", type=str, default=None)
parser.add_argument("--eval", help="evaluate a trained model", action="store_true")
args = parser.parse_args()

# create environment and agent, then begin training/evaluation
environment = env.UnityMultiAgent(evaluation_only=args.eval, file_name='Tennis_Linux_NoVis/Tennis.x86_64')
agent = agent.MADDPG(load_file=args.load, evaluation_only=args.eval)
run.train(environment, agent)

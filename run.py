#!/usr/bin/env python2
#
# Fit all model to Cell 5 data using CMA-ES and MCMC
import argparse
import subprocess

#
# Check whether to run optimisation or MCMC
#

parser = argparse.ArgumentParser(description='Fit all the hERG models to sine wave data')
parser.add_argument('--mode', type=int, default=1, metavar='N', \
      help='optimisation: 1, MCMC: 2, ModelStats: 3' )

args = parser.parse_args()
if args.mode == 1:
    for i in xrange(4):
        bashCommand = 'python sinefit.py --cell 5 --model '+str(i+1)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)

#!/usr/bin/env python2
#
# Fit all model to Cell 5 data using CMA-ES and MCMC
import argparse
import subprocess

#
# Check whether to run optimisation or MCMC
#

parser = argparse.ArgumentParser(
    description='Fit all the hERG models to sine wave data')
parser.add_argument('--mode', type=int, default=1, metavar='N',
                    help='optimisation: 1, AP predictions 2, MCMC: 3, ModelStats: 4')

args = parser.parse_args()
if args.mode == 1:
    for i in xrange(30):
        bashCommand = 'python cmaesfit.py --cell 5 --model ' + str(i+1)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)

elif args.mode == 2:
    bashCommand = 'python make_predictions.py --cell 5'
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)

elif args.mode == 3:
    for i in xrange(30):
        bashCommand = 'python sinemcmc.py --cell 5 --model ' + \
            str(i+1)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)

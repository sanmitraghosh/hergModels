#!/usr/bin/env python2
#
# Fit Kylie's model to Cell 5 data using CMA-ES
#
from __future__ import division, print_function
import models_forward.LogPrior as prior
import models_forward.pintsForwardModel as forwardModel
import models_forward.Rates as Rates
import models_forward.util as util
import os
import sys
import pints
import pints.plot as pplot
import numpy as np
import cPickle
import random
import myokit
import argparse
import matplotlib.pyplot as plt


# Load a hERG model and prior
cmaes_result_files = 'cmaes_results/'

# Check input arguments

parser = argparse.ArgumentParser(
    description='Make AP predictions based on the CMAES fit to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N',
                    help='cell number : 1, 2, ..., 5')
parser.add_argument('--model', type=int, default=3, metavar='N',
                    help='model number')
parser.add_argument('--plot', type=bool, default=True, metavar='N',
                    help='plot fitted traces')
args = parser.parse_args()

# Import markov models from the models file, and rate dictionaries.
model_name = 'model-'+str(args.model)
root = os.path.abspath('models_myokit')
myo_model = os.path.join(root, model_name + '.mmt')
root = os.path.abspath('rate_dictionaries')
rate_file = os.path.join(root, model_name + '-priors.p')
rate_dict = cPickle.load(open(rate_file, 'rb'))

print("loading  model: "+str(args.model))
model_name = 'model-'+str(args.model)

cell = args.cell

#
# Select data file
#
root = os.path.abspath('ap-data')
print(root)
data_file = os.path.join(root, 'cell-' + str(cell) + '.csv')
protocol_file = os.path.join(root, 'ap.csv')

#
# Cell-specific parameters
#
temperature = forwardModel.temperature(cell)
lower_conductance = forwardModel.conductance_limit(cell)

#
# Load protocol
#
log = myokit.DataLog.load_csv(data_file).npview()
prot_times = log.time()
prot_voltages = log['voltage']
del(log)
protocol = [prot_times, prot_voltages]

#
# Load data
#
log = myokit.DataLog.load_csv(data_file).npview()
time = log.time()
current = log['current']
voltage = log['voltage']
del(log)


#
# Estimate noise from start of data
# Kylie uses the first 200ms, where I = 0 + noise
#
sigma_noise = np.std(current[:2000], ddof=1)


#
# Apply capacitance filter based on protocol
#
#print('Applying capacitance filtering')
# time, voltage, current = forwardModel.capacitance(
#    protocol, 0.1, time, voltage, current)


#
# Create forward model
#
transform = 0  # we don't need to bother with transforms for a forward run...
model = forwardModel.ForwardModel(
    protocol, temperature, myo_model, rate_dict,  transform, sine_wave=False)
n_params = model.n_params

#
# Define problem
#
problem = pints.SingleOutputProblem(model, time, current)

#
# Define log-posterior
#
log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
log_prior = prior.LogPrior(
    rate_dict, lower_conductance, n_params, transform)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)
rate_checker = Rates.ratesPrior(transform, lower_conductance)

# Define starting point for mcmc routine
parameter_set = np.loadtxt(cmaes_result_files + model_name +
                           '-cell-' + str(cell) + '-cmaes.txt')
print('Model parameters start point: ', parameter_set)

ll_score = log_likelihood(parameter_set)
print('LogLikelihood for data (proportional to square error): ', ll_score)

if args.plot:

    root = os.path.abspath('figures/ap')
    if not os.path.exists(root):
        os.makedirs(root)
    ap_filename = os.path.join(
        root, model_name + '-cell-' + str(cell) + '-ap-prediction.eps')

    ap_sol = model.simulate(parameter_set, time)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time, voltage, color='orange', label='Voltage', lw=0.5)
    plt.xlim(0, 8000)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.subplot(3, 1, 2)
    plt.plot(time, current, '-', color='blue',
             lw=0.5, label='measured current')
    plt.plot(time, ap_sol, color='SeaGreen',
             lw=0.5, label='inferred current', alpha=0.1)
    plt.xlim(0, 8000)
    plt.subplot(3, 1, 3)
    plt.plot(time, current, '-', color='blue',
             lw=0.5, label='measured current')
    plt.plot(time, ap_sol, color='SeaGreen',
             lw=0.5, label='inferred current', alpha=0.1)
    plt.xlim(0, 8000)
    plt.ylim(-0.5, 2)
    plt.savefig(ap_filename)
    plt.close()

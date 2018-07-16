#!/usr/bin/env python2
#
# Show Kylie's model, for cell 5
#
# Compare with her simulation and score
#
from __future__ import division
from __future__ import print_function
import os
import sys
import pints
import pints.io
import numpy as np
import myokit

# Load beattie model and prior
sys.path.append(os.path.abspath('..'))
import beattie


# Check input arguments
args = sys.argv[1:]
if len(args) == 1:
    filename = args[0]
else:
    print('Syntax:  mcmc.py <filename>')
    sys.exit(1)


#
# Select cell
#
cell = 5


#
# Select protocols
#
protocols = [
    'pr1-activation-kinetics-1',
    'pr2-activation-kinetics-2',
    'pr3-steady-activation',
    'pr4-inactivation',
    'pr5-deactivation',
]


#
# Select data files
#
root = os.path.realpath(os.path.join('..', '..'))
data_files = [
    os.path.join(root, 'traditional-data', p + '-cell-' + str(cell) + '.csv')
    for p in protocols]


#
# Load protocols
#
print('Loading all protocols')
protocol_files = [
    os.path.join(root, 'traditional-data', p + '.mmt')
    for p in protocols]
protocols = [myokit.load_protocol(p) for p in protocol_files]


#
# Cell-specific parameters
#
temperature = beattie.temperature(cell)
lower_conductance = beattie.conductance_limit(cell)


#
# Load initial parameters
#
parameter_filename = 'solution1.txt'
print('Loading parameters from ' + parameter_filename)
with open(parameter_filename, 'r') as f:
    initial_parameters = [float(x) for x in f.readlines()]
initial_parameters = np.array(initial_parameters)


#
# Load data
#
print('Loading all data files')
times = []
currents = []
voltages = []
for data_file in data_files:
    log = myokit.DataLog.load_csv(data_file).npview()
    times.append(log.time())
    currents.append(log['current'])
    voltages.append(log['voltage'])
    del(log)


#
# Estimate noise from start of data (first 200ms, safe for all protocols)
#
sigma_noise = [np.std(current[:2000], ddof=1) for current in currents]


#
# Apply capacitance filtering based on protocols
#
print('Applying capacitance filtering')
for i, protocol in enumerate(protocols):
    times[i], voltages[i], currents[i] = beattie.capacitance(
        protocol, 0.1, times[i], voltages[i], currents[i])


#
# Create forward models
#
print('Creating forward models')
models = [
    beattie.BeattieModel(protocol, temperature, sine_wave=False)
    for protocol in protocols]


#
# Define problem
#
print('Defining problems')
problems = []
for i, model in enumerate(models):
    problems.append(pints.SingleSeriesProblem(model, times[i], currents[i]))


#
# Define log-posterior
#
print('Defining log-likelihood, log-prior, and log-posterior')
log_likelihoods = []
for i, problem in enumerate(problems):
    log_likelihoods.append(
        pints.KnownNoiseLogLikelihood(problem, sigma_noise[i]))
log_likelihood = pints.SumOfIndependentLogLikelihoods(log_likelihoods)
log_prior = beattie.BeattieLogPrior(lower_conductance)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)


#
# Run
#

# Create sampler
nchains = 3
x0 = [initial_parameters] * nchains
sigma0 = np.diag(initial_parameters) * 1e-36

mcmc = pints.MCMCSampling(log_posterior, nchains, x0, sigma0)
mcmc.set_log_to_file('log.txt')
mcmc.set_max_iterations(250000)
mcmc.set_adaptation_free_iterations(0)
mcmc.set_parallel(True)

# Run
#with np.errstate(all='ignore'): # Tell numpy not to issue warnings
chains = mcmc.run()

# Store results
pints.io.save_samples(chain_filename, *chains)

print('Done')

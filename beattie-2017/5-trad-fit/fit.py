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
    print('Syntax:  fit.py <filename>')
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

# Run repeated optimisations
repeats = 25
params, scores = [], []
for i in xrange(repeats):
    # Choose random starting point
    x0 = log_prior.sample()

    # Create optimiser
    opt = pints.Optimisation(log_posterior, x0, method=pints.CMAES)
    opt.set_max_iterations(None)
    opt.set_parallel(True)

    # Run optimisation
    try:
        with np.errstate(all='ignore'): # Tell numpy not to issue warnings
            p, s = opt.run()
            params.append(p)
            scores.append(s)
    except ValueError:
        import traceback
        traceback.print_exc()

# Order from best to worst
order = np.argsort(scores)[::-1]
scores = np.asarray(scores)[order]
params = np.asarray(params)[order]

# Show results
print('Best 3 scores:')
for i in xrange(3):
    print(scores[i])
print('Mean & std of score:')
print(np.mean(scores))
print(np.std(scores))
print('Worst score:')
print(scores[-1])

# Extract best
obtained_log_posterior = scores[0]
obtained_parameters = params[0]

# Store result
with open(filename, 'w') as f:
    for x in obtained_parameters:
        f.write(pints.strfloat(x) + '\n')


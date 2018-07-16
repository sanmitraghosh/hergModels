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
import matplotlib.pyplot as plt
import matplotlib.gridspec


# Load beattie model and prior
sys.path.append(os.path.abspath('..'))
import beattie


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
# Load parameters
#
sine_parameters = os.path.join('..', 'sine-cmaes.txt')
trad_parameters = os.path.join('..', 'trad-cmaes.txt')

def load_parameters(filename):
    print('Loading parameters from ' + filename)
    with open(filename, 'r') as f:
        return np.array([float(x) for x in f.readlines()])

sine_parameters = load_parameters(sine_parameters)
trad_parameters = load_parameters(trad_parameters)


#
# Load protocols
#
print('Loading all protocols')
protocol_files = [
    os.path.join(root, 'traditional-data', p + '.mmt')
    for p in protocols]
myokit_protocols = [myokit.load_protocol(p) for p in protocol_files]


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
for i, protocol in enumerate(myokit_protocols):
    times[i], voltages[i], currents[i] = beattie.capacitance(
        protocol, 0.1, times[i], voltages[i], currents[i])


#
# Create forward models
#
print('Creating forward models')
models = [
    beattie.BeattieModel(protocol, temperature, sine_wave=False)
    for protocol in myokit_protocols]


#
# Define problem
#
print('Defining problems')
problems = []
for i, model in enumerate(models):
    problems.append(pints.SingleSeriesProblem(model, times[i], currents[i]))


#
# Define log-likelihoods
#
print('Defining log-likelihoods, log-prior, and log-posterior')
log_likelihoods = []
for i, problem in enumerate(problems):
    log_likelihoods.append(
        pints.KnownNoiseLogLikelihood(problem, sigma_noise[i]))
log_likelihood = pints.SumOfIndependentLogLikelihoods(log_likelihoods)
log_prior = beattie.BeattieLogPrior(lower_conductance)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)


#
# Evaluate and show
#
for i, protocol in enumerate(protocols):

    print('-'*50)
    print(protocol)

    i_real = currents[i]
    i_sine = problems[i].evaluate(sine_parameters)
    i_trad = problems[i].evaluate(trad_parameters)

    print('Log-likelihood, sine fit: '
          + str(log_likelihoods[i](sine_parameters)))
    print('Log-likelihood, trad fit: '
          + str(log_likelihoods[i](trad_parameters)))

    '''
    plt.figure()
    plt.suptitle(protocol)
    matplotlib.gridspec.GridSpec(3, 3)

    plt.subplot2grid((3, 1), (0, 0))
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')
    plt.plot(times[i], voltages[i], color='tab:green')

    plt.subplot2grid((3, 1), (1, 0), rowspan=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('I (nA)')
    plt.plot(times[i], i_real, label='Cell ' + str(cell))
    plt.plot(times[i], i_sine, label='Sine fit')
    plt.plot(times[i], i_trad, label='Trad. fit')
    plt.legend()
    '''

    beattie.fold_plot(
        protocol, times[i], voltages[i],
        [i_real, i_sine, i_trad],
        ['Cell 5', 'Sine', 'Trad. prot.']
    )
    plt.suptitle(protocol)

print('='*50)
print('Combined')
print('Log-posterior, sine fit: ' + str(log_posterior(sine_parameters)))
print('Log-posterior, trad fit: ' + str(log_posterior(trad_parameters)))

plt.show()

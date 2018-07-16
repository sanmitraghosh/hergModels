#!/usr/bin/env python2
#
# Compare AP simulations from different fitting methods
#
#
from __future__ import division
from __future__ import print_function
import os
import sys
import pints
import numpy as np
import myokit
import matplotlib.pyplot as pl

# Load beattie model and prior
sys.path.append(os.path.abspath('..'))
import beattie


#
# Select cell
#
cell = 5


#
# Select data file
#
root = os.path.realpath(os.path.join('..', '..'))
data_file = os.path.join(root, 'ap-data', 'ap-cell-' + str(cell) + '.csv')


#
# Cell-specific parameters
#
temperature = beattie.temperature(cell)
lower_conductance = beattie.conductance_limit(cell)


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
# Load data (with embedded protocol)
#
print('Loading data and protocol file')
log = myokit.DataLog.load_csv(data_file).npview()
time = log.time()
current = log['current']
voltage = log['voltage']
del(log)


#
# Estimate noise from start of data (first 200ms, safe for all protocols)
#
sigma_noise = np.std(current[:2000], ddof=1)


#
# Create forward model
#
print('Creating forward model')
model = beattie.BeattieModel((time, voltage), temperature, sine_wave=False)


#
# Define problem
#
print('Defining problem')
problem = pints.SingleSeriesProblem(model, time, current)


#
# Define log-posterior
#
print('Defining log-likelihood, log-prior, and log-posterior')
log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
log_prior = beattie.BeattieLogPrior(lower_conductance)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)


#
# Evaluate and show
#

i_real = current
i_sine = problem.evaluate(sine_parameters)
i_trad = problem.evaluate(trad_parameters)

print('Log-posterior, sine fit: ' + str(log_posterior(sine_parameters)))
print('Log-posterior, trad fit: ' + str(log_posterior(trad_parameters)))

pl.figure()

pl.subplot(4, 1, 1)
pl.xlabel('Time (ms)')
pl.ylabel('V (mV)')
pl.plot(time, voltage, label='V (real)')
#pl.plot(time, model.simulated_v, label='V (simulated)')
pl.legend()

pl.subplot(4, 1, 2)
pl.xlabel('Time (ms)')
pl.ylabel('I (nA)')
pl.plot(time, i_real, label='cell 5')
pl.plot(time, i_sine, label='sine fit')
pl.plot(time, i_trad, label='trad fit')
pl.legend()

pl.subplot(2, 1, 2)
pl.xlabel('Time (ms)')
pl.ylabel('I (nA)')
pl.plot(time, i_real, label='cell 5')
pl.plot(time, i_sine, label='sine fit')
pl.plot(time, i_trad, label='trad fit')
pl.legend()
pl.xlim(3500, 7000)
pl.ylim(0, 1)

pl.show()


#!/usr/bin/env python2
#
# Compare different fits on the sine wave data set
#
from __future__ import division
from __future__ import print_function
import os
import sys
import pints
import numpy as np
import myokit
import matplotlib.pyplot as plt

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
data_file = os.path.join(root, 'sine-wave-data', 'cell-' + str(cell) + '.csv')


#
# Select protocol file
#
protocol_file = os.path.join(root, 'sine-wave-data', 'steps.mmt')


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
# Load protocol
#
protocol = myokit.load_protocol(protocol_file)


#
# Load data
#
log = myokit.DataLog.load_csv(data_file).npview()
time = log.time()
current = log['current']
voltage = log['voltage']
del(log)


#
# Estimate noise from start of data (first 200ms works for every protocol)
#
sigma_noise = np.std(current[:2000], ddof=1)


#
# Cell-specific parameters
#
temperature = beattie.temperature(cell)
lower_conductance = beattie.conductance_limit(cell)


#
# Apply capacitance filtering based on protocols
#
time, voltage, current = beattie.capacitance(
        protocol, 0.1, time, voltage, current)


#
# Create forward model
#
print('Creating forward model')
model = beattie.BeattieModel(protocol, temperature, sine_wave=True)


#
# Define problem
#
print('Defining problem')
problem = pints.SingleSeriesProblem(model, time, current)


#
# Define log-posterior
#
print('Defining log-posterior')
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

plt.figure()

plt.subplot(4, 1, 1)
plt.xlabel('Time (ms)')
plt.ylabel('V (mV)')
plt.plot(time, voltage, color='tab:green')

plt.subplot(4, 1, 2)
plt.xlabel('Time (ms)')
plt.ylabel('I (nA)')
plt.plot(time, i_real, label='Cell ' + str(cell))
plt.plot(time, i_sine, label='Sine fit')
plt.plot(time, i_trad, label='Trad. fit')
plt.legend()

plt.subplot(2, 1, 2)
plt.xlim(4000, 6000)
plt.ylim(-1.1, 1.3)
plt.xlabel('Time (ms)')
plt.ylabel('I (nA)')
plt.plot(time, i_real, label='Cell ' + str(cell))
plt.plot(time, i_sine, label='Sine fit')
plt.plot(time, i_trad, label='Trad. fit')
plt.legend()

plt.show()


#!/usr/bin/env python2
#
# Show traditional protocol data plus simulation
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


#
# Select cell
#
cell = 5


#
# Select protocol
#
protocol = 'pr1-activation-kinetics-1'
protocol = 'pr2-activation-kinetics-2'
protocol = 'pr3-steady-activation'
#protocol = 'pr4-inactivation'
#protocol = 'pr5-deactivation'


#
# Select data file
#
root = os.path.realpath(os.path.join('..', '..'))
data_file = os.path.join(
    root, 'traditional-data', protocol + '-cell-' + str(cell) + '.csv')


#
# Load protocol
#
protocol_file = os.path.join(root, 'traditional-data', protocol + '.mmt')
myokit_protocol = myokit.load_protocol(protocol_file)


#
# Cell-specific parameters
#
temperature = beattie.temperature(cell)
lower_conductance = beattie.conductance_limit(cell)


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
#
sigma_noise = np.std(current[:2000], ddof=1)


#
# Apply capacitance filter based on protocol
#
print('Applying capacitance filtering')
time, voltage, current = beattie.capacitance(
    myokit_protocol, 0.1, time, voltage, current)


#
# Create ForwardModel
#
model = beattie.BeattieModel(myokit_protocol, temperature, sine_wave=False)


#
# Define problem
#
problem = pints.SingleSeriesProblem(model, time, current)


#
# Define a log-posterior
#
log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
log_prior = beattie.BeattieLogPrior(lower_conductance)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)


#
# Load earlier result
#
filename = '../1-sine-simulations/kylies-solution.txt'
with open(filename, 'r') as f:
    obtained_parameters = [float(x) for x in f.readlines()]
obtained_log_posterior = log_posterior(obtained_parameters)


#
# Show obtained parameters and score
#
print('Kylie sine-wave parameters:')
for x in obtained_parameters:
    print(pints.strfloat(x))
print('Final log-posterior:')
print(pints.strfloat(obtained_log_posterior))


#
# Show result
#

# Simulate
simulated = model.simulate(obtained_parameters, time)

# Plot
import matplotlib.pyplot as pl

#
# Full comparison: protocol, transitions, current, current at transitions
# This plot works best with the capacitance filter switched off
#
'''
pl.figure()
pl.subplot(4, 1, 1)
pl.xlabel('Time (ms)')
pl.ylabel('V (mV)')
pl.plot(time, voltage, '-', alpha=0.75, label='measured')
pl.plot(time, model.simulated_v, '-', alpha=0.75, label='simulated')
pl.legend()
pl.subplot(4, 1, 2)
pl.xlabel('Time (ms)')
pl.ylabel('V (mV)')
pl.plot(time, voltage - model.simulated_v, '-', label='real - simulated')
pl.legend()
pl.subplot(4, 1, 3)
pl.xlabel('Time (ms)')
pl.ylabel('I (nA)')
pl.plot(time, current, '-', alpha=0.75, label='measured')
pl.plot(time, simulated, '-', alpha=0.75, label='simulated')
pl.legend()
pl.subplot(4, 1, 4)
pl.xlabel('Time (ms)')
pl.ylabel('I (nA)')
pl.plot(time, current - simulated, '-', label='real - simulated')
pl.legend()
'''

beattie.fold_plot(protocol, time, voltage, [current, simulated])

# Finalise
pl.show()

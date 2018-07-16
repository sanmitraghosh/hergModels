#!/usr/bin/env python2
#
# Fit Kylie's model to Cell 5 data using CMA-ES
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
if len(args) == 1 and args[0][:1] != '-':
    chain_filename = args[0]
else:
    print('Syntax:  1-mcmc.py <filename>')
    sys.exit(1)


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
# Cell-specific parameters
#
temperature = beattie.temperature(cell)
lower_conductance = beattie.conductance_limit(cell)


#
# Load initial parameters
#
parameter_filename = 'solution3.txt'
print('Loading parameters from ' + parameter_filename)
with open(parameter_filename, 'r') as f:
    initial_parameters = [float(x) for x in f.readlines()]
initial_parameters = np.array(initial_parameters)


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
# Estimate noise from start of data
# Kylie uses the first 200ms, where I = 0 + noise
#
sigma_noise = np.std(current[:2000], ddof=1)


#
# Apply capacitance filter based on protocol
#
print('Applying capacitance filtering')
time, voltage, current = beattie.capacitance(
    protocol, 0.1, time, voltage, current)


#
# Create forward model
#
model = beattie.BeattieModel(protocol, temperature, sine_wave=True)


#
# Define problem
#
problem = pints.SingleSeriesProblem(model, time, current)


#
# Define log-posterior
#
log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
log_prior = beattie.BeattieLogPrior(lower_conductance)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)


#
# Run
#

# Create sampler
nchains = 3
x0 = [initial_parameters] * nchains

mcmc = pints.MCMCSampling(log_posterior, nchains, x0)
mcmc.set_log_to_file('log.txt')
mcmc.set_max_iterations(250000)
mcmc.set_parallel(True)

# Run
with np.errstate(all='ignore'): # Tell numpy not to issue warnings
    chains = mcmc.run()

# Store results
pints.io.save_samples(chain_filename, *chains)

print('Done')

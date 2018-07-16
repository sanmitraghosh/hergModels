#!/usr/bin/env python2
#
# Fit Kylie's model to Cell 5 data using CMA-ES
#
from __future__ import division, print_function
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
show = '-show' in args
args = [x for x in args if x != '-show']
if len(args) == 1 and args[0][:1] != '-':
    filename = args[0]
else:
    print('Syntax:  fit.py <filename>')
    print('    or:  fit.py <filename> -show')
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
# Run, or load earlier result
#
if show:
    # Load existing result
    with open(filename, 'r') as f:
        obtained_parameters = [float(x) for x in f.readlines()]
    obtained_log_posterior = log_posterior(obtained_parameters)

else:

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


#
# Load Kylie's solution
#
with open('kylies-solution.txt', 'r') as f:
    kylie = [float(x) for x in f.readlines()]


#
# Show obtained parameters and log posterior
#
print('Obtained parameters:')
for i, x in enumerate(obtained_parameters):
    x = pints.strfloat(x)
    y = pints.strfloat(kylie[i])
    print(x + ' New')
    print(y + ' Kylie')
    for j, c in enumerate(x):
        if c != y[j:j+1]:
            print(' ' * j + '^')
            break
    print('')
print('Final log-posterior:')
print(pints.strfloat(obtained_log_posterior))


#
# Show result
#
import matplotlib.pyplot as pl
pl.figure()
pl.subplot(2,1,1)
pl.plot(time, voltage)
pl.subplot(2,1,2)
pl.plot(time, current, label='real')
pl.plot(time, model.simulate(obtained_parameters, time), label='fit')
pl.legend(loc='lower right')
pl.show()

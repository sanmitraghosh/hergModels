#!/usr/bin/env python2
#
# All model to Cell 5 data using CMA-ES
#
from __future__ import division, print_function
import models_forward.pintsForwardModel as forwardModel
import models_forward.LogPrior as prior
import models_forward.Rates as Rates
import models_forward.util as util
import os
import sys
import pints
import numpy as np
import myokit
import argparse
import cPickle
import matplotlib.pyplot as plt


# Check input arguments

#
# Select cell
#

parser = argparse.ArgumentParser(
    description='Fit all the hERG models to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N',
                    help='cell number : 1, 2, ..., 5')
parser.add_argument('--model', type=int, default=16, metavar='N',
                    help='model number : 1 for C-O-I-IC, 2 for C-O and so on')
parser.add_argument('--transform', type=int, default=1, metavar='N',
                    help='Choose between loglog/loglinear parameter transform : 1 for loglinear, 2 for loglog'), \
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

sys.path.append(os.path.abspath('models_forward'))


print("loading  model: "+str(args.model))
model_name = 'model-'+str(args.model)

cell = args.cell

#
# Select data file
#
root = os.path.abspath('sine-wave-data')
print(root)
data_file = os.path.join(root, 'cell-' + str(cell) + '.csv')


#
# Select protocol file
#
protocol_file = os.path.join(root, 'steps.mmt')


#
# Cell-specific parameters
#
temperature = forwardModel.temperature(cell)
lower_conductance = forwardModel.conductance_limit(cell)


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
time, voltage, current = forwardModel.capacitance(
    protocol, 0.1, time, voltage, current)


#
# Create forward model
#
transform = args.transform
model = forwardModel.ForwardModel(
    protocol, temperature, myo_model, rate_dict,  transform, sine_wave=True, logTransform=True)
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
    rate_dict, lower_conductance, n_params,  transform, logTransform=True)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)
rate_checker = Rates.ratesPrior(lower_conductance)


# Run repeated optimisations
repeats = 5
params, scores = [], []

func_calls = []
for i in xrange(repeats):
    # Choose random starting point

    if i == 0:
        gary_guess = []
        for j in xrange(int(n_params/2)):
            gary_guess.append(2e-3)  # A parameter [in A*exp(+/-B*V)]
            gary_guess.append(0.05)  # B parameter [in A*exp(+/-B*V)]
        gary_guess.append(2*lower_conductance)

        x0 = np.array(gary_guess)
    else:
        x0 = log_prior.sample()
    print('Initial guess (untransformed model parameters) = ', x0)

    # Create optimiser and log transform parameters
    if transform == 1:
        x0 = util.transformer('loglinear', x0, rate_dict, True)
        boundaries = rate_checker._get_boundaries('loglinear', rate_dict)
    elif transform == 2:
        x0 = util.transformer('loglog', x0, rate_dict, True)
        boundaries = rate_checker._get_boundaries('loglog', rate_dict)

    Boundaries = pints.RectangularBoundaries(boundaries[0], boundaries[1])

    print('Initial guess LogLikelihood = ', log_likelihood(x0))
    print('Initial guess LogPrior = ',      log_prior(x0))
    print('Initial guess LogPosterior = ',  log_posterior(x0))

    print('Initial guess (transformed optimisation parameters) = ', x0)
    opt = pints.Optimisation(
        log_posterior, x0, boundaries=Boundaries, method=pints.CMAES)
    opt.set_max_iterations(None)
    opt.set_parallel(True)
    # opt.set_log_to_file(filename, csv=True)

    # Run optimisation
    try:
        with np.errstate(all='ignore'):  # Tell numpy not to issue warnings
            p, s = opt.run()
            if transform == 1:
                p = util.transformer('loglinear', p, rate_dict, False)
            elif transform == 2:
                p = util.transformer('loglog', p, rate_dict, False)

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

if repeats > 1:
    print('Best 3 scores:')
    for i in xrange(3):
        print(scores[i])
        print('Mean & std of score:')
        print(np.mean(scores))
        print(np.std(scores))
        print('Worst score:')
        print(scores[-1])
else:
    print('Score:')
    print(scores)
    print('Best parameter set:')
    print(params)

# Extract best
obtained_log_posterior = scores[0]
obtained_parameters = params[0]

root = os.path.abspath('cmaes_results')
cmaes_filename = os.path.join(
    root, model_name + '-cell-' + str(cell) + '-cmaes.txt')

with open(cmaes_filename, 'w') as f:
    for x in obtained_parameters:
        f.write(pints.strfloat(x) + '\n')

print ('CMAES fitting is done for model', args.model)
#
# Show result
#
#
# re-create forward model
#

plot = args.plot
if plot:
    root = os.path.abspath('figures/cmaesfit')
    fig_filename = os.path.join(
        root, model_name + '-cell-' + str(cell) + '-cmaes_test.eps')
    model = forwardModel.ForwardModel(
        protocol, temperature, myo_model, rate_dict, transform, sine_wave=True, logTransform=False)
    print('Writing plot to ', fig_filename)

    plt.figure()
    plt.subplot(2, 1, 1)
    # plt.plot(time, voltage)
    plt.subplot(2, 1, 2)
    plt.plot(time, current, label='real')
    plt.plot(time, model.simulate(obtained_parameters, time), label='fit')
    plt.legend(loc='lower right')
    plt.savefig(fig_filename)   # save the figure to file
    plt.close()

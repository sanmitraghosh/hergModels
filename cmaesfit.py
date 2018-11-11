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
parser.add_argument('--model', type=int, default=3, metavar='N',
                    help='model number : 1 for C-O, 2 for C-O-I and so on')
parser.add_argument('--repeats', type=int, default=10, metavar='N',
                    help='number of CMA-ES runs from different initial guesses')
parser.add_argument('--transform', type=int, default=1, metavar='N',
                    help='Choose between loglog/loglinear parameter transform : 0 for no transform, 1 for loglinear, 2 for loglog')
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

results_log_folder = 'cmaes_results/individual_runs'
if not os.path.exists(results_log_folder):
    os.makedirs(results_log_folder)


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
protocol = myokit.load_protocol(protocol_file)

#
# Cell-specific parameters
#
temperature = forwardModel.temperature(cell)
lower_conductance = forwardModel.conductance_limit(cell)

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
    protocol, temperature, myo_model, rate_dict, transform, sine_wave=1)
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


# Run repeated optimisations
params, scores = [], []

func_calls = []
for i in xrange(args.repeats):
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
    x0 = util.transformer(transform, x0, rate_dict, True)
    boundaries = rate_checker._get_boundaries(rate_dict)
    Boundaries = pints.RectangularBoundaries(boundaries[0], boundaries[1])

    print('Initial guess LogLikelihood = ', log_likelihood(x0))
    print('Initial guess LogPrior = ',      log_prior(x0))
    print('Initial guess LogPosterior = ',  log_posterior(x0))

    print('Initial guess (transformed optimisation parameters) = ', x0)
    opt = pints.Optimisation(
        log_posterior, x0, boundaries=Boundaries, method=pints.CMAES)
    opt.set_max_iterations(None)
    opt.set_parallel(True)
    log_filename = model_name + '_cell_' + \
        str(cell) + '_transform_' + str(transform) + \
        '_cmaes_run_' + str(i) + '.log'
    opt.set_log_to_file(results_log_folder + '/' + log_filename, csv=True)

    # Run optimisation
    try:
        with np.errstate(all='ignore'):  # Tell numpy not to issue warnings
            p, s = opt.run()
            p = util.transformer(transform, p, rate_dict, False)

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

if args.repeats > 2:
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

transformed_best_params = util.transformer(transform, obtained_parameters, rate_dict, True)
obtained_log_likelihood = log_likelihood(transformed_best_params)
print('Log likelihood for best parameter set is: ', obtained_log_likelihood)

root = os.path.abspath('cmaes_results')
cmaes_filename = os.path.join(
    root, model_name + '-cell-' + str(cell) + '-cmaes.txt')

write_out_results = False

# Check to see if we have done a minimisation before
if os.path.isfile(cmaes_filename):
    previous_params = np.loadtxt(cmaes_filename)
    # Check what likelihood that was
    transformed_best_params = util.transformer(transform, previous_params, rate_dict, True)
    previous_best = log_likelihood(transformed_best_params)
    print('Previous best log likelihood was: ', previous_best)
    if obtained_log_likelihood > previous_best:
        print('Overwriting previous results')
        write_out_results = True
else:
    write_out_results = True

if write_out_results:
    with open(cmaes_filename, 'w') as f:
        for x in obtained_parameters:
            f.write(pints.strfloat(x) + '\n')

print ('CMAES fitting is done for model', args.model)
#
# Show result
#
if args.plot and write_out_results:
    root = os.path.abspath('figures/cmaesfit')
    fig_filename = os.path.join(
        root, model_name + '-cell-' + str(cell) + '-cmaes.eps')
    print('Writing plot to ', fig_filename)

    plt.figure()
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})

    a0.plot(time, voltage)
    a0.set_ylabel('Voltage (mV)')

    a1.plot(time, current, label='real', lw=0.5)
    a1.plot(time, model.simulate(util.transformer(
        transform, obtained_parameters, rate_dict, True), time), label='fit', lw=0.5)
    a1.legend(loc='lower right')
    a1.set_xlabel('Time (ms)')
    a1.set_ylabel('Current (nA)')
    plt.savefig(fig_filename)   # save the figure to file
    plt.close()

#!/usr/bin/env python2
#
# Fit Kylie's model to Cell 5 data using CMA-ES
#
from __future__ import division, print_function
import models_forward.LogPrior as prior
import models_forward.tiLogLikelihood as LogLikelihood
import models_forward.pintsForwardModel as forwardModel
import models_forward.Rates as Rates
import models_forward.util as util
import os
import sys
import pints
import pints.io
import pints.plot as pplot
import numpy as np
import cPickle
import random
import myokit
import argparse
import matplotlib.pyplot as plt
import mcmcsampling

# Load a hERG model and prior
cmaes_result_files = 'cmaes_results/'

# Check input arguments

parser = argparse.ArgumentParser(
    description='Fit all the hERG models to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N',
                    help='cell number : 1, 2, ..., 5')
parser.add_argument('--model', type=int, default=3, metavar='N',
                    help='model number')
parser.add_argument('--plot', type=bool, default=True, metavar='N',
                    help='plot fitted traces')
parser.add_argument('--transform', type=int, default=1, metavar='N',
                    help='Choose between loglog/loglinear parameter transform : 1 for loglinear, 2 for loglog')
parser.add_argument('--niter', type=int, default=100000, metavar='N',
                    help='number of mcmc iterations')
parser.add_argument('--burnin', type=int, default=50000, metavar='N',
                    help='number of burn-in samples')
args = parser.parse_args()

# Import markov models from the models file, and rate dictionaries.
model_name = 'model-'+str(args.model)
root = os.path.abspath('models_myokit')
myo_model = os.path.join(root, model_name + '.mmt')
root = os.path.abspath('rate_dictionaries')
rate_file = os.path.join(root, model_name + '-priors.p')
rate_dict = cPickle.load(open(rate_file, 'rb'))

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
transform = 0#args.transform
model = forwardModel.ForwardModel(
    protocol, temperature, myo_model, rate_dict,  transform, sine_wave=1)
n_params = model.n_params
#
# Define problem
#
problem = pints.SingleOutputProblem(model, time, current)


#
# Define log-posterior
#
log_likelihood = LogLikelihood.tiLogLikelihood(problem, sigma_noise, voltage)
log_prior_rates = prior.LogPrior(
    rate_dict, lower_conductance, n_params, transform)
log_prior_discrp = pints.NormalLogPrior(0.,1.5)
log_prior = pints.ComposedLogPrior(log_prior_rates,log_prior_discrp)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)
rate_checker = Rates.ratesPrior(transform, lower_conductance)

#
# Run
#
nchains = 1

# Define starting point for mcmc routine
xs = []

x0 = np.loadtxt(cmaes_result_files + model_name +
                '-cell-' + str(cell) + '-cmaes.txt')
ds = log_prior_discrp.sample().reshape((1,))
x0 = [np.concatenate((x0,ds))]
print('Model parameters start point: ', x0)
"""
x0 = util.transformer(transform, x0, rate_dict, True)
for i in xrange(nchains):
    xs.append(x0)
"""
print('MCMC starting point: ')
for x0 in xs:
    print(x0)
print('MCMC starting Log-Posterior: ')
for x0 in xs:
    print(log_posterior(x0))
print(log_posterior(x0))
# Create sampler
mcmc = mcmcsampling.MCMCSampling(log_posterior, nchains, x0,
                                 method=pints.AdaptiveCovarianceMCMC)

# mcmc.set_log_to_file('log.txt')
mcmc.set_log_to_screen(True)
iterations = args.niter
mcmc.set_max_iterations(iterations)
mcmc.set_parallel(True)
# Run
with np.errstate(all='ignore'):  # Tell numpy not to issue warnings

    trace, LLs = mcmc.run(returnLL=True)
    if nchains > 1:
        print('R-hat:')
        print(pints.rhat_all_params(trace))
# save traces
root = os.path.abspath('mcmc_results')
if not os.path.exists(root):
    os.makedirs(root)
param_filename = os.path.join(
    root, model_name + '-cell-' + str(cell) + '-mcmc_traces.p')
cPickle.dump(trace, open(param_filename, 'wb'))
pints.io.save_samples('mcmc_results/%s-chain.csv' %
                      (model_name + '-cell-' + str(cell)), *trace)
pints.io.save_samples('mcmc_results/%s-LLs.csv' %
                      (model_name + '-cell-' + str(cell)), LLs)


burnin = args.burnin
samples_all_chains = trace[:, burnin:, :]
sample_chain_1 = samples_all_chains[0]


plot = args.plot
print(plot)
# Plot
if plot:

    root = os.path.abspath('figures/mcmc')    
    if not os.path.exists(root):
        os.makedirs(root)

    ppc_filename = os.path.join(
        root, model_name + '-cell-' + str(cell) + '-mcmc_ppc.eps')
    pairplt_filename = os.path.join(
        root, model_name + '-cell-' + str(cell) + '-mcmc_pairplt.eps')
    traceplt_filename = os.path.join(
        root, model_name + '-cell-' + str(cell) + '-mcmc_traceplt.eps')

    new_values = []
    for ind in random.sample(range(0, np.size(sample_chain_1, axis=0)), 400):
        ppc_sol = model.simulate(sample_chain_1[ind, :n_params], time)
        new_values.append(ppc_sol)
    new_values = np.array(new_values)
    mean_values = np.mean(new_values, axis=0)
    new_values.shape
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time, voltage, color='orange', label='measured voltage', lw=0.5)
    plt.xlim(0, 8000)
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(time, current, '-', color='blue',
             lw=0.5, label='measured current')
    for i in xrange(np.size(new_values, axis=0)-1):
        plt.plot(time, new_values[i, :], color='SeaGreen',
                 lw=0.5, label='inferred current', alpha=0.05)
    plt.xlim(0, 8000)
    # plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(time[-40000:], current[-40000:], '-', color='blue',
             lw=0.5, label='measured current blow-up')
    for i in xrange(np.size(new_values, axis=0)-1):
        plt.plot(time[-40000:], new_values[i, -40000:], color='SeaGreen',
                 lw=0.5, label='inferred current', alpha=0.05)
    plt.xlim(4000, 6000)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current')
    # plt.legend()
    plt.savefig(ppc_filename)
    plt.close()

    pplot.pairwise(sample_chain_1[:, :n_params], opacity=1)
    plt.savefig(pairplt_filename)
    plt.close()

    pplot.trace(samples_all_chains)
    plt.savefig(traceplt_filename)
    plt.close()

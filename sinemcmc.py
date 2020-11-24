#!/usr/bin/env python2
#
# Fit hERG ion-channel models to sine-wave data using MCMC, with initialsation from CMA-ES
#
from __future__ import division, print_function
import models_forward.LogPrior as prior
import models_forward.tiLogLikelihood as LogLikelihood
import models_forward.pintsForwardModel as forwardModel
import models_forward.tiLogLikelihood as tiLogLikelihood
import models_forward.dsLogLikelihood as dsLogLikelihood
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
from joblib import Parallel, delayed
import multiprocessing
import time as timer

# Load a hERG model and prior
cmaes_result_files = 'cmaes_results/'

# Check input arguments

parser = argparse.ArgumentParser(
    description='Fit all the hERG models to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N',
                    help='cell number : 1, 2, ..., 5')
parser.add_argument('--model', type=int, default=3, metavar='N',
                    help='model number')
parser.add_argument('--thermo', type=bool, default=False, metavar='N',
                    help='Run thermodynamic integration (takes longer)')
parser.add_argument('--discrepancy', type=bool, default=False, metavar='N',
                    help='Run with discrepancy noise')
parser.add_argument('--plot', type=bool, default=True, metavar='N',
                    help='plot fitted traces')
parser.add_argument('--transform', type=int, default=1, metavar='N',
                    help='Choose between loglog/loglinear parameter transform : 1 for loglinear, 2 for loglog')
parser.add_argument('--init_ds', type=bool, default=False, metavar='N',
                    help='Initial values of chais taken from CMA-ES fit of discrepancy model')
parser.add_argument('--niter', type=int, default=150000, metavar='N',
                    help='number of mcmc iterations')
parser.add_argument('--burnin', type=int, default=100000, metavar='N',
                    help='number of burn-in samples')
parser.add_argument('--ntemps', type=int, default=8, metavar='N',
                    help='number of temperatures in geomteric ladder')
parser.add_argument('--ppc_samples', type=int, default=1000, metavar='N',
                    help='number of burn-in samples')
args = parser.parse_args()
if args.thermo and args.discrepancy:
    raise ValueError('Running thermodynamic integration for discrepancy model not supported currently')
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
transform = 0
model = forwardModel.ForwardModel(
    protocol, temperature, myo_model, rate_dict,  transform, sine_wave=1)
n_params = model.n_params
#
# Define problem for pints
#
problem = pints.SingleOutputProblem(model, time, current)
if args.discrepancy:
    log_prior_model_params = prior.LogPrior(rate_dict, lower_conductance, n_params, transform)
    log_prior_current_ds = pints.NormalLogPrior(0.1,0.5)
    log_prior_voltage_ds = pints.NormalLogPrior(0.1,0.5)
    log_prior = pints.ComposedLogPrior(log_prior_model_params, log_prior_current_ds, log_prior_voltage_ds)
else:
    log_prior = prior.LogPrior(rate_dict, lower_conductance, n_params, transform)
burnin = args.burnin
iterations = args.niter
do_thermo = args.thermo
if do_thermo:
    #
    # Define a MCMC runner function that will run a single MCMC chain for a given 
    # Thermodynamic temperature `t`
    #
    nchains = 1
    num_temps = args.ntemps
    print('Running in iid noise mode')
    timer.sleep(2.5) 
    def mcmc_runner(temps, n_chains = nchains):
        #
        # Define the unnormalised tempered target density 
        #   
        
        tempered_log_likelihood = tiLogLikelihood.temperedLogLikelihood(problem, sigma_noise, temps)
        tempered_log_posterior = pints.LogPosterior(tempered_log_likelihood, log_prior)

        # Define starting point for mcmc routine
        xs = []

        x0 = np.loadtxt(cmaes_result_files + model_name +
                        '-cell-' + str(cell) + '-cmaes.txt')
        print('Model parameters start point: ', x0)


        for _ in xrange(nchains):
            xs.append(x0)

        print('MCMC starting point: ')
        for x0 in xs:
            print(x0)

        print('MCMC starting Log-Posterior: ')
        for x0 in xs:
            print(tempered_log_likelihood(x0))
        print(tempered_log_likelihood(x0))
        # Create sampler
        mcmc = mcmcsampling.MCMCSampling(tempered_log_posterior, n_chains, xs,
                                        method=pints.AdaptiveCovarianceMCMC)

        
        mcmc.set_log_to_screen(False)

        mcmc.set_max_iterations(iterations)
        mcmc.set_parallel(False)

        trace, LLs = mcmc.run(returnLL=True)

        return trace, LLs

else:
    nchains = 1 #change this later
    # Load CMA-ES params
    xs = []
    if args.discrepancy and args.init_ds:
        print('Initial values of chain from discrepancy model')   
        timer.sleep(1.5) 
        x0 = np.loadtxt(cmaes_result_files + model_name +
                    '-cell-' + str(cell) + '-cmaes_ds.txt')  
    else:
        x0 = np.loadtxt(cmaes_result_files + model_name +
                    '-cell-' + str(cell) + '-cmaes.txt')          
    # Define Likelihood 
    if args.discrepancy:
        if not(args.init_ds):
            ds1 = log_prior_current_ds.sample().reshape((1,))
            ds2 = log_prior_voltage_ds.sample().reshape((1,))
            x0 = np.concatenate((x0,ds1,ds2))    
            print('Initial values of chain from iid noise model')   
            timer.sleep(1.5)  
        model_inputs = [voltage, current]
        log_likelihood = dsLogLikelihood.discrepancyLogLikelihood(problem, sigma_noise, model_inputs)   
        print('Experimental Warning: Running in discrepancy mode')
        timer.sleep(2.5)      
    else:
        print('Running in iid noise mode')
        timer.sleep(2.5) 
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma_noise)

    # And posterior
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)


    print('Model parameters start point: ', x0)
    for _ in xrange(nchains):
        xs.append(x0)
    print('MCMC starting point: ')
    for x0 in xs:
        print(x0)
    print('MCMC starting Log-Posterior: ')
    for x0 in xs:
        print(log_likelihood(x0))
    print(log_likelihood(x0))

    # Create sampler
    mcmc = mcmcsampling.MCMCSampling(log_posterior, nchains, xs,
                                    method=pints.AdaptiveCovarianceMCMC)

    # mcmc.set_log_to_file('log.txt')
    mcmc.set_log_to_screen(True)
    iterations = args.niter
    mcmc.set_max_iterations(iterations)
    mcmc.set_parallel(True)
    
# Run the Sampler
with np.errstate(all='ignore'):  # Tell numpy not to issue warnings
    
    if do_thermo:
        # Create a geometric temperature ladder in [0,1]
        temperature = np.logspace(-3, 0, num_temps)
        num_cores = multiprocessing.cpu_count()
        print('Running in thermodynamic mode')
        timer.sleep(5.5) 
        results = Parallel(n_jobs=num_cores-2)(delayed(mcmc_runner)(t) for t in temperature)
        #trace, LLs = mcmc_runner(1)
        #
        # Extract the MCMC outputs    
        #
        # These are for thermodynamic integration

        # Save the trace from the chain with t=1
        trace = results[len(temperature)-1][0]
        tempered_LLs = np.array([results[i][1] for i in range(len(temperature))]).reshape((len(temperature),iterations))
        untempered_LLs = np.array([tempered_LLs[i,:]/temperature[i] for i in range(len(temperature))])
        loglike = untempered_LLs[:,burnin:].T
        #
        # This next function carries out the thermodynamic integration to find 
        # the log Marginal likelihood Z = \int p(y|\theta)p(\theta)d(\theta)
        #
        def thermo_int(likelihoods, temperatures, print_schedule = False):

            # This function carries out the thermodynamic integration with
            # the Friel correction
            ti=temperatures
            if print_schedule:
                print('The temperature schedule is :',ti)
            Eloglike_std = np.mean(likelihoods,axis=0)
            E2loglike_std = np.mean(likelihoods**2,axis=0)
            Vloglike_std = E2loglike_std - (np.mean(likelihoods,axis=0))**2
            I_MC = []

            for i in xrange(len(ti)-1):

                I_MC.append( (Eloglike_std[i] + Eloglike_std[i+1])/2 * (ti[i+1]-ti[i]) \
                        - (Vloglike_std[i+1] - Vloglike_std[i])/12 * (ti[i+1]-ti[i])**2  ) 
            
            return np.sum(I_MC)
        # Get the log Z
        logZ = thermo_int(loglike, temperature, print_schedule = True)
        print('The log marginal likelihood is: ', logZ)
    else:
        print('Running in non-thermodynamic mode ')
        timer.sleep(5.5) 
        trace, LLs = mcmc.run(returnLL=True)
        loglike = LLs
    if nchains > 1:
                print('R-hat:')
                print(pints.rhat_all_params(trace))


# These are for pints plotting, and other ppc quantities,
## TODO: R-hat, Combine outputs from both chains for further processing

samples_all_chains = trace[:, burnin:, :]
sample_chain_1 = samples_all_chains[0]

# save pickled traces, and loglikelihoods from all chains(at different temperatures)
root = os.path.abspath('mcmc_results')
if not os.path.exists(root):
    os.makedirs(root)
param_filename = os.path.join(
    root, model_name + '-cell-' + str(cell) + '-mcmc_traces_test.p')# remove the test tag
cPickle.dump(trace, open(param_filename, 'wb'))
likelihood_filename = os.path.join(
    root, model_name + '-cell-' + str(cell) + '-mcmc_lls_test.p')
cPickle.dump(loglike, open(likelihood_filename, 'wb'))

# save csv traces, and loglikelihoods from the chain with t=1
pints.io.save_samples('mcmc_results/%s-chain_test.csv' %
                      (model_name + '-cell-' + str(cell)), *trace)
##################### 
# Throws errors                     
# pints.io.save_samples('mcmc_results/%s-LLs_test.csv' %
#                      (model_name + '-cell-' + str(cell)), untempered_LLs[:,-1].T)
#####################


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
    if (args.niter - args.burnin) <= args.ppc_samples:
        ppc_samples = args.niter - args.burnin 
    else:
        ppc_samples = args.ppc_samples

    for ind in random.sample(range(0, np.size(sample_chain_1, axis=0)), ppc_samples):
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
        plt.plot(time, new_values[i, :], color='lightyellow',
                 lw=0.5, label='inferred current', alpha=0.01)
    plt.plot(time, mean_values, color='SeaGreen',
                 lw=0.5, label='mean inferred current', alpha=0.05)
    plt.xlim(0, 8000)
    # plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(time[-40000:], current[-40000:], '-', color='blue',
             lw=0.5, label='measured current blow-up')
    for i in xrange(np.size(new_values, axis=0)-1):
        plt.plot(time[-40000:], new_values[i, -40000:], color='lightyellow',
                 lw=0.5, label='inferred current', alpha=0.01)
    plt.plot(time[-40000:], mean_values[-40000:], color='SeaGreen',
                 lw=0.5, label='mean inferred current', alpha=0.05)
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

#!/usr/bin/env python2
#
# Fit Kylie's model to Cell 5 data using CMA-ES
#
from __future__ import division, print_function
import models_forward.LogPrior as prior
import models_forward.pintsForwardModel as forwardModel
import models_forward.Rates as Rates
import models_forward.util as util
import os
import sys
import pints
import pints.plot as pplot
import numpy as np
import cPickle
import random
import myokit
import argparse

# Load a hERG model and prior
cmaes_result_files = 'cmaes_results/'
mcmc_result_files = 'mcmc_results/'

# Check input arguments

parser = argparse.ArgumentParser(
    description='Make AP predictions based on the CMAES fit to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N',
                    help='cell number : 1, 2, ..., 5')
args = parser.parse_args()

protocols = ['sine-wave','ap','original-sine'] # Keep sine wave first to get good sigma estimate, and load params properly
indices = range(len(protocols))
num_models = 30

cell = args.cell

likelihood_results = np.zeros((num_models, len(protocols)+1))
for model_num in range(1,num_models+1):

    # Import markov models from the models file, and rate dictionaries.
    model_name = 'model-' + str(model_num)
    root = os.path.abspath('models_myokit')
    myo_model = os.path.join(root, model_name + '.mmt')
    root = os.path.abspath('rate_dictionaries')
    rate_file = os.path.join(root, model_name + '-priors.p')
    rate_dict = cPickle.load(open(rate_file, 'rb'))

    print("LOADING MODEL "+str(model_num))

    for protocol_index in indices:
        protocol_name = protocols[protocol_index]
        print('Looking at Model ', model_num, ' and protocol ', protocol_name, protocol_index)

        #
        # Select data file
        #
        root = os.path.abspath(protocol_name + '-data')
        data_file = os.path.join(root, 'cell-' + str(cell) + '.csv')

        #
        # Load data
        #
        log = myokit.DataLog.load_csv(data_file).npview()
        time = log.time()
        current = log['current']
        voltage = log['voltage']
        del(log)

        #
        # Load protocol
        #
        if protocol_name=='sine-wave':
                protocol_file = os.path.join(root, 'steps.mmt')
                protocol = myokit.load_protocol(protocol_file)
                sw=1
                #
                # Apply capacitance filter based on protocol
                #
                print('Applying capacitance filtering')
                time, voltage, current = forwardModel.capacitance(protocol, 0.1, time, voltage, current)

        elif protocol_name=='original-sine':
                root = os.path.abspath('original-sine-data')
                protocol_file = os.path.join(root, 'steps.mmt')
                protocol = myokit.load_protocol(protocol_file) # Same steps before sine wave
                print('Applying capacitance filtering')
                time, voltage, current = forwardModel.capacitance(protocol, 0.1, time, voltage, current)
        
        if protocol_name!='sine-wave':
                sw=0
                print('Defining the protocol from ', root)
                protocol_file = os.path.join(root, protocol_name + '.csv')
                log = myokit.DataLog.load_csv(protocol_file).npview()
                prot_times = log.time()
                prot_voltages = log['voltage']
                del(log)
                protocol = [prot_times, prot_voltages]

        if model_num==1:
                root = os.path.abspath('predictions/' + protocol_name + '/cell-' + str(cell))
                if not os.path.exists(root):
                        os.makedirs(root)
                np.savetxt(root + '/spike-filtered-data.csv', np.transpose([time, voltage, current]), delimiter=',')

        #
        # Cell-specific parameters
        #
        temperature = forwardModel.temperature(cell)
        lower_conductance = forwardModel.conductance_limit(cell)

        if protocol_name=='sine-wave' or protocol_name=='original-sine':
                #
                # Estimate noise from start of data
                # Kylie uses the first 200ms, where I = 0 + noise
                #
                sigma_noise = np.std(current[:2000], ddof=1)


        #
        # Create forward model
        #
        transform = 0  # we don't need to bother with transforms for a forward run...        
        
        model = forwardModel.ForwardModel(
                protocol, temperature, myo_model, rate_dict,  transform, sine_wave=sw)
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

        # Define parameter set from best ones we have found so far.
        # Only refresh thse on the first sine wave fit protocol
        if protocol_name=='sine-wave':
                parameter_set = np.loadtxt(cmaes_result_files + model_name +
                                '-cell-' + str(cell) + '-cmaes.txt')
                ll_score = log_likelihood(parameter_set)
                print('CMAES model parameters start point: ', parameter_set)
                print('LogLikelihood (proportional to square error): ', ll_score)

                mcmc_best_param_file = mcmc_result_files + model_name +'-cell-' + str(cell) + '-best-parameters.txt'
                if os.path.isfile(mcmc_best_param_file):
                        mcmc_parameter_set = np.loadtxt(mcmc_best_param_file)
                        mcmc_parameter_set = util.transformer(1,mcmc_parameter_set,rate_dict,False)# Transform hard coded to 1
                        mcmc_ll_score = log_likelihood(mcmc_parameter_set)
                        print('MCMC model parameters start point: ', mcmc_parameter_set)
                        print('LogLikelihood (proportional to square error): ', mcmc_ll_score)
                        if (mcmc_ll_score>ll_score):
                                ll_score = mcmc_ll_score
                                parameter_set = mcmc_parameter_set
                                print('Replacing best fit parameters with MCMC max posterior sample')
        else:
                ll_score = log_likelihood(parameter_set)

        # Keep track of the likelihoods for each protocol, and the best and worst for colour bar scalings.
        likelihood_results[model_num-1,0]=model_num
        likelihood_results[model_num-1,protocol_index+1]=ll_score

        root = os.path.abspath('predictions/' + protocol_name + '/cell-' + str(cell))
        if not os.path.exists(root):
                os.makedirs(root)
        
        #print('Running sim with set ', parameter_set)
        sol = model.simulate(parameter_set, time)
        np.savetxt(root + '/model-' + str(model_num) + '.csv', np.transpose([time, sol]), delimiter=',')
        np.savetxt(root + '/for-teun-model-' + str(model_num) + '.csv', np.transpose([time, model.simulated_v, sol, model.simulated_o]), delimiter=',')

np.savetxt('predictions/likelihoods-cell-' + str(cell) + '.csv', likelihood_results, delimiter=',')


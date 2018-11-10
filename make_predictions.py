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
import matplotlib.pyplot as plt


# Load a hERG model and prior
cmaes_result_files = 'cmaes_results/'

# Check input arguments

parser = argparse.ArgumentParser(
    description='Make AP predictions based on the CMAES fit to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N',
                    help='cell number : 1, 2, ..., 5')
parser.add_argument('--plot', type=bool, default=True, metavar='N',
                    help='plot fitted traces')
args = parser.parse_args()

likelihood_results = np.zeros((30, 3))
for model_num in range(1,31):

    # Import markov models from the models file, and rate dictionaries.
    model_name = 'model-'+str(model_num)
    root = os.path.abspath('models_myokit')
    myo_model = os.path.join(root, model_name + '.mmt')
    root = os.path.abspath('rate_dictionaries')
    rate_file = os.path.join(root, model_name + '-priors.p')
    rate_dict = cPickle.load(open(rate_file, 'rb'))

    print("loading  model: "+str(model_num))
    model_name = 'model-'+str(model_num)

    cell = args.cell

    protocols = ['sine-wave','ap'] # Keep sine wave first to get good sigma estimate.
    indices = range(len(protocols))
    for protocol_index in indices:
        protocol_name = protocols[protocol_index]
        print('Looking at Model ', model_num, ' and protocol ', protocol_name, protocol_index)

        #
        # Select data file
        #
        root = os.path.abspath(protocol_name + '-data')
        print(root)
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
                sw=True
        else:
                sw=False
                protocol_file = os.path.join(root, protocol_name + '.csv')
                log = myokit.DataLog.load_csv(protocol_file).npview()
                prot_times = log.time()
                prot_voltages = log['voltage']
                del(log)
                protocol = [prot_times, prot_voltages]


        #
        # Cell-specific parameters
        #
        temperature = forwardModel.temperature(cell)
        lower_conductance = forwardModel.conductance_limit(cell)

        if protocol_name=='sine-wave':
                #
                # Estimate noise from start of data
                # Kylie uses the first 200ms, where I = 0 + noise
                #
                sigma_noise = np.std(current[:2000], ddof=1)

                #
                # Apply capacitance filter based on protocol
                #
                print('Applying capacitance filtering')
                time, voltage, current = forwardModel.capacitance(protocol, 0.1, time, voltage, current)

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

        # Define starting point for mcmc routine
        parameter_set = np.loadtxt(cmaes_result_files + model_name +
                                '-cell-' + str(cell) + '-cmaes.txt')
        print('Model parameters start point: ', parameter_set)

        ll_score = log_likelihood(parameter_set)
        print('LogLikelihood (proportional to square error): ', ll_score)
        likelihood_results[model_num-1,0]=model_num
        likelihood_results[model_num-1,protocol_index+1]=ll_score

        if args.plot:
                voltage_colour = 'black'
                measured_colour = 'red'
                model_colour = 'blue'

                root = os.path.abspath('figures/' + protocol_name)
                if not os.path.exists(root):
                        os.makedirs(root)
                fig_filename = os.path.join(root, model_name + '-cell-' + str(cell) + '-' + protocol_name + '-prediction.eps')

                ap_sol = model.simulate(parameter_set, time)

                if protocol_name=='ap':
                        plt.figure(figsize=(9, 7))
                        plt.subplot(4, 1, 1)
                        plt.plot(time, voltage, color=voltage_colour, label='Voltage', lw=0.5)
                        plt.xlim(0, 8000)
                        plt.xlabel('Time (ms)')
                        plt.ylabel('Voltage (mV)')
                        plt.subplot(4, 1, 2)
                        plt.plot(time, current, '-', color=measured_colour,
                                lw=0.5, label='measured')
                        plt.plot(time, ap_sol, color=model_colour,
                                lw=0.5, label='predicted', alpha=0.1)
                        plt.xlim(0, 8000)
                        plt.subplot(4, 1, 3)
                        plt.plot(time, current, '-', color=measured_colour,
                                lw=0.5, label='measured')
                        plt.plot(time, ap_sol, color=model_colour,
                                lw=0.5, label='predicted', alpha=0.1)
                        plt.xlim(0, 8000)
                        plt.ylim(-0.5, 2)
                        plt.subplot(4, 1, 4)
                        plt.plot(time, current, '-', color=measured_colour,
                                lw=0.5, label='measured')
                        plt.plot(time, ap_sol, color=model_colour,
                                lw=0.5, label='predicted', alpha=0.1)
                        plt.legend(loc='upper right')
                        plt.xlim(4000, 4500)
                        plt.ylim(0, 6) # nA
                        plt.savefig(fig_filename)
                        plt.close()

                elif protocol_name=='sine-wave':
                        plt.figure()
                        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})

                        a0.plot(time, voltage, color=voltage_colour,lw=0.5)
                        a0.set_ylabel('Voltage (mV)')

                        a1.plot(time, current, label='measured', color=measured_colour, lw=0.5)
                        a1.plot(time, model.simulate(parameter_set, time), label='fitted', color=model_colour, lw=0.5)
                        a1.legend(loc='lower right')
                        a1.set_xlabel('Time (ms)')
                        a1.set_ylabel('Current (nA)')
                        plt.savefig(fig_filename)   # save the figure to file
                        plt.close()

np.savetxt('figures/likelihoods.txt', likelihood_results, delimiter=',')
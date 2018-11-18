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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Load a hERG model and prior
cmaes_result_files = 'cmaes_results/'
mcmc_result_files = 'mcmc_results/'

# Check input arguments

parser = argparse.ArgumentParser(
    description='Make AP predictions based on the CMAES fit to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N',
                    help='cell number : 1, 2, ..., 5')
parser.add_argument('--plot', type=bool, default=True, metavar='N',
                    help='plot fitted traces')
args = parser.parse_args()

protocols = ['sine-wave','ap','original-sine'] # Keep sine wave first to get good sigma estimate, and load params properly
indices = range(len(protocols))
num_models = 30

likelihood_results = np.zeros((num_models, len(protocols)+1))
best_likelihood_each_protocol = np.ones((len(protocols)+1))*-np.inf
worst_likelihood_each_protocol = np.ones((len(protocols)+1))*np.inf

# Make a big new plot that will show the fitting and prediction quality for all models through each protocol
if args.plot:
        for fig_num in indices + list(i+len(indices) for i in indices):
                big_fig = plt.figure(fig_num+1,figsize=(12, 7))
                outer_grid = gridspec.GridSpec(11, 1, wspace=0.0, hspace=0.0)

                for i in range(11):
                        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 4,
                        subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)

                        # First add an axis for each voltage plot
                        if i==0:
                                for j in range(4):
                                        ax = plt.subplot(inner_grid[:,j])
                                        big_fig.add_subplot(ax)
                        
                        # All this slightly mad logic sets up Gary's periodic table of models
                        if i==1 or i==3 or i==5 or i==8 or i==10:
                                ax = plt.subplot(inner_grid[:,0])
                                big_fig.add_subplot(ax)
                        
                        if i==5 or i==8 or i==10:
                                ax = plt.subplot(inner_grid[0,1])
                                big_fig.add_subplot(ax)
                                ax = plt.subplot(inner_grid[1,1])
                                big_fig.add_subplot(ax)
                        elif i==3:
                                ax = plt.subplot(inner_grid[:,1])
                                big_fig.add_subplot(ax)

                        if i>=2: 
                                ax = plt.subplot(inner_grid[:,2])
                                big_fig.add_subplot(ax)


                        if i==6:
                                ax = plt.subplot(inner_grid[0,3])
                                big_fig.add_subplot(ax)
                                ax = plt.subplot(inner_grid[1,3])
                                big_fig.add_subplot(ax)
                        elif i>=3:
                                ax = plt.subplot(inner_grid[:,3])
                                big_fig.add_subplot(ax)


for model_num in range(1,num_models+1):

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

        if (ll_score is not np.nan) and (ll_score > best_likelihood_each_protocol[protocol_index]):
                best_likelihood_each_protocol[protocol_index] = ll_score
        if (ll_score is not np.nan) and (ll_score < worst_likelihood_each_protocol[protocol_index]):
                worst_likelihood_each_protocol[protocol_index] = ll_score

        if args.plot:
                voltage_colour = 'black'
                measured_colour = 'red'
                model_colour = 'blue'

                root = os.path.abspath('figures/' + protocol_name)
                if not os.path.exists(root):
                        os.makedirs(root)
                fig_filename = os.path.join(root, model_name + '-cell-' + str(cell) + '-' + protocol_name + '-prediction.eps')

                print('Running sim with set ', parameter_set)
                sol = model.simulate(parameter_set, time)

                big_fig = plt.figure(protocol_index+1)
                model_axes = big_fig.get_axes()
                big_likelihood_fig = plt.figure(protocol_index+1+len(protocols))
                model_likelihood_axes = big_likelihood_fig.get_axes()

                # Only plot the voltage protocol in the Big Figures for first model
                if model_num==1:
                        for i in range(4):
                                model_axes[i].plot(time, voltage, color=voltage_colour, label='Voltage', lw=0.5)
                                model_likelihood_axes[i].plot(time, voltage, color=voltage_colour, label='Voltage', lw=0.5)
                
                error_measure = (current-sol)*np.sign(current)
                #error_measure = uniform_filter1d(error_measure, size=1000)
                max_err = max(error_measure)
                min_err = min(error_measure)
                biggest_err = max([max_err, abs(min_err)])
                #print('Biggest error is: ', biggest_err)
                error_measure /= (0.5*biggest_err) # Scale to be in [-2,2] for colormaps to show strong results
                #print('Error measure rescaled to be in [',min(error_measure),',',max(error_measure),'].')

                cmap = plt.cm.get_cmap('RdBu')
                N = 100 # Number of time points we do windows for
                for i in range(0, int(len(error_measure)/N)):
                        # There are 4 voltage axes then each models' axis in turn, so start at axes[4] for model 1.
                        model_axes[model_num+3].axvspan(time[N*i],time[N*i+N-1], facecolor=cmap(0.5+np.mean(error_measure[N*i:N*i+N-1])), alpha=0.5)
                model_axes[model_num+3].plot([time[0], time[-1]],[0, 0],'k-',lw=0.5)
                model_axes[model_num+3].set_ylim(0, 1)
                

                if protocol_name=='ap':
                        fig = plt.figure(figsize=(9, 7))
                        plt.subplot(4, 1, 1)
                        plt.plot(time, voltage, color=voltage_colour, label='Voltage', lw=0.5)
                        plt.xlim(0, 8000)
                        plt.xlabel('Time (ms)')
                        plt.ylabel('Voltage (mV)')
                        plt.subplot(4, 1, 2)
                        plt.plot(time, current, '-', color=measured_colour,
                                lw=0.5, label='measured')
                        plt.plot(time, sol, color=model_colour,
                                lw=0.5, label='predicted', alpha=0.1)
                        plt.ylabel('Current (nA)')
                        plt.xlim(0, 8000)
                        for i in range(0, int(len(error_measure)/N)):
                                plt.axvspan(time[N*i],time[N*i+N-1], facecolor=cmap(0.5+np.mean(error_measure[N*i:N*i+N-1])), alpha=0.5)

                        plt.subplot(4, 1, 3)
                        plt.plot(time, current, '-', color=measured_colour,
                                lw=0.5, label='measured')
                        plt.plot(time, sol, color=model_colour,
                                lw=0.5, label='predicted', alpha=0.1)
                        plt.ylabel('Current (nA)')
                        plt.xlim(500, 8000)
                        plt.ylim(-0.5, 1.5)
                        for i in range(0, int(len(error_measure)/N)):
                                plt.axvspan(time[N*i],time[N*i+N-1], facecolor=cmap(0.5+np.mean(error_measure[N*i:N*i+N-1])), alpha=0.5)

                        plt.subplot(4, 1, 4)
                        plt.plot(time, current, '-', color=measured_colour,
                                lw=0.5, label='measured')
                        plt.plot(time, sol, color=model_colour,
                                lw=0.5, label='predicted', alpha=0.1)
                        plt.ylabel('Current (nA)')
                        plt.xlabel('Time (ms)')
                        plt.legend(loc='upper right')
                        plt.xlim(4000, 4500)
                        for i in range(0, int(len(error_measure)/N)):
                                plt.axvspan(time[N*i],time[N*i+N-1], facecolor=cmap(0.5+np.mean(error_measure[N*i:N*i+N-1])), alpha=0.5)

                        plt.ylim(0, 6) # nA
                        plt.savefig(fig_filename)
                        plt.close(fig)

                elif protocol_name in ['sine-wave','original-sine']:
                        fig = plt.figure()
                        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})

                        a0.plot(time, voltage, color=voltage_colour,lw=0.5)
                        a0.set_ylabel('Voltage (mV)')
                        a1.plot(time, current, label='measured', color=measured_colour, lw=0.5)

                        # Output these as they have capacitive spikes removed.
                        if model_num==1 and protocol_name is 'original-sine':
                                np.savetxt('figures/original-sine/original-sine-data.csv', np.transpose([time, current]), delimiter=',')
                        if protocol_name=='sine-wave':
                                label_text = 'fitted'
                        else:
                                label_text = 'predicted'
                        a1.plot(time, sol, label=label_text, color=model_colour, lw=0.5)
                        if protocol_name is 'original-sine':
                                np.savetxt('figures/original-sine/model-' + str(model_num) + '-original-sine.csv', np.transpose([time, sol]), delimiter=',')
                        a1.legend(loc='lower right')
                        a1.set_xlabel('Time (ms)')
                        a1.set_ylabel('Current (nA)')
                        plt.savefig(fig_filename)   # save the figure to file
                        plt.close(fig)

np.savetxt('figures/likelihoods.txt', likelihood_results, delimiter=',')

if args.plot:
        for protocol_index in indices:
                fig = plt.figure(protocol_index+1)

                all_axes = fig.get_axes()
                #show only the outside spines
                for ax in all_axes:
                        ax.tick_params(labelbottom=False)
                        # for sp in ax.spines.values():
                        #         sp.set_visible(False)
                        # if ax.is_first_row():
                        #         ax.spines['top'].set_visible(True)
                        # if ax.is_last_row():
                        #         ax.spines['bottom'].set_visible(True)
                        # if ax.is_first_col():
                        #         ax.spines['left'].set_visible(True)
                        # if ax.is_last_col():
                        #         ax.spines['right'].set_visible(True)

                plt.subplots_adjust(bottom=0.075, left=0.05, right=0.90, top=0.925)
                cax = plt.axes([0.92, 0.075, 0.02, 0.775])
         
                norm = mpl.colors.Normalize(vmin=-1, vmax=1)
                cmap1_reversed = plt.cm.get_cmap('RdBu_r')
                cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap1_reversed, norm=norm, orientation='vertical')

                ticks = [-1,0,1]
                cb1.set_ticks(ticks)
                cb1.set_ticklabels(['Too low','No error','Too high'])

                cb1.set_label('Error size (normalised for each model)')    

                plt.savefig('figures/' + protocols[protocol_index] + '_all_model_errors_cell_' + str(cell) + '.eps')   # save the figure to file
                plt.close(fig)

                fig = plt.figure(protocol_index+1+len(indices))
                model_likelihood_axes = fig.get_axes()
                for model_num in range(1,num_models+1):
                        ll_score = likelihood_results[model_num-1,protocol_index+1]
                        cmap2 = plt.cm.get_cmap('viridis')
                        scaled_score = (np.log10(-ll_score)-np.log10(-worst_likelihood_each_protocol[protocol_index]))/(np.log10(-best_likelihood_each_protocol[protocol_index])-np.log10(-worst_likelihood_each_protocol[protocol_index]))
                        print('Scaled score for model ', model_num, ' protocol ', protocols[protocol_index], ' is ', scaled_score)
                        model_likelihood_axes[model_num+3].axvspan(0,1, facecolor=cmap2(scaled_score), alpha=0.5)
                        model_likelihood_axes[model_num+3].plot([0, 1],[0, 0],'k-',lw=0.5)
                        model_likelihood_axes[model_num+3].set_ylim(0, 1)
                
                for ax in model_likelihood_axes:
                        ax.tick_params(labelbottom=False)

                # create an axes on the right side of the gridspec
                plt.subplots_adjust(bottom=0.075, left=0.05, right=0.90, top=0.925)
                cax2 = plt.axes([0.92, 0.075, 0.02, 0.775])
                
                cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap2, orientation='vertical')
                cb2.set_ticks([0,1])
                cb2.set_ticklabels(['Worst','Best'])
                cb2.set_label('Goodness of fit')

                plt.savefig('figures/' + protocols[protocol_index] + '_all_model_likelihoods_cell_' + str(cell) + '.eps')   # save the figure to file
                plt.close(fig)

#!/usr/bin/env python2
#
# Fit Kylie's model to Cell 5 data using CMA-ES
#
from __future__ import division, print_function
import os
import sys
import pints
import pints.plot as pplot
import numpy as np
import cPickle
import random
import math
import myokit
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Check input arguments

parser = argparse.ArgumentParser(
    description='Make AP predictions based on the CMAES fit to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N',
                    help='cell number : 1, 2, ..., 5')
args = parser.parse_args()

protocols = ['sine-wave','ap','original-sine'] # Keep sine wave first to get good sigma estimate, and load params properly
indices = range(len(protocols))
num_models = 30

err_map = 'PRGn'
cmap = plt.cm.get_cmap(err_map)
cmap1_reversed = plt.cm.get_cmap(err_map + '_r')

cell = args.cell

likelihood_results = np.loadtxt('predictions/likelihoods-cell-' + str(cell) + '.csv', delimiter=',')

# Make a big new plot that will show the fitting and prediction quality for all models through each protocol
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
    print("PLOTTING FOR MODEL "+str(model_num))
    model_name = 'model-'+str(model_num)

    for protocol_index in indices:
        protocol_name = protocols[protocol_index]
        print('Protocol ', protocol_index, ': ', protocol_name, )

        numpy_load = np.loadtxt('predictions/' + protocol_name + '/cell-' + str(cell) + '/spike-filtered-data.csv', delimiter=',')

        time = numpy_load[:,0]
        voltage = numpy_load[:,1]
        current = numpy_load[:,2]

        voltage_colour = 'black'
        measured_colour = 'red'
        model_colour = 'blue'

        root = os.path.abspath('figures/predictions/' + protocol_name)
        if not os.path.exists(root):
                os.makedirs(root)
        fig_filename = os.path.join(root, model_name + '-cell-' + str(cell) + '-' + protocol_name + '-prediction.eps')

        #print('Running sim with set ', parameter_set)
        numpy_load = np.loadtxt('predictions/' + protocol_name + '/cell-' + str(cell) + '/model-' + str(model_num) + '.csv', delimiter=',')
        sol = numpy_load[:,1]
        #sol = model.simulate(parameter_set, time)


        big_fig = plt.figure(protocol_index+1)
        model_axes = big_fig.get_axes()
        big_likelihood_fig = plt.figure(protocol_index+1+len(protocols))
        model_likelihood_axes = big_likelihood_fig.get_axes()

        # Only plot the voltage protocol in the Big Figures for first model
        if model_num==1:
                for i in range(4):
                        model_axes[i].plot(time, voltage, color=voltage_colour, label='Voltage', lw=0.5)
                        model_likelihood_axes[i].plot(time, voltage, color=voltage_colour, label='Voltage', lw=0.5)
        
        # Now load up the predictions from 'make_predictions.py'



        N = 100 # Number of time points in each windows
        error_measure = (current-sol)*np.sign(current)
        num_windows = int(len(error_measure)/N)
        windowed_error = np.zeros(num_windows)
        for i in range(0, num_windows):
                windowed_error[i] = np.mean(error_measure[N*i:N*i+N-1]) # in nA
        #max_err = max(windowed_error)
        #min_err = min(windowed_error)
        #biggest_err = max([max_err, abs(min_err)])
        #windowed_error /= biggest_err # Scale to be in [-1,1] for colormaps to show strong results
        # (it should be [-0.5,0.5] to give a min of zero and max of one, but this shows peaks but not most of trace...)
        #print('Error measure rescaled to be in [',min(error_measure),',',max(error_measure),'].')

        
        
        for i in range(0, num_windows):
                # There are 4 voltage axes then each models' axis in turn, so start at axes[4] for model 1.
                model_axes[model_num+3].axvspan(time[N*i],time[N*i+N-1], facecolor=cmap(0.5+2*windowed_error[i]), alpha=0.5)
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
                for i in range(0, num_windows):
                        plt.axvspan(time[N*i],time[N*i+N-1], facecolor=cmap(0.5+2*windowed_error[i]), alpha=0.5)

                plt.subplot(4, 1, 3)
                plt.plot(time, current, '-', color=measured_colour,
                        lw=0.5, label='measured')
                plt.plot(time, sol, color=model_colour,
                        lw=0.5, label='predicted', alpha=0.1)
                plt.ylabel('Current (nA)')
                plt.xlim(500, 8000)
                plt.ylim(-0.5, 1.5)
                for i in range(0, num_windows):
                        plt.axvspan(time[N*i],time[N*i+N-1], facecolor=cmap(0.5+2*windowed_error[i]), alpha=0.5)

                plt.subplot(4, 1, 4)
                plt.plot(time, current, '-', color=measured_colour,
                        lw=0.5, label='measured')
                plt.plot(time, sol, color=model_colour,
                        lw=0.5, label='predicted', alpha=0.1)
                plt.ylabel('Current (nA)')
                plt.xlabel('Time (ms)')
                plt.legend(loc='upper right')
                plt.xlim(4000, 4500)
                for i in range(0, num_windows):
                        plt.axvspan(time[N*i],time[N*i+N-1], facecolor=cmap(0.5+2*windowed_error[i]), alpha=0.5)

                plt.ylim(0, 6) # nA


                # Squeeze in a colorbar axis
                plt.subplots_adjust(bottom=0.065, left=0.08, right=0.85, top=0.98)
                cax = plt.axes([0.88, 0.065, 0.02, 0.68])
                norm = mpl.colors.Normalize(vmin=-1, vmax=1)
                cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap1_reversed, norm=norm, orientation='vertical')
                ticks = [-1,0,1]
                cb1.set_ticks(ticks)
                cb1.set_ticklabels(['-0.25nA','0','+0.25nA'])
                cb1.set_label('Error', labelpad=-20)    

                plt.savefig(fig_filename)
                plt.close(fig)
                del fig

        elif protocol_name in ['sine-wave','original-sine']:
                fig = plt.figure()
                f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})

                a0.plot(time, voltage, color=voltage_colour,lw=0.5)
                a0.set_ylabel('Voltage (mV)')
                a1.plot(time, current, label='measured', color=measured_colour, lw=0.5)

                if protocol_name=='sine-wave':
                        label_text = 'fitted'
                else:
                        label_text = 'predicted'
                a1.plot(time, sol, label=label_text, color=model_colour, lw=0.5)
                for i in range(0, num_windows):
                        plt.axvspan(time[N*i],time[N*i+N-1], facecolor=cmap(0.5+2*windowed_error[i]), alpha=0.5)

                a1.legend(loc='lower right')
                a1.set_xlabel('Time (ms)')
                a1.set_ylabel('Current (nA)')

                # Squeeze in a colorbar axis
                plt.subplots_adjust(bottom=0.1, left=0.12, right=0.85, top=0.95)
                cax = plt.axes([0.86, 0.1, 0.02, 0.515])
                norm = mpl.colors.Normalize(vmin=-1, vmax=1)
                cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap1_reversed, norm=norm, orientation='vertical')
                ticks = [-1,0,1]
                cb1.set_ticks(ticks)
                cb1.set_ticklabels(['-0.25nA','0','+0.25nA'])
                cb1.set_label('Error', labelpad=-20)    

                plt.savefig(fig_filename)   # save the figure to file
                plt.close(fig)
                del fig

for protocol_index in indices:
        fig = plt.figure(protocol_index+1)

        all_axes = fig.get_axes()
        #show only the outside spines
        for ax in all_axes:
                ax.tick_params(labelbottom=False)
                ax.tick_params(labelleft=False)
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

        # Squeeze in a colorbar axis
        plt.subplots_adjust(bottom=0.075, left=0.05, right=0.90, top=0.925)
        cax = plt.axes([0.92, 0.075, 0.02, 0.775])
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap1_reversed, norm=norm, orientation='vertical')
        ticks = [-1,0,1]
        cb1.set_ticks(ticks)
        cb1.set_ticklabels(['-0.25nA','0','+0.25nA'])
        cb1.set_label('Error',labelpad=-20)    

        # Add a label saying which model is which to both big plots
        for i in range(1,num_models+1):
                all_axes[i+3].text(0,0.9,str(i),verticalalignment='top', horizontalalignment='left',fontsize=10)

        plt.savefig('figures/predictions/' + protocols[protocol_index] + '_all_model_errors_cell_' + str(cell) + '.eps')   # save the figure to file
        plt.close(fig)

        fig = plt.figure(protocol_index+1+len(indices))
        model_likelihood_axes = fig.get_axes()
        sorted_scores = np.sort(likelihood_results[:,protocol_index+1])
        sorted_scores = [i for i in sorted_scores if not math.isnan(i)]
        print('Sorted first = ', sorted_scores[0], ' last = ', sorted_scores[-1])
        for model_num in range(1,num_models+1):
                ll_score = likelihood_results[model_num-1,protocol_index+1]
                cmap2 = plt.cm.get_cmap('viridis')
                scaled_score = (np.log10(-ll_score)-np.log10(-sorted_scores[1]))/(np.log10(-sorted_scores[-1])-np.log10(-sorted_scores[1]))
                print('Scaled score for model ', model_num, ' protocol ', protocols[protocol_index], ' is ', scaled_score)
                model_likelihood_axes[model_num+3].axvspan(0,1, facecolor=cmap2(scaled_score), alpha=0.5)
                model_likelihood_axes[model_num+3].plot([0, 1],[0, 0],'k-',lw=0.5)
                model_likelihood_axes[model_num+3].set_ylim(0, 1)
                txt = model_likelihood_axes[model_num+3].text(0.1,0.9,str(model_num),verticalalignment='top', horizontalalignment='left',fontsize=10)
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
                plt.draw()
        
        for ax in model_likelihood_axes:
                ax.tick_params(labelbottom=False)
                ax.tick_params(labelleft=False)

        # create an axes on the right side of the gridspec
        plt.subplots_adjust(bottom=0.075, left=0.05, right=0.90, top=0.925)
        cax2 = plt.axes([0.92, 0.075, 0.02, 0.775])
        
        cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap2, orientation='vertical')
        cb2.set_ticks([0,1])
        cb2.set_ticklabels(['Worst','Best'])
        cb2.set_label('Goodness of fit', labelpad=-10)

        plt.savefig('figures/predictions/' + protocols[protocol_index] + '_all_model_likelihoods_cell_' + str(cell) + '.eps')   # save the figure to file
        plt.close(fig)

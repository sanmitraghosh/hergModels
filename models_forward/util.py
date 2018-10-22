#!/usr/bin/env python2
#
# Pints ForwardModel that runs simulations with Kylie's model.
# Sine waves optional
#
from __future__ import division
from __future__ import print_function
import os
import pints
import numpy as np
import myokit
import myokit.pacing as pacing


def erev(temperature):
    T = 273.15 + temperature
    F = 96485.0
    R = 8314.0
    K_i = 130.0
    k_o = 4.0
    return ((R*T)/F) * np.log(k_o/K_i)

class Pr1Error(pints.ErrorMeasure):
    def __init__(self, problem):
        self._problem = problem

    def dimension(self):
        return self._model.dimension()

    def __call__(self, x):
        # Run simulation
        values = self._problem.evaluate()

        # Extract points of interest
        raise NotImplementedError


def conductance_limit(cell):
    """
    Returns a lower limit for the conductance of the cell with the
    given integer index ``cell``.
    """
    #
    # Guesses for lower conductance
    #
    lower_conductances = {
        5: 0.0612,  # 16713110
    }
    #if strcmp(exp_ref,'16708118')==1    lower_conductance = 0.0170;
    #if strcmp(exp_ref,'16704047')==1    lower_conductance = 0.0434;
    #if strcmp(exp_ref,'16704007')==1    lower_conductance = 0.0886;
    #if strcmp(exp_ref,'16707014')==1    lower_conductance = 0.0203;
    #if strcmp(exp_ref,'16708060')==1    lower_conductance = 0.0305;
    #if strcmp(exp_ref,'16708016')==1    lower_conductance = 0.0417;
    #if strcmp(exp_ref,'16713003')==1    lower_conductance = 0.0478;
    #if strcmp(exp_ref,'16715049')==1    lower_conductance = 0.0255;
    #if strcmp(exp_ref,'average')==1     lower_conductance = 0.0410;

    return lower_conductances[cell]


def temperature(cell):
    """
    Returns the temperature (in degrees Celsius) for the given integer index
    ``cell``.
    """
    temperatures = {
        5 : 21.4,   # 16713110
    }
    #if strcmp(exp_ref,'16708016')==1    temperature = 21.8;
    #if strcmp(exp_ref,'16708060')==1    temperature = 21.7;
    #if strcmp(exp_ref,'16704047')==1    temperature = 21.6;
    #if strcmp(exp_ref,'16704007')==1    temperature = 21.2;
    #if strcmp(exp_ref,'16713003')==1    temperature = 21.3;
    #if strcmp(exp_ref,'16715049')==1    temperature = 21.4;
    #if strcmp(exp_ref,'16707014')==1    temperature = 21.4;
    #if strcmp(exp_ref,'16708118')==1    temperature = 21.7;

    return temperatures[cell]


def capacitance(protocol, dt, *signals):
    """
    Create and apply a capacitance filter
    """
    cap_duration = 5    # Same as Kylie
    fcap = np.ones(len(signals[0]), dtype=int)
    steps = [step for step in protocol]
    for step in steps[1:]:
        i1 = int(step.start() / dt)
        i2 = i1 + int(cap_duration / dt)
        fcap[i1:i2] = 0
    fcap = fcap > 0

    # Apply filter
    return [x[fcap] for x in signals]


def fold_plot(protocol, time, voltage, currents, labels=None):
    """
    Create a folded plot of a given protocol
    """
    # Points to split signal at
    splits = {
        'pr1-activation-kinetics-1' : [
            # Numbers 2, 3, and 4 are a bit off for some reason
            0, 51770, 103519, 155269, 207019, 258770, 310520,
        ],
        'pr2-activation-kinetics-2' : [
            0, 51770, 103519, 155269, 207019, 258770, 310520,
        ],
        'pr3-steady-activation' : [
            0, 82280, 164609, 246889, 329169, 411449, 493729, 576010,
        ],
        'pr4-inactivation' : [
            0, 28657, 57363, 86019, 114674, 143331, 171987, 200642, 229299,
            257955, 286611, 315267, 343922, 372578, 401235, 429891, 458546,
        ],
        'pr5-deactivation' : [
            0, 102974, 205897, 308822, 411746, 514670, 617593, 720518, 823442,
            926366,
        ],
    }

    # Find split points
    try:
        split = splits[protocol]
        split = zip(split[:-1], split[1:])
    except KeyError:
        # Try to auto-detect!
        repeats = {
            'pr1-activation-kinetics-1' : 6,
            'pr2-activation-kinetics-2' : 6,
            'pr3-steady-activation' : 7,
            'pr4-inactivation' : 16,
            'pr5-deactivation' : 9,
        }
        period = (time[-1] + 0.1) / repeats[protocol]
        print('Period: ' + str(period))

        print('Guessed splits:')
        splits = [int(i * period * 10) for i in range(repeats)]
        splits.append(len(time))
        splits = zip(splits[:-1], splits[1:])
        first = np.where(voltage != voltage[0])[0][0]
        for lower, upper in splits:
            step = np.where(
                np.abs(voltage[lower:upper] - voltage[lower]) > 10)[0][0]
            print(lower + step - first)
        print(len(time))

        # Will need customisation, so halt!
        raise

    # Define zoom points
    zoom = {
        #'pr1-activation-kinetics-1' : (),
        #'pr2-activation-kinetics-2' : (),
        'pr3-steady-activation' : ((500, 6500), (-0.1, 1.75)),
        'pr4-inactivation' : ((1180, 1500), (-3.2, 6.5)),
        'pr5-deactivation' : ((2300, 8000), (-4, 2)),
    }

    # Load matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec

    # Create colormap
    #cmap = matplotlib.cm.get_cmap('Dark2')
    cmap = matplotlib.cm.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(0, len(split) - 1)

    # Create plot
    plt.figure()
    matplotlib.gridspec.GridSpec(4, 1)

    plt.subplot2grid((4, 1), (0, 0))
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')
    for i, bounds in enumerate(split):
        lower, upper = bounds
        plt.plot(time[lower:upper] - time[lower], voltage[lower:upper],
                 color=cmap(norm(i)))

    plt.subplot2grid((4, 1), (1, 0), rowspan=3)
    plt.xlabel('Time (ms)')
    plt.ylabel('I (nA)')
    styles = ['-', '--', ':']
    for i, bounds in enumerate(split):
        lower, upper = bounds
        for j, current in enumerate(currents):
            label = labels[j] if labels and i == 0 else None
            plt.plot(time[lower:upper] - time[lower], current[lower:upper],
                     styles[j], color=cmap(norm(i)), label=label)
    if labels:
        plt.legend()

    try:
        xlim, ylim = zoom[protocol]
        plt.xlim(*xlim)
        plt.ylim(*ylim)
    except KeyError:
        pass

def transformer(transform, parameters, rate_dict, logexp = True):
    txd_params = np.copy(parameters)
    if transform == 'loglog':

        for names, rate in rate_dict.iteritems():
            if rate[2] == 'positive' or rate[2] == 'negative':
                if logexp:
                    txd_params[rate[0]] = np.log(txd_params[rate[0]])
                    txd_params[rate[1]] = np.log(txd_params[rate[1]])
                else:
                    txd_params[rate[0]] = np.exp(txd_params[rate[0]])
                    txd_params[rate[1]] = np.exp(txd_params[rate[1]])
            if rate[2] == 'vol_ind':
                if logexp:
                    txd_params[rate[0]] = np.log(txd_params[rate[0]])
                    
                else:
                    txd_params[rate[0]] = np.exp(txd_params[rate[0]])
        if logexp:
            txd_params[-1] = np.log(txd_params[-1])
        else:
            txd_params[-1] = np.exp(txd_params[-1])
    if transform == 'loglinear':

        for names, rate in rate_dict.iteritems():

            if logexp:
                txd_params[rate[0]] = np.log(txd_params[rate[0]])
            else:
                txd_params[rate[0]] = np.exp(txd_params[rate[0]])
        
    
    return txd_params
            
                    
                        

    
    	













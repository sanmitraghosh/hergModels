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
        1: 0.0478,  # 16713003
        2: 0.0255,  # 16715049
        3: 0.0417,  # 16708016
        4: 0.0305,  # 16708060
        5: 0.0612,  # 16713110
        6: 0.0170,  # 16708118
        7: 0.0886,  # 16704007
        8: 0.0434,  # 16704047
        9: 0.0203,  # 16707014
    }
    return lower_conductances[cell]


def temperature(cell):
    """
    Returns the temperature (in degrees Celsius) for the given integer index
    ``cell``.
    """
    temperatures = {
        1: 21.3,    # 16713003
        2: 21.4,    # 16715049
        3: 21.8,    # 16708016
        4: 21.7,    # 16708060
        5: 21.4,    # 16713110
        6: 21.7,    # 16708118
        7: 21.2,    # 16704007
        8: 21.6,    # 16704047
        9: 21.4,    # 16707014
    }

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
        'pr1-activation-kinetics-1': [
            # Numbers 2, 3, and 4 are a bit off for some reason
            0, 51770, 103519, 155269, 207019, 258770, 310520,
        ],
        'pr2-activation-kinetics-2': [
            0, 51770, 103519, 155269, 207019, 258770, 310520,
        ],
        'pr3-steady-activation': [
            0, 82280, 164609, 246889, 329169, 411449, 493729, 576010,
        ],
        'pr4-inactivation': [
            0, 28657, 57363, 86019, 114674, 143331, 171987, 200642, 229299,
            257955, 286611, 315267, 343922, 372578, 401235, 429891, 458546,
        ],
        'pr5-deactivation': [
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
            'pr1-activation-kinetics-1': 6,
            'pr2-activation-kinetics-2': 6,
            'pr3-steady-activation': 7,
            'pr4-inactivation': 16,
            'pr5-deactivation': 9,
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
        # 'pr1-activation-kinetics-1' : (),
        # 'pr2-activation-kinetics-2' : (),
        'pr3-steady-activation': ((500, 6500), (-0.1, 1.75)),
        'pr4-inactivation': ((1180, 1500), (-3.2, 6.5)),
        'pr5-deactivation': ((2300, 8000), (-4, 2)),
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

# Last argument is whether we are going True = from model to optimiser, or False from optimiser to model.


def transformer(transform, parameters, rate_dict, logexp=True):
    txd_params = np.copy(parameters)
    # First the no transform case.
    if transform == 0:
        return txd_params
    # Then the Log A Linear B, linear conductance case
    elif transform == 1:
        for names, rate in rate_dict.iteritems():
            if logexp:
                txd_params[rate[0]] = np.log(txd_params[rate[0]])
            else:
                txd_params[rate[0]] = np.exp(txd_params[rate[0]])

    # Now the Log everything transform case
    elif transform == 2:
        if logexp:
            txd_params = np.log(txd_params)
        else:
            txd_params = np.exp(txd_params)
    else:
        Exception(
            'Unrecognised transform type, should be 0=Nothing, 1=LogLinear or 2=LogLog')

    return txd_params

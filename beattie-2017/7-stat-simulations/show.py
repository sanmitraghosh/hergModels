#!/usr/bin/env python2
#
# Show traditional protocol data plus simulation
#
from __future__ import division
from __future__ import print_function
import os
import sys
import pints
import numpy as np
import myokit

# Load beattie model and prior
sys.path.append(os.path.abspath('..'))
import beattie


#
# Select cell
#
cell = 5


#
# Select protocol
#
protocol = 'pr1-activation-kinetics-1'
protocol = 'pr2-activation-kinetics-2'
protocol = 'pr3-steady-activation'
#protocol = 'pr4-inactivation'
#protocol = 'pr5-deactivation'


#
# Select data file
#
print('Loading data for ' + protocol + ', cell ' + str(cell))
root = os.path.realpath(os.path.join('..', '..'))
data_file = os.path.join(
    root, 'traditional-data', protocol + '-cell-' + str(cell) + '.csv')


#
# Load protocol
#
print('Loading myokit protocol file')
protocol_file = os.path.join(root, 'traditional-data', protocol + '.mmt')
myokit_protocol = myokit.load_protocol(protocol_file)


#
# Cell-specific parameters
#
temperature = beattie.temperature(cell)
lower_conductance = beattie.conductance_limit(cell)


#
# Load data
#
print('Loading data')
log = myokit.DataLog.load_csv(data_file).npview()
time = log.time()
current = log['current']
voltage = log['voltage']
del(log)


#
# Estimate noise from start of data
#
sigma_noise = np.std(current[:2000], ddof=1)


#
# Apply capacitance filter based on protocol
#
print('Applying capacitance filtering')
time, voltage, current = beattie.capacitance(
    myokit_protocol, 0.1, time, voltage, current)


#
# Create ForwardModel
#
model = beattie.BeattieModel(myokit_protocol, temperature, sine_wave=False)


#
# Define problem
#
problem = pints.SingleSeriesProblem(model, time, current)


#
# Define a log-posterior
#
log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
log_prior = beattie.BeattieLogPrior(lower_conductance)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)


#
# Load earlier result
#
print('Loading model parameters')
filename = '../1-sine-simulations/kylies-solution.txt'
with open(filename, 'r') as f:
    obtained_parameters = [float(x) for x in f.readlines()]
obtained_log_posterior = log_posterior(obtained_parameters)


#
# Show result
#

# Simulate
print('Simulating')
simulated = model.simulate(obtained_parameters, time)

# Plot
import matplotlib.pyplot as plt

#
# Full comparison: protocol, transitions, current, current at transitions
# This plot works best with the capacitance filter switched off
#
'''
plt.figure()
plt.subplot(4, 1, 1)
plt.xlabel('Time (ms)')
plt.ylabel('V (mV)')
plt.plot(time, voltage, '-', alpha=0.75, label='measured')
plt.plot(time, model.simulated_v, '-', alpha=0.75, label='simulated')
plt.legend()
plt.subplot(4, 1, 2)
plt.xlabel('Time (ms)')
plt.ylabel('V (mV)')
plt.plot(time, voltage - model.simulated_v, '-', label='real - simulated')
plt.legend()
plt.subplot(4, 1, 3)
plt.xlabel('Time (ms)')
plt.ylabel('I (nA)')
plt.plot(time, current, '-', alpha=0.75, label='measured')
plt.plot(time, simulated, '-', alpha=0.75, label='simulated')
plt.legend()
plt.subplot(4, 1, 4)
plt.xlabel('Time (ms)')
plt.ylabel('I (nA)')
plt.plot(time, current - simulated, '-', label='real - simulated')
plt.legend()
'''

if protocol == 'pr1-activation-kinetics-1':
    steps = [
        (57483, 50),
        (109232, 250),
        (160982, 950),
        (212732, 2950),
        (264483, 9950),
    ]

    # Plot voltage protocol and current
    # Highlight critical step used for summary statistic
    plt.figure()
    plt.suptitle('Pr1: Activation kinetics 1 (cell ' + str(cell) + ')')
    plt.subplot(2, 1, 1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.plot(time, voltage, label='measured')
    label = 'measured, critical step'
    for i, j in steps:
        plt.plot(time[i:i+j], voltage[i:i+j], 'x-', color='tab:orange',
                 label=label)
        label = None
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (nA)')
    plt.plot(time, current, ':', label='measured')
    label = 'measured, critical step'
    for i, j in steps:
        plt.plot(time[i:i+j], current[i:i+j], '-', color='tab:orange',
                 label=label)
        label = None

    # Isolate summary statistics, show in figure which points are used
    print('Gathering summary statistics:')
    summary = []
    label1 = 'measured, used points'
    label2 = 'measured, mean of used points'
    for i, j in steps:
        x2 = i + j
        x1 = x2 - 10
        mean = np.mean(current[x1:x2])
        summary.append(mean)

        plt.plot(time[x1:x2], current[x1:x2], 'x-', color='k', label=label1)
        plt.plot(time[x1:x2], mean * np.ones(time[x1:x2].shape), color='g',
            label=label2)
        label1 = label2 = None
    plt.legend()
    print(summary)

    # Repeat last figure, for first and last step
    plt.figure(figsize=(9, 4))
    plt.suptitle('Pr1: Activation kinetics 1 (cell ' + str(cell) + ')')
    plt.subplot(1, 2, 1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (nA)')
    plt.plot(time, current, ':', label='measured')
    plt.subplot(1, 2, 2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (nA)')
    plt.plot(time, current, ':', label='measured')
    label = 'measured, critical step'
    for i, j in steps:
        plt.subplot(1, 2, 1)
        plt.plot(time[i:i+j], current[i:i+j], '-', color='tab:orange',
                 label=label)
        plt.subplot(1, 2, 2)
        plt.plot(time[i:i+j], current[i:i+j], '-', color='tab:orange',
                 label=label)
        label = None
    label = 'measured, used points'
    for i, j in steps:
        x2 = i + j
        x1 = x2 - 10
        plt.subplot(1, 2, 1)
        plt.plot(time[x1:x2], current[x1:x2], 'x-', color='k', label=label)
        plt.subplot(1, 2, 2)
        plt.plot(time[x1:x2], current[x1:x2], 'x-', color='k', label=label)
        label = None
    plt.subplot(1, 2, 1)
    plt.legend()
    plt.xlim(5778, 5798)
    plt.ylim(-0.05, 0.05)
    plt.subplot(1, 2, 2)
    plt.legend()
    plt.xlim(26500, 28000)
    plt.ylim(-0.025, 0.35)

    # Plot summary statistic
    durations = [10, 30, 100, 300, 1000]
    plt.figure()
    plt.suptitle('Summary statistic for Pr1: Activation kinetics 1 (cell '
                 + str(cell) + ')')
    plt.xlabel('Variable step duration (ms)')
    plt.ylabel('Final current during step (nA)')
    plt.plot(durations, summary, 'x-')

elif protocol == 'pr2-activation-kinetics-2':

    steps = [
        (57483, 50),
        (109232, 250),
        (160982, 950),
        (212732, 2950),
        (264483, 9950),
    ]

    # Plot voltage protocol and current
    # Highlight critical step used for summary statistic
    plt.figure()
    plt.suptitle('Pr2: Activation kinetics 2 (cell ' + str(cell) + ')')
    plt.subplot(2, 1, 1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.plot(time, voltage, label='measured')
    label = 'measured, critical step'
    for i, j in steps:
        plt.plot(time[i:i+j], voltage[i:i+j], 'x-', color='tab:orange',
                 label=label)
        label = None
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (nA)')
    plt.plot(time, current, ':', label='measured')
    label = 'measured, critical step'
    for i, j in steps:
        plt.plot(time[i:i+j], current[i:i+j], '-', color='tab:orange',
                 label=label)
        label = None

    # Isolate summary statistics, show in figure which points are used
    print('Gathering summary statistics:')
    summary = []
    label1 = 'measured, used points'
    label2 = 'measured, mean of used points'
    for i, j in steps:
        x2 = i + j
        x1 = x2 - 10
        mean = np.mean(current[x1:x2])
        summary.append(mean)

        plt.plot(time[x1:x2], current[x1:x2], 'x-', color='k', label=label1)
        plt.plot(time[x1:x2], mean * np.ones(time[x1:x2].shape), color='g',
            label=label2)
        label1 = label2 = None
    plt.legend()
    print(summary)

    # Repeat last figure, for first and last step
    plt.figure(figsize=(9, 4))
    plt.suptitle('Pr2: Activation kinetics 2 (cell ' + str(cell) + ')')
    plt.subplot(1, 2, 1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (nA)')
    plt.plot(time, current, ':', label='measured')
    plt.subplot(1, 2, 2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (nA)')
    plt.plot(time, current, ':', label='measured')
    label = 'measured, critical step'
    for i, j in steps:
        plt.subplot(1, 2, 1)
        plt.plot(time[i:i+j], current[i:i+j], '-', color='tab:orange',
                 label=label)
        plt.subplot(1, 2, 2)
        plt.plot(time[i:i+j], current[i:i+j], '-', color='tab:orange',
                 label=label)
        label = None
    label = 'measured, used points'
    for i, j in steps:
        x2 = i + j
        x1 = x2 - 10
        plt.subplot(1, 2, 1)
        plt.plot(time[x1:x2], current[x1:x2], 'x-', color='k', label=label)
        plt.subplot(1, 2, 2)
        plt.plot(time[x1:x2], current[x1:x2], 'x-', color='k', label=label)
        label = None
    plt.subplot(1, 2, 1)
    plt.legend()
    plt.xlim(5778, 5798)
    plt.ylim(-0.05, 0.05)
    plt.subplot(1, 2, 2)
    plt.legend()
    plt.xlim(26500, 28000)
    plt.ylim(-0.025, 0.35)

    # Plot summary statistic
    durations = [10, 30, 100, 300, 1000]
    plt.figure()
    plt.suptitle('Summary statistic for Pr2: Activation kinetics 2 (cell '
                 + str(cell) + ')')
    plt.xlabel('Variable step duration (ms)')
    plt.ylabel('Final current during step (nA)')
    plt.plot(durations, summary, 'x-')


elif protocol == 'pr3-steady-activation':

    steps = [
        (56141, 9950),
        (138441, 9950),
        (220750, 9950),
        (303030, 9950),
        (385310, 9950),
        (467590, 9950),
        (549871, 9950),
    ]

    #
    # Figure: Protocol and current
    #
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, voltage)
    for i, j in steps:
        plt.plot(time[i:i+j], voltage[i:i+j], 'x-', color='tab:orange')

    plt.subplot(2, 1, 2)
    plt.plot(time, current, ':')
    for i, j in steps:
        plt.plot(time[i:i+j], current[i:i+j], '-', color='tab:orange')

    #
    # Figure: Critical step and data used for analysis
    #
    plt.figure(figsize=(9, 5))
    n = len(steps)
    m = n // 2 + 1
    summary = []
    for k, step in enumerate(steps):
        i, j = step
        plt.subplot(m, 2, 1 + (2 * k) % n)
        print(m, 2, 1 + (2 * k) % n)

        # Plot current
        plt.plot(time, current, ':')
        plt.plot(time[i:i+j], current[i:i+j], '-', color='tab:orange')

        # Get peak
        c = current[i:i+j]
        z = np.argmax(c)
        zlo = max(z - 5, 0)
        zhi = zlo + 10
        summary.append(np.mean(c[zlo:zhi]))

        # Plot points used to determine peak
        plt.plot(time[i+zlo:i+zhi], current[i+zlo:i+zhi], 'x-', color='k')

        # Determine axis limits
        plt.xlim(time[i] - 100, time[i + j] + 100)
        c = current[i-1000:i+j+1000]
        ymd = 0.5 * (np.max(c) + np.min(c))
        ylo = ymd - 1.3 * (ymd - np.min(c))
        yhi = ymd - 1.3 * (ymd - np.max(c))
        plt.ylim(ylo, yhi)
    plt.tight_layout()

else:
    raise NotImplementedError(protocol)

# Finalise
plt.show()

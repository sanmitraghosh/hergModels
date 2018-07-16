#!/usr/bin/env python
import os
import numpy as np
import scipy.io
import pints.io
import pints.plot
import matplotlib.pyplot as plt


# Load my data
print('Loading Myokit/Pints data')
michael = pints.io.load_samples('chain_0.csv')


# Load Aidan's data
print('Loading Web Lab data')
order = [
    'kO1',
    'kO2',
    'kC1',
    'kC2',
    'kI1',
    'kI2',
    'kA1',
    'kA2',
]
aidan = []
for filename in order:
    # Get filename
    filename = 'herg:rapid_delayed_rectifier_potassium_channel_' + filename
    filename += '_histogram_gnuplot_data.csv'
    filename = os.path.join('aidan', filename)
    # Load data
    xs = []
    ys = []
    with open(filename, 'r') as f:
        for line in f:
            x, y = line.strip().split(',')
            xs.append(float(x))
            ys.append(float(y))
    # Normalise
    xs = np.array(xs)
    ys = np.array(ys)
    bin_width = np.mean(xs[1:] - xs[:-1])
    ys /= (np.sum(ys) * bin_width)
    # Store
    aidan.append((xs, ys))


# Load Kylie's data
print('Loading Matlab data')
kylie = scipy.io.loadmat('MCMCChain_16713110_hh_sine_wave_30082148.mat')['T']

# Remove burn-in
michael = michael[50000:]
kylie = kylie[50000:]

# Load Kylie's parameters
print('Loading Kylie\'s parameters')
with open('../2-sine-fit/kylies-solution.txt', 'r') as f:
    pkylie = np.array([float(x) for x in f.readlines()])

# Load Michael's parameters
print('Loading Michael\'s parameters')
with open('../2-sine-fit/solution3.txt', 'r') as f:
    pmichael = np.array([float(x) for x in f.readlines()])

# Compare traces
print('Creating plot')
plt.rc('xtick', labelsize=8)
plt.figure(figsize=(12, 2*9))
parameters = ['Parameter ' + str(1 + i) for i in xrange(9)]
parameters[-1] = 'g'
for i, p in enumerate(parameters):
    # Add histogram subplot
    plt.subplot(9, 2, 1 + 2 * i)
    plt.xlabel(p)
    plt.ylabel('Frequency')
    plt.hist(kylie[:, i], bins=40, alpha=0.5, label='Matlab', normed=True)
    plt.hist(michael[:, i], bins=40, alpha=0.5, label='Myokit/Pints',
        normed=True)
    try:
        xs, ys = aidan[i]
        plt.plot(xs, ys, drawstyle='steps-mid', label='Web Lab')
    except IndexError:
        pass
    plt.axvline(pkylie[i], label='Matlab CMAES', color='tab:blue')
    plt.axvline(pmichael[i], label='Myokit/Pints CMAES', color='tab:orange')
    plt.legend()

    # Add trace subplot
    plt.subplot(9, 2, 2 + 2 * i)
    plt.xlabel('Iteration')
    plt.ylabel(p)
    plt.plot(kylie[:, i], alpha=0.5, label='Matlab')
    plt.plot(michael[:, i], alpha=0.5, label='Myokit/Pints')
plt.tight_layout()
plt.savefig('sine-wave-mcmc-trace.png')
#plt.show()

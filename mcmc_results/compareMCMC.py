#!/usr/bin/env python
import os
import sys
import numpy as np
import scipy.io
import pints.io
import pints.plot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)

#change to mcmc results directory
#sys.path.append(os.path.abspath('mcmc_results'))
# Load Michael's data
print('Loading Myokit/Pints data')
michael = pints.io.load_samples('cell-5-chain.csv')


# Load Kylie's data
print('Loading Matlab data')
kylie = scipy.io.loadmat('MCMCChain_16713110_hh_sine_wave_30082148.mat')['T']

# Load SMC data
print('Loading Matlab data')
df=pd.read_csv('sine-samples500.csv')
SMC = df.values[:,1:]
print(SMC.shape)
# Remove burn-in
#michael = michael[50000:]
#kylie = kylie[50000:]
print(michael.shape)
michael = michael[50000::40]
kylie = kylie[50000::40]

print(michael.shape)
print(kylie.shape)
# Load Kylie's parameters
print('Loading Kylie\'s parameters')
with open('kylies-solution.txt', 'r') as f:
    pkylie = np.array([float(x) for x in f.readlines()])

# Load Michael's parameters
print('Loading Michael\'s parameters')
with open('solution3.txt', 'r') as f:
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
    #plt.hist(kylie[:, i], bins=40, alpha=0.5, label='Matlab', normed=True)
    plt.hist(michael[:, i], bins=40, alpha=0.5, label='Myokit/Pints',
        normed=True)
    plt.hist(SMC[:, i], bins=40, alpha=0.5, label='SMC/PyMC3',
        normed=True)
    sns.kdeplot(michael[:, i], color='blue', label='Myokit/Pints KDE')
    sns.kdeplot(SMC[:, i], color='green', label='SMC KDE')
    #sns.kdeplot(kylie[:, i], color='magenta', label='MATLAB KDE')
    #plt.axvline(pkylie[i], label='Matlab CMAES', color='tab:blue')
    plt.axvline(pmichael[i], label='Myokit/Pints CMAES', color='tab:orange')
    plt.legend()

    # Add trace subplot
    plt.subplot(9, 2, 2 + 2 * i)
    plt.xlabel('Iteration')
    plt.ylabel(p)
    plt.plot(SMC[:, i], alpha=0.5, color='green', label='SMC trace')
    #plt.plot(kylie[:, i], alpha=0.5, label='Matlab')
    plt.plot(michael[:, i], color='blue', alpha=0.5, label='Myokit/Pints')
    plt.legend()
plt.tight_layout()
plt.savefig('sine-wave-mcmc-trace.png')

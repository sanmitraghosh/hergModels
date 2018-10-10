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
import scipy.stats
from collections import namedtuple
from scipy.special import logsumexp
import warnings
import cPickle
import myokit
import argparse
import matplotlib.pyplot as plt

outfile = './figures/model_metrics.txt'
model_metrics = np.loadtxt(outfile)
model=np.zeros((5,7))
a=model_metrics[0,:]#kylie
b=model_metrics[4,:]#best
model[0:3,:] = model_metrics[1:4,:]
model[3,:] = a#kylie alloted
model[4,:]=b#best alloted
outfile2 = './figures/model_metrics2.txt'
np.savetxt(outfile2,model)
N = 5
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind,np.abs(model[:,6]) , width, color='r')
ax.set_ylabel('RMSE AP data')
ax.set_title('RMSE AP data, smaller the better')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(['C-O-I','C-C-O-I','C-C-C-O-I','Beattie', 'C-C-O-I-IC-IC'])
plt.show()
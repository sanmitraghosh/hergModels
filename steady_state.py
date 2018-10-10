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
import myokit
import argparse
import matplotlib.pyplot as plt

cell = 5
root = os.path.abspath('traditional-data')
print(root)

protocol_file = os.path.join(root,'steady-state-pacer.mmt')
protocol = myokit.load_protocol(protocol_file)

time = np.arange(0,50000,0.1)
sys.path.append(os.path.abspath('models_forward'))
import linearCCCOI as forwardModel


temperature = forwardModel.temperature(cell)
initial_parameters = forwardModel.fetch_parameters()
model = forwardModel.ForwardModel(protocol, temperature, sine_wave=False, logTransform=False)
current=model.simulate(initial_parameters, time)
states=model.simulated_states
print(states.T[0,:])
print(states.T[-1,:])
plt.plot(time,states.T)
plt.xlim([0,22000])
#plt.ylim([0,1])
plt.show()
#print(states[4,-1])
#!/usr/bin/env python2
#
# This is a script to simulate the sine wave protocol uisng new markov builder model
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

root = os.path.abspath('sine-wave-data')
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
# Select protocol file
#
protocol_file = os.path.join(root,'steps.mmt')
protocol = myokit.load_protocol(protocol_file)
time = np.arange(0,8000,0.1)


sys.path.append(os.path.abspath('models_forward'))
root = os.path.abspath('models_myokit')

from models import *
import pintsForwardModel as forwardModel
import LogPrior as prior
model = 'Model'+str(1)
myo_model, rate_dict_maker, n_params = globals()[model]()
print("loading  model: "+str(1))
model_name ='model-'+str(1)


temperature = forwardModel.temperature(cell)
initial_parameters = np.array([3.36487755e-05, 2.37551124e-02, 5.16502797e-07, 2.09294011e-01,
 5.68025341e-06, 1.24980119e-01, 2.54514804e-07, 9.46091145e-02,
 1.46737182e-01])
model = forwardModel.ForwardModel(protocol, temperature, myo_model, n_params, sine_wave=True, logTransform=False)
current=model.simulate(initial_parameters, time)

plt.plot(time,current)
plt.xlim([0,8000])
plt.show()

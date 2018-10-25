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

cell = 16
print("loading  model: "+str(16))
model_name ='model-'+str(3)

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
root = os.path.abspath('traditional-data')
protocol_file = os.path.join(root,'steady-state-pacer.mmt')
protocol = myokit.load_protocol(protocol_file)
time = np.arange(0,50000,0.1)


sys.path.append(os.path.abspath('models_forward'))
root = os.path.abspath('models_myokit')
myo_model = os.path.join(root, model_name + '.mmt')


sys.path.append(os.path.abspath('models_forward'))

import pintsForwardModel as forwardModel
import LogPrior as prior

temperature = forwardModel.temperature(cell)
initial_parameters = np.array([
 2.26072535774749178e-04,
 6.99209750759895721e-02,
 3.44949205359378056e-05,
 5.46123824706467934e-02,
 8.73163312716947609e-02,
 8.92935890222958106e-03,
 5.14857252048054204e-03,
 3.15628546404693316e-02,
 1.52432868579321312e-01
])
"""
print('Applying capacitance filtering')
time, voltage, current = forwardModel.capacitance(
    protocol, 0.1, time, voltage, current)
"""
model = forwardModel.ForwardModel(protocol, temperature, myo_model, 1, sine_wave=True, logTransform=False)
current_fitted=model.simulate(initial_parameters, time)
print(model.simulated_minf[-1])
print(model.simulated_hinf[-1])
plt.plot(time,model.simulated_minf,label='real')
plt.plot(time,model.simulated_hinf, label='fit')
plt.legend(loc='lower right')
plt.xlim([0,50000])
plt.show()

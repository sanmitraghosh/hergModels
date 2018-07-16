#!/usr/bin/env python
#
# Imports (leak-corrected, dofetilide subtracted) data from Kylie's AP protocol
# experiments, as well as a protocol file (both in MatLab)
#
# Stores a CSV data file and a CSV protocol file
#
# Capacitance artefacts are NOT filtered out before storing, although filtered
# data is show for debugging purposes.
#
from __future__ import print_function
import os
import sys
import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as pl
import myokit
import myokit.formats.axon

show_debug = True

cell = 'cell-5'

protocol = 'ap'

cells = {
    'cell-1': '16713003',
    'cell-2': '16715049',
    'cell-3': '16708016',
    'cell-4': '16708060',
    'cell-5': '16713110',
    'cell-6': '16708118',
    'cell-7': '16704007',
    'cell-8': '16704047',
    'cell-9': '16707014',
}

idx = cells[cell]
pro = protocol

# Load protocol from protocol file
print('Reading matlab protocol')
mat = pro + '_protocol.mat'
mat = scipy.io.loadmat(mat)
vm = mat['T']
vm = vm[:,0]  # Convert from matrix to array
del(mat)

# Load leak-corrected, dofetilide-subtracted IKr data from matlab file
print('Reading matlab data')
mat = pro + '_' + idx + '_dofetilide_subtracted_leak_subtracted.mat'
mat = scipy.io.loadmat(mat)
current = mat['T']
current = current[:,0]  # Convert from matrix to array
del(mat)

# Create times array, using dt=0.1ms
dt = 0.1
time = np.arange(len(current)) * dt


# Create datalog
d = myokit.DataLog()
d.set_time_key('time')
d['time'] = time
d['voltage'] = vm
d['current'] = current

# Store data in csv
if True:
    filename = protocol + '-' + cell + '.csv'
    print('  Writing data to ' + filename)
    d.save_csv(filename)
print('Done')

# Show folded data
if show_debug:
    # Show data with capacitance artefacts
    print('Plotting data with artefacts')
    pl.figure()
    pl.subplot(2,1,1)
    pl.plot(time, vm, color='darkgreen', label='V (exp)')
    pl.xlabel('Time (ms)')
    pl.ylabel('V (mV)')
    pl.legend()
    pl.subplot(2,1,2)
    pl.plot(time, current, color='darkblue', label='I (exp)')
    pl.xlabel('Time (ms)')
    pl.ylabel('I (nA)')
    pl.legend()

    # Show graphs
    pl.show()


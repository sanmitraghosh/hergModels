#!/usr/bin/env python
#
# Imports (leak-corrected, dofetilide subtracted) data from Kylie's traditional
# protocol experiments, as well as a protocol file (both in MatLab)
#
# Stores a CSV data file and a Myokit protocol file.
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

show_debug = False

cell = 'cell-5'

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

protocols = {
    'pr1-activation-kinetics-1' : 'activation_kinetics_1',
    'pr2-activation-kinetics-2' : 'activation_kinetics_2',
    'pr3-steady-activation' : 'steady_activation',
    'pr4-inactivation' : 'inactivation',
    'pr5-deactivation' : 'deactivation',
}

repeats = {
    'pr1-activation-kinetics-1' : 6,
    'pr2-activation-kinetics-2' : 6,
    'pr3-steady-activation-1' : 7,
    'pr4-inactivation' : 16,
    'pr5-deactivation' : float('nan'), #9,
}

for protocol, pro in protocols.items():

    idx = cells[cell]
    pro = protocols[protocol]

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

    # Correct tiny shift in stored data (doubling final point)
    #vm[:-1] = vm[1:]
    #current[:-1] = current[1:]

    # Get jump times
    i = np.array([0] + list(np.where(np.abs(vm[1:] - vm[:-1]))[0]))
    start_guess = time[i]

    # Account for tiny shift in stored data
    start_guess[1:] += dt

    # Store protocol in Myokit file
    if True:
        filename = protocol + '.mmt'
        print('  Writing protocol to ' + filename)
        level_guess = vm[i + 2]
        length_guess = list(start_guess)
        length_guess.append(time[-1] + (time[-1] - time[-2]))
        length_guess = np.array(length_guess)
        length_guess = length_guess[1:] - length_guess[:-1]
        with open(filename, 'w') as f:
            f.write('[[protocol]]\n')
            f.write('# ' + protocol + '\n')
            f.write('# Level Start   Length  Period  Multiplier\n')
            for k in xrange(len(start_guess)):
                for x in [level_guess, start_guess, length_guess]:
                    f.write(str(x[k]) + (8 - len(str(x[k]))) * ' ' )
                f.write('0       0\n')
        del(filename, level_guess, length_guess)

    jumps = start_guess[1:]
    del(start_guess, i)

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

        # Remove capacitance artefacts
        print('Removing capacitance artefacts for plot only')
        cap_duration = 5 # Same as Kylie (also, see for example step at 500ms)
        for t in jumps:
            # Get indices of capacitance start and end
            i1 = (np.abs(time - t)).argmin()
            i2 = (np.abs(time - t - cap_duration)).argmin()
            # Flatten signal during capacitance artefact
            current[i1:i2] = np.mean(current[i1-(i2-i1): i1])

        # Show data without capacitance artefacts
        pl.subplot(2,1,2)
        pl.plot(
            time, current, color='tab:orange', label='I (exp, cap flattened)')
        pl.legend()

        # Show folded data
        if np.isfinite(repeats[protocol]):
            print('Showing folded current data')
            period = (time[-1] + dt) / repeats[protocol]
            print('period: ' + str(period))
            d = d.fold(period)
            pl.figure()
            pl.subplot(1, 2, 1)
            for key in d.keys_like('voltage'):
                pl.plot(d.time(), d[key])
            pl.xlabel('Time (ms)')
            pl.ylabel('V (mV)')
            pl.subplot(1, 2, 2)
            for key in d.keys_like('current'):
                pl.plot(d.time(), d[key])
            pl.xlabel('Time (ms)')
            pl.ylabel('I (nA)')

        # Show graphs
        pl.show()


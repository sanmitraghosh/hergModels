#!/usr/bin/env python2
#
# Compare Kylie's simulation data with Myokit simulation data, for cell 5.
#
from __future__ import division
from __future__ import print_function
import os
import sys
import pints
import numpy as np
import scipy.io
import myokit
import myokit.pacing as pacing

root = os.path.realpath(os.path.join('..', '..'))

# Show graphs
OPTION_GRAPHS = True

# Simulation options
OPTION_TOL10 = True
OPTION_MAX_STEP_SIZE = True

# Comparison options
OPTION_MATLAB_STD = False
OPTION_KYLIE_FCAP = False
OPTION_TRANSFORM  = True
OPTION_KYLIE_DATA = False

# Model and data file
cell = 5
model_file = os.path.join(root, 'models', 'beattie-2017-ikr-markov.mmt')
data_file = os.path.join(root, 'sine-wave-data',
    'cell-' + str(cell) + '.csv')

# Load Kylie's data
sine_root = os.path.join('..', '..', 'sine-wave-data')
if OPTION_TOL10:
    kylie_sim = 'cell5_tol10.mat'
else:
    kylie_sim = 'cell5_tol8.mat'
kylie_sim = scipy.io.loadmat(os.path.join(sine_root, kylie_sim))
kylie_sim = kylie_sim['I'].reshape((80000, ))
kylie_pro = scipy.io.loadmat(os.path.join(sine_root, 'sine_wave_protocol.mat'))
kylie_pro = kylie_pro['T'].reshape((80000, ))

# Create time array
times = np.arange(80000) * 0.1

#
# Cell temperature
#
temperatures = {
    5 : 21.4,   # 16713110
    }
#if strcmp(exp_ref,'16708016')==1    temperature = 21.8;
#if strcmp(exp_ref,'16708060')==1    temperature = 21.7;
#if strcmp(exp_ref,'16704047')==1    temperature = 21.6;
#if strcmp(exp_ref,'16704007')==1    temperature = 21.2;
#if strcmp(exp_ref,'16713003')==1    temperature = 21.3;
#if strcmp(exp_ref,'16715049')==1    temperature = 21.4;
#if strcmp(exp_ref,'16707014')==1    temperature = 21.4;
#if strcmp(exp_ref,'16708118')==1    temperature = 21.7;
temperature = temperatures[cell]

#
# Load leak-corrected, dofetilide-subtracted IKr data from matlab file
#
current = 'sine_wave_16713110_dofetilide_subtracted_leak_subtracted.mat'
current = scipy.io.loadmat(os.path.join(sine_root, current))
current = current['T'].reshape((80000, ))


#
# Estimate noise from start of data
#
if OPTION_MATLAB_STD:
    sigma_noise = np.std(current[:2000], ddof=1)    # ddof=1 mimmicks Matlab
else:
    sigma_noise = np.std(current[:2000])
print('Noise estimate: ' + pints.strfloat(sigma_noise))

#
# Protocol info
#
dt = 0.1
steps = [
    [-80, 250.1],
    [-120, 50],
    [-80, 200],
    [40, 1000],
    [-120, 500],
    [-80, 1000],
    [-30, 3500],
    [-120, 500],
    [-80, 1000],
    ]

#
# Create capacitance filter based on protocol
#
cap_duration = 5 # Same as Kylie (also, see for example step at 500ms)
fcap = np.ones(len(current), dtype=int)
if False:
    if OPTION_KYLIE_FCAP:
        # Manual filter, ported from Kylie's code
        fcap = np.zeros(len(current), dtype=int)
        fcap[:2499] = 1
        fcap[2548:2999] = 1
        fcap[3048:4999] = 1
        fcap[5048:14999] = 1
        fcap[15048:19999] = 1
        fcap[20048:29999] = 1
        fcap[30048:64999] = 1
        fcap[65048:69999] = 1
        fcap[70048:] = 1
        fcap = fcap > 0
        print(len(fcap))
        print(len(fcap[fcap]))
    else:
        # Skip this bit to show/use all data
        offset = 0
        for f, t in steps[:-1]:
            offset += t
            i1 = int(offset / dt)
            i2 = i1 + int(cap_duration / dt)
            fcap[i1:i2] = 0
        fcap = fcap > 0

    # Apply capacitance filter to data
    times = times[fcap]
    current = current[fcap]
    kylie_sim = kylie_sim[fcap]
    kylie_pro = kylie_pro[fcap]

#
# Calculate reversal potential like Kylie does
#
def erev(temperature):
    T = 273.15 + temperature
    F = 96485.0
    R = 8314.0
    K_i = 130.0
    k_o = 4.0
    return ((R*T)/F) * np.log(k_o/K_i)
E = erev(temperature)

#
# Create ForwardModel
#
class ModelWithProtocol(pints.ForwardModel):
    parameters = [
        'ikr.p1',
        'ikr.p2',
        'ikr.p3',
        'ikr.p4',
        'ikr.p5',
        'ikr.p6',
        'ikr.p7',
        'ikr.p8',
        'ikr.p9',
        ]
    def __init__(self):
        # Load model
        model = myokit.load_model(model_file)
        # Set reversal potential
        model.get('nernst.EK').set_rhs(E)
        # Add sine-wave equation to model
        model.get('membrane.V').set_rhs(
            'if(engine.time >= 3000.1 and engine.time < 6500.1,'
            + ' - 30'
            + ' + 54 * sin(0.007 * (engine.time - 2500.1))'
            + ' + 26 * sin(0.037 * (engine.time - 2500.1))'
            + ' + 10 * sin(0.190 * (engine.time - 2500.1))'
            + ', engine.pace)')
        # Create step protocol
        protocol = myokit.Protocol()
        for f, t in steps:
            protocol.add_step(f, t)
        # Create simulation
        self.simulation = myokit.Simulation(model, protocol)
        # Set solver tolerances
        if OPTION_TOL10:
            self.simulation.set_tolerance(1e-10, 1e-10)
        else:
            self.simulation.set_tolerance(1e-8, 1e-8)
        if OPTION_MAX_STEP_SIZE:
            self.simulation.set_max_step_size(0.1)
    def dimension(self):
        return len(self.parameters)
    def simulate(self, parameters, times):
        # Note: Kylie doesn't do pre-pacing!
        # Update model parameters
        for i, name in enumerate(self.parameters):
            self.simulation.set_constant(name, parameters[i])
        # Run
        self.simulation.reset()
        try:
            d = self.simulation.run(
                np.max(times+0.5*dt),
                log_times = times,
                log = ['engine.time', 'ikr.IKr', 'membrane.V'],
                ).npview()
        except myokit.SimulationError:
            return times * float('inf')
        # Store membrane potential for debugging
        self.simulated_v = d['membrane.V']
        # Return
        return d['ikr.IKr']

class ModelDataClamp(pints.ForwardModel):
    parameters = [
        'ikr.p1',
        'ikr.p2',
        'ikr.p3',
        'ikr.p4',
        'ikr.p5',
        'ikr.p6',
        'ikr.p7',
        'ikr.p8',
        'ikr.p9',
        ]
    def __init__(self):
        # Load model
        model = myokit.load_model(model_file)
        # Set reversal potential to Kylie value
        model.get('nernst.EK').set_rhs(E)
        # Create simulation
        self.simulation = myokit.Simulation(model)
        # Apply data-clamp
        self.simulation.set_fixed_form_protocol(times, voltage)
        # Set Kylie tolerances
        self.simulation.set_tolerance(1e-8, 1e-8)
        # Set Kylie max step size (also needed for data clamp)
        self.simulation.set_max_step_size(0.1)
    def dimension(self):
        return len(self.parameters)
    def simulate(self, parameters, times):
        # Update model parameters
        for i, name in enumerate(self.parameters):
            self.simulation.set_constant(name, parameters[i])
        # Run
        self.simulation.reset()
        try:
            d = self.simulation.run(
                np.max(times),
                log_times = times,
                log = ['ikr.IKr', 'membrane.V'],
                ).npview()
        except myokit.SimulationError:
            return times * float('inf')
        # Store membrane potential for debugging
        self.simulated_v = d['membrane.V']
        # Return
        return d['ikr.IKr']

#
# Choose simulation implementation
#
model = ModelWithProtocol()

#
# Define problem
#
problem = pints.SingleSeriesProblem(model, times, current)

#
# Select a score function
#
score = pints.SumOfSquaresError(problem)

#
# Load kylie's parameters
#
filename = 'kylies-solution.txt'
with open(filename, 'r') as f:
    kylies_parameters = [float(x) for x in f.readlines()]
kylies_parameters = np.array(kylies_parameters)

#
# Transform and untransform Kylie's parameters
#
if OPTION_TRANSFORM:
    print('Transforming and untransforming parameters')
    kylies_parameters = np.log10(kylies_parameters)
    kylies_parameters = 10 ** kylies_parameters

#
# Show kylie's parameters
#
print('Kylie\'s parameters:')
for x in kylies_parameters:
    print(pints.strfloat(x))
print('These parameters are used in all simulations!')

#
# Simulate data, or use Kylie's
#
if OPTION_KYLIE_DATA:
    simulated = kylie_sim
else:
    simulated = model.simulate(kylies_parameters, times)

#
# Calculate Log-likelihood
#
def ll_signal(x, y, sigma):
    """ Calculate the log-likelihood of two signals matching """
    x, y = np.asarray(x), np.asarray(y)
    return (
        -0.5 * len(x) * np.log((2*np.pi) * (sigma**2))
        -0.5 * (1. / ((sigma)**2)) * (np.sum((x - y)**(2)))
        )
ll = ll_signal(current, simulated, sigma_noise)

# Print reference value
if OPTION_TOL10:
    print('-1.510637413819120e+06 Kylie 10, Log')
    #print('-1.510637413318209e+06 Kylie 10, No Log')
else:
    print('-1.510637010057245e+06 Kylie 8, Log')
    #print('-1.510637042276509e+06 Kylie 8, No Log')

# Print calculated value
print(pints.strfloat(ll))
print('Using Matlab std() function   : ' + str(OPTION_MATLAB_STD))
print('Using Kylie capacitance filter: ' + str(OPTION_KYLIE_FCAP))
print('Using Log10 transform         : ' + str(OPTION_TRANSFORM))
print('Using Kylie simulation data   : ' + str(OPTION_KYLIE_DATA))


'''
simulated = model.simulate(kylies_parameters, times)
print('Myokit, manual log-likelihood calculation, Kylie\'s sigma:')
print('  Log-likelihood: '
    + pints.strfloat(loglikelihood(simulated, current, sigma_kylie)))

#print('Matlab data, manual log-likelihood calculation in this script:')
#print('  Log-likelihood: '
#    + pints.strfloat(loglikelihood(kylie_sim, current, sigma_noise)))

print(
    'Matlab data, manual log-likelihood calculation in this script, '
    'Kylie\'s sigma:')
print('  Log-likelihood: '
    + pints.strfloat(loglikelihood(kylie_sim, current, sigma_kylie)))

print('--------------------------')
print('Comparison with value reported by Kylie:')
print(
    pints.strfloat(loglikelihood(kylie_sim, current, sigma_kylie))
    + ' Matlab data, Kylie sigma')




print(myokit.strfloat(ll_matlab))
'''


if not OPTION_GRAPHS:
    sys.exit(1)


#
# Show result
#

# Simulate
simulated = model.simulate(kylies_parameters, times)
if OPTION_KYLIE_DATA:
    print('GRAPHS GENERATED USING MYOKIT DATA')

# Plot
import matplotlib.pyplot as pl

#
# Full comparison: protocol, transitions, current, current at transitions
# This plot works best with the capacitance filter switched off
#
pl.figure()
pl.suptitle('Simulated protocols and currents in Matlab and Myokit')

# Show whole voltage trace
pl.subplot(4,1,1)
pl.plot(times, kylie_pro, 'x-', lw=4, alpha=0.75, label='kylie')
pl.plot(times, model.simulated_v, 'o-', alpha=0.75, label='michael')
pl.xlabel('Time (ms)')
pl.ylabel('V (mV)')
pl.legend(loc='upper right')

# Indicate transitions in whole voltage trace
n = len(steps) + 1
offset = 0
for v, t in steps:
    offset += t
    pl.axvline(offset - 1, alpha=0.25)
    pl.axvline(offset + 1, alpha=0.25)

# Show boxes with voltage transitions
# (This plot is empty with capacitance filtering turned on)
nlo, nhi = 2, 2
n = len(steps) + 1
pl.subplot(4,n,1+n)
pl.ylabel('V (mV)')
offset = 0
lo, hi = 0, int(offset/dt) + nhi + 2
pl.plot(times[lo:hi], kylie_pro[lo:hi], 'x-', lw=4, alpha=0.75, label='kylie')
pl.plot(times[lo:hi], model.simulated_v[lo:hi], 'o-', alpha=0.75,
        label='michael')
pl.xlim(offset - nlo*dt, offset + nhi*dt)
pl.tick_params(axis='x', labelsize=6)
pl.grid(True)
for k, step in enumerate(steps):
    pl.subplot(4,n,2+n+k)
    v, t = step
    offset += t
    lo = int(offset/dt) - nlo - 2
    hi = min(int(offset/dt) + nhi + 2,len(times))
    pl.plot(times[lo:hi], kylie_pro[lo:hi], 'x-', lw=4, alpha=0.75,
            label='kylie')
    pl.plot(times[lo:hi], model.simulated_v[lo:hi], 'o-', alpha=0.75,
            label='michael')
    pl.xlim(offset - nlo*dt, offset + nhi*dt)
    pl.tick_params(axis='x', labelsize=6)
    pl.grid(True)

# Show whole current trace
pl.subplot(4,1,3)
pl.grid(True)
pl.plot(times, kylie_sim, 'x-', label='kylie')
pl.plot(times, simulated, 'o-', label='michael')
pl.legend(loc='lower right')
pl.xlabel('Time (ms)')
pl.ylabel('I (nA)')

# Indicate transitions in whole current trace
n = len(steps) + 1
offset = 0
for v, t in steps:
    offset += t
    pl.axvline(offset, alpha=0.25)

# Show current at transitions in boxes
# (This plot is empty with capacitance filtering turned on)
nlo, nhi = 3, 10
n = len(steps) + 1
pl.subplot(4,n,1+3*n)
pl.ylabel('I (nA)')
offset = 0
lo, hi = 0, int(offset/dt) + nhi
pl.plot(times[lo:hi], kylie_sim[lo:hi], 'x-', label='kylie')
pl.plot(times[lo:hi], simulated[lo:hi], 'o-', label='michael')
pl.xlim(offset - dt*nlo, offset + dt*nhi)
for k, step in enumerate(steps):
    pl.subplot(4,n,2+3*n+k)
    v, t = step
    offset += t
    lo, hi = int(offset/dt) - nlo, min(int(offset/dt) + nhi, len(times))
    pl.plot(times[lo:hi], kylie_sim[lo:hi], 'x-', label='kylie')
    pl.plot(times[lo:hi], simulated[lo:hi], 'o-', label='michael')
    pl.xlim(offset - dt*nlo, offset + dt*nhi)

pl.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.25)

#
# Big plot of my current versus kylie's
#
pl.figure()
pl.title('Simulated currents in Matlab and Myokit')
pl.plot(times, kylie_sim, 'x-', lw=2, alpha=0.5, label='kylie')
pl.plot(times, simulated, 'o-', alpha=0.5, label='michael')
#pl.plot(times, current, label='real')
pl.legend(loc='lower right')
n = len(steps) + 1
offset = 0
for v, t in steps:
    offset += t
    pl.axvline(offset, alpha=0.25)
pl.xlabel('Time (ms)')
pl.ylabel('I (nA)')

#
# Show error statistics
#
print('-'*40)
print('Max error in current: '
      + str(np.max(np.abs(kylie_sim - simulated))) + ' nA')
print('Max error in voltage: '
      + str(np.max(np.abs(kylie_pro - model.simulated_v))) + ' mV')

#
# Big plot of the error between my current and Kylie's
#
pl.figure(figsize=(6,4))
pl.title('Error in simulated currents')
pl.grid(True)
pl.plot(times, kylie_sim - simulated, color='tab:blue',
    label='I_error (matlab - myokit)')
pl.legend()
n = len(steps) + 1
offset = 0
for v, t in steps:
    offset += t
    pl.axvline(offset, color='tab:green', alpha=0.25)
pl.xlabel('Time (ms)')
pl.ylabel('I (nA)')

#
# Big plot of the error between my protocol and Kylie's
#
pl.figure(figsize=(6,4))
pl.suptitle('Error in simulated protocols')
pl.grid(True)
pl.plot(times, kylie_pro - model.simulated_v, color='tab:green',
    label='V_error (matlab - myokit)')
pl.legend()
n = len(steps) + 1
offset = 0
for v, t in steps:
    offset += t
    pl.axvline(offset, color='tab:green', alpha=0.25)
pl.xlabel('Time (ms)')
pl.ylabel('V (mV)')
pl.tight_layout()

pl.show()


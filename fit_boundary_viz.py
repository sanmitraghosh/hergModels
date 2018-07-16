#!/usr/bin/env python2
#
# Fit Kylie's model to Cell 5 data using CMA-ES
#
from __future__ import division, print_function
import os
import sys
import pints
import numpy as np
import myokit
from boundaries import go
import matplotlib.pyplot as plt
# Load beattie model and prior
sys.path.append(os.path.abspath('models_forward'))
import circularCOIIC as forwardModel
model_name ='CCOIIC'

# Check input arguments

#
# Select cell
#
cell = 5


#
# Select data file
#
root = os.path.abspath('sine-wave-data')
print(root)
data_file = os.path.join(root, 'cell-' + str(cell) + '.csv')


#
# Select protocol file
#
protocol_file = os.path.join(root,'steps.mmt')


#
# Cell-specific parameters
#
temperature = forwardModel.temperature(cell)
lower_conductance = forwardModel.conductance_limit(cell)


#
# Load protocol
#
protocol = myokit.load_protocol(protocol_file)


#
# Load data
#
log = myokit.DataLog.load_csv(data_file).npview()
time = log.time()
current = log['current']
voltage = log['voltage']
del(log)


#
# Estimate noise from start of data
# Kylie uses the first 200ms, where I = 0 + noise
#
sigma_noise = np.std(current[:2000], ddof=1)


#
# Apply capacitance filter based on protocol
#
print('Applying capacitance filtering')
time, voltage, current = forwardModel.capacitance(
    protocol, 0.1, time, voltage, current)


#
# Create forward model
#
model = forwardModel.ForwardModel(protocol, temperature, sine_wave=True)


#
# Define problem
#
problem = pints.SingleOutputProblem(model, time, current)


#
# Define log-posterior
#
log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
log_prior = forwardModel.LogPrior(lower_conductance)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)

#
# Plot prior area
#
log=True
lower_alpha = 1e-7              # Kylie: 1e-7
upper_alpha = 1e3               # Kylie: 1e3
lower_beta  = 1e-7              # Kylie: 1e-7
upper_beta  = 0.4               # Kylie: 0.4

n = 1000
b = np.linspace(lower_beta, upper_beta, n)
if log:
	a = np.exp(np.linspace(np.log(lower_alpha), np.log(upper_alpha), n))
else:
	a = np.linspace(lower_alpha, upper_alpha, n)

rmin = 1.67e-5
rmax = 1000

vmin = -120
vmax = 58.25

title = 'r = p1 * np.exp(p2 * v)'
title += ' v in [' + str(vmin) + ', ' + str(vmax) + ']'
title += ' r in [' + str(rmin) + ', ' + str(rmax) + ']'

plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.xlabel('p1,p5')
plt.ylabel('p2,p6')
plt.axvline(lower_alpha, color='k', alpha=0.25)
plt.axvline(upper_alpha, color='k', alpha=0.25)
plt.axhline(lower_beta, color='k', alpha=0.25)
plt.axhline(upper_beta, color='k', alpha=0.25)
if log:
	plt.xscale('log')
	plt.yscale('log')
       	plt.xlim(lower_alpha * 0.1, upper_alpha * 10)
	plt.ylim(lower_beta *0, upper_beta *10)
else:
	plt.xlim(lower_alpha - 50, upper_alpha + 50)

bmin = (1 / vmax) * (np.log(rmin) - np.log(a))
bmax = (1 / vmax) * (np.log(rmax) - np.log(a))
bmin = np.maximum(bmin, lower_beta)
bmax = np.minimum(bmax, upper_beta)
plt.fill_between(a, bmin, bmax, color='k', alpha=0.1, label='Prior')
plt.plot(a, bmin, label='Lower bound')
plt.plot(a, bmax, label='Upper bound')

#plt.figure(figsize=(12,6))
plt.subplot(1, 2, 2)
plt.xlabel('p3,p7')
plt.ylabel('p4,p8')
plt.axvline(lower_alpha, color='k', alpha=0.25)
plt.axvline(upper_alpha, color='k', alpha=0.25)
plt.axhline(lower_beta, color='k', alpha=0.25)
plt.axhline(upper_beta, color='k', alpha=0.25)
if log:
	plt.xscale('log')
	plt.yscale('log')
       	plt.xlim(lower_alpha * 0.1, upper_alpha * 10)
	plt.ylim(lower_beta *0, upper_beta *10)
else:
	plt.xlim(lower_alpha - 50, upper_alpha + 50)
	
bmin = (-1 / vmin) * (np.log(rmin) - np.log(a))
bmax = (-1 / vmin) * (np.log(rmax) - np.log(a))
bmin = np.maximum(bmin, lower_beta)
bmax = np.minimum(bmax, upper_beta)
plt.fill_between(a, bmin, bmax, color='k', alpha=0.1, label='Prior')
plt.plot(a, bmin, label='Lower bound')
plt.plot(a, bmax, label='Upper bound')

# Run repeated optimisations
repeats = 10
params, scores = [], []
for i in xrange(repeats):
	# Choose random starting point
	x0 = log_prior.sample()
	
	# Create optimiser
	x0=np.log(x0)
	#x0[0],x0[2],x0[4],x0[6] =np.log([x0[0],x0[2],x0[4],x0[6]])
	
	opt = pints.Optimisation(log_posterior, x0, method=pints.CMAES)
	opt.set_max_iterations(None)
	opt.set_parallel(True)

	# Run optimisation
	try:
	    with np.errstate(all='ignore'): # Tell numpy not to issue warnings
		p, s = opt.run()
		params.append(p)
		scores.append(s)
		print(p)
		#go(True,np.exp(p))
		plt.subplot(1, 2, 1)
		plt.plot(np.exp(p)[0], np.exp(p)[1], 'x', color='r', label='optim run '+str(i))
		#plt.plot(np.exp(p)[0], p[1], 'x', color='r', label='optim run '+str(i))
    		plt.plot(np.exp(p)[4], np.exp(p)[5], 'x', color='r')
		#plt.plot(np.exp(p)[4], p[5], 'x', color='r')
		plt.legend(loc='upper right').get_frame().set_alpha(1)	
		plt.subplot(1, 2, 2)
		plt.plot(np.exp(p)[2], np.exp(p)[3], 'x', label='optim run '+str(i))
		#plt.plot(np.exp(p)[2], p[3], 'x', label='optim run '+str(i))
    		plt.plot(np.exp(p)[6], np.exp(p)[7], 'x')
		#plt.plot(np.exp(p)[6], p[7], 'x')
		plt.legend(loc='upper right').get_frame().set_alpha(1)
	
	except ValueError:
	    import traceback
	    traceback.print_exc()

plt.show()


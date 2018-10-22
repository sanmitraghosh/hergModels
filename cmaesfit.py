#!/usr/bin/env python2
#
# All model to Cell 5 data using CMA-ES
#
from __future__ import division, print_function
import os
import sys
import pints
import numpy as np
import myokit
import argparse
import cPickle
import matplotlib.pyplot as plt

# Load beattie model and prior



# Check input arguments

#
# Select cell
#

parser = argparse.ArgumentParser(description='Fit all the hERG models to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N', \
      help='cell number : 1, 2, ..., 5' )
parser.add_argument('--model', type=int, default=16, metavar='N', \
      help='model number : 1 for C-O-I-IC, 2 for C-O and so on' )
parser.add_argument('--plot', type=bool, default=True, metavar='N', \
      help='plot fitted traces' )
args = parser.parse_args()


# Import markov models from the models file, and rate dictionaries.
model_name ='model-'+str(args.model)
root = os.path.abspath('models_myokit')
myo_model = os.path.join(root, model_name + '.mmt')
root = os.path.abspath('rate_dictionaries')
rate_file = os.path.join(root, model_name +'-priors.p')
rate_dict = cPickle.load(open(rate_file, 'rb'))

sys.path.append(os.path.abspath('models_forward'))

import pintsForwardModel as forwardModel
import LogPrior as prior

print("loading  model: "+str(args.model))
model_name ='model-'+str(args.model)

cell = args.cell

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
model = forwardModel.ForwardModel(protocol, temperature, myo_model, sine_wave=True, logTransform=True)
n_params = model.n_params
#
# Define problem
#
problem = pints.SingleOutputProblem(model, time, current)


#
# Define log-posterior
#


log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
log_prior = prior.LogPrior(rate_dict, lower_conductance, n_params, logTransform=True)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)


# Run repeated optimisations
repeats = 1
params, scores = [], []

func_calls = []
for i in xrange(repeats):
	# Choose random starting point
	x0 = log_prior.sample()
	print(x0)
	# Create optimiser and log transform parameters
	x0=np.log(x0)
	
	#x0[1],x0[3],x0[5],x0[7],x0[9],x0[11] =np.log([x0[1],x0[3],x0[5],x0[7],x0[9],x0[11]])
	#x0 = np.array(x0)
	print(x0)
	opt = pints.Optimisation(log_posterior, x0, method=pints.XNES)
	opt.set_max_iterations(10)
	opt.set_parallel(True)

	# Run optimisation
	try:
	    with np.errstate(all='ignore'): # Tell numpy not to issue warnings
		p, s = opt.run()
		p = np.exp(p)
		#tm = np.zeros(n_params)
		#tm = p
		#p.setflags(write=1)
		#p[1],p[3],p[5],p[7],p[9],p[11] =np.exp([p[1],p[3],p[5],p[7],p[9],p[11]])
		

		func_calls.append(model.func_call)
		print(func_calls)
		params.append(p)
		scores.append(s)
		

	except ValueError:
	    import traceback
	    traceback.print_exc()

# Order from best to worst
order = np.argsort(scores)[::-1]
scores = np.asarray(scores)[order]
params = np.asarray(params)[order]

# Show results

if repeats > 1:
	print('Best 3 scores:')
	for i in xrange(3):
		print(scores[i])
		print('Mean & std of score:')
		print(np.mean(scores))
		print(np.std(scores))
		print('Worst score:')
		print(scores[-1])
else:
	print('Score:')
	print(scores)
	print(params)

# Extract best
obtained_log_posterior = scores[0]
obtained_parameters = params[0]

root = os.path.abspath('cmaes_results')
cmaes_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-cmaes.txt')

with open(cmaes_filename, 'w') as f:
    for x in obtained_parameters:
        f.write(pints.strfloat(x) + '\n')

print ('CMAES fitting is done for model', args.model)
#
# Show result
#
#
# re-create forward model
#
plot = args.plot
print(plot)
if plot:
	root = os.path.abspath('figures/cmaesfit')
	fig_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-cmaes_test.eps')
	model = forwardModel.ForwardModel(protocol, temperature, myo_model, sine_wave=True, logTransform=False)
	
	plt.figure()
	plt.subplot(2,1,1)
	#plt.plot(time, voltage)
	plt.subplot(2,1,2)
	plt.plot(time, current, label='real')
	plt.plot(time, model.simulate(obtained_parameters, time), label='fit')
	plt.legend(loc='lower right')
	plt.savefig(fig_filename)   # save the figure to file
	plt.close()
	

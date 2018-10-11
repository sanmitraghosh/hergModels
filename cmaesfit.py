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
import matplotlib.pyplot as plt
# Load beattie model and prior



# Check input arguments

#
# Select cell
#

parser = argparse.ArgumentParser(description='Fit all the hERG models to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N', \
      help='cell number : 1, 2, ..., 5' )
parser.add_argument('--model', type=int, default=1, metavar='N', \
      help='model number : 1 for C-O-I-IC, 2 for C-O and so on' )
parser.add_argument('--plot', type=bool, default=True, metavar='N', \
      help='plot fitted traces' )
args = parser.parse_args()

#
# Find out the full HH models. NB for these ones our Markov builder doesn't work
#

if args.model == 3 or args.model == 9 or args.model == 19:
	model_name ='model-'+str(args.model)
	root = os.path.abspath('models_myokit')
	model_file = os.path.join(root, 'modelFullHH-'+str(args.model)+'.mmt')
	myo_model = myokit.load_model(model_file)
	sys.path.append(os.path.abspath('models_forward'))
	import pintsForwardModel as forwardModel
	n_params = 4
	print("loading  full HH model: " + str(args.model))
	
else:
	# Import markov models from the models file
	model_name ='model-'+str(args.model)
	root = os.path.abspath('models_myokit')
	sys.path.append(os.path.abspath('models_forward'))
	from models import *
	import pintsForwardModel as forwardModel
	import LogPrior as prior
	model = 'Model'+str(args.model)
	myo_model, rate_dict_maker, n_params = globals()[model]()
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
model = forwardModel.ForwardModel(protocol, temperature, myo_model, n_params, sine_wave=True, logTransform=True)

#
# Define problem
#
problem = pints.SingleOutputProblem(model, time, current)


#
# Define log-posterior
#
log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
log_prior = prior.LogPrior(rate_dict_maker, lower_conductance, n_params, logTransform=True)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)


# Run repeated optimisations
repeats = 1
params, scores = [], []
for i in xrange(repeats):
	# Choose random starting point
	x0 = log_prior.sample()
	print(x0)
	# Create optimiser and log transform parameters
	x0=np.log(x0)
	
	#x0[0],x0[2],x0[4],x0[6] =np.log([x0[0],x0[2],x0[4],x0[6]])
	
	opt = pints.Optimisation(log_posterior, x0, method=pints.CMAES)
	opt.set_max_iterations(None)
	opt.set_parallel(True)

	# Run optimisation
	try:
	    with np.errstate(all='ignore'): # Tell numpy not to issue warnings
		p, s = opt.run()
		p = np.exp(p)
		params.append(p)
		scores.append(s)
		print(p)

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
	model = forwardModel.ForwardModel(protocol, temperature, myo_model, n_params, sine_wave=False, logTransform=False)
	
	plt.figure()
	plt.subplot(2,1,1)
	#plt.plot(time, voltage)
	plt.subplot(2,1,2)
	plt.plot(time, current, label='real')
	plt.plot(time, model.simulate(obtained_parameters, time), label='fit')
	plt.legend(loc='lower right')
	plt.savefig(fig_filename)   # save the figure to file
	plt.close()
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
import matplotlib.pyplot as plt
# Load beattie model and prior



# Check input arguments

#
# Select cell
#

parser = argparse.ArgumentParser(description='Fit all the hERG models to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N', \
      help='cell number : 1, 2, ..., 5' )
parser.add_argument('--model', type=int, default=1, metavar='N', \
      help='model number : 1 for C-O-I-IC, 2 for C-O and so on' )
parser.add_argument('--plot', type=bool, default=True, metavar='N', \
      help='plot fitted traces' )
args = parser.parse_args()

#
# Find out the full HH models. NB for these ones our Markov builder doesn't work
#

if args.model == 3 or args.model == 9 or args.model == 19:
	model_name ='model-'+str(args.model)
	root = os.path.abspath('models_myokit')
	model_file = os.path.join(root, 'modelFullHH-'+str(args.model)+'.mmt')
	myo_model = myokit.load_model(model_file)
	sys.path.append(os.path.abspath('models_forward'))
	import pintsForwardModel as forwardModel
	n_params = 4
	print("loading  full HH model: " + str(args.model))
	
else:
	# Import markov models from the models file
	model_name ='model-'+str(args.model)
	root = os.path.abspath('models_myokit')
	sys.path.append(os.path.abspath('models_forward'))
	from models import *
	import pintsForwardModel as forwardModel
	import LogPrior as prior
	model = 'Model'+str(args.model)
	myo_model, rate_dict_maker, n_params = globals()[model]()
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
model = forwardModel.ForwardModel(protocol, temperature, myo_model, n_params, sine_wave=True, logTransform=True)

#
# Define problem
#
problem = pints.SingleOutputProblem(model, time, current)


#
# Define log-posterior
#
log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
log_prior = prior.LogPrior(rate_dict_maker, lower_conductance, n_params, logTransform=True)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)


# Run repeated optimisations
repeats = 1
params, scores = [], []
for i in xrange(repeats):
	# Choose random starting point
	x0 = log_prior.sample()
	print(x0)
	# Create optimiser and log transform parameters
	x0=np.log(x0)
	
	#x0[0],x0[2],x0[4],x0[6] =np.log([x0[0],x0[2],x0[4],x0[6]])
	
	opt = pints.Optimisation(log_posterior, x0, method=pints.CMAES)
	opt.set_max_iterations(20)
	opt.set_parallel(True)

	# Run optimisation
	try:
	    with np.errstate(all='ignore'): # Tell numpy not to issue warnings
		p, s = opt.run()
		p = np.exp(p)
		params.append(p)
		scores.append(s)
		print(p)

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
	model = forwardModel.ForwardModel(protocol, temperature, myo_model, n_params, sine_wave=False, logTransform=False)
	
	plt.figure()
	plt.subplot(2,1,1)
	#plt.plot(time, voltage)
	plt.subplot(2,1,2)
	plt.plot(time, current, label='real')
	plt.plot(time, model.simulate(obtained_parameters, time), label='fit')
	plt.legend(loc='lower right')
	plt.savefig(fig_filename)   # save the figure to file
	plt.close()
	

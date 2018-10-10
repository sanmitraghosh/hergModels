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
# Load a hERG model and prior

# Check input arguments

parser = argparse.ArgumentParser(description='Fit all the hERG models to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N', \
      help='cell number : 1, 2, ..., 5' )
parser.add_argument('--model', type=int, default=1, metavar='N', \
      help='model number : 1 for C-O-I-IC, 2 for C-O and so on' )
parser.add_argument('--plot', type=bool, default=False, metavar='N', \
      help='plot fitted traces' )
parser.add_argument('--niter', type=int, default=100000, metavar='N', \
      help='number of mcmc iterations' )
parser.add_argument('--burnin', type=int, default=50000, metavar='N', \
      help='number of burn-in samples')
args = parser.parse_args()
sys.path.append(os.path.abspath('models_forward'))

if args.model == 1:	
	import circularCOIIC as forwardModel
	model_name ='model-1'
	print("loading  C-O-I-IC model")
	
elif args.model == 2:
	import linearCOI as forwardModel
	model_name ='model-2'
	print("loading  C-O-I model")

elif args.model == 3:
	import linearCCOI as forwardModel
	print("loading  C-C-O-I model")
	model_name ='model-3'

elif args.model == 4:
	import linearCCCOI as forwardModel
	print("loading  C-C-C-O-I model")
	model_name ='model-4'

elif args.model == 5:
	import circularCCOIICIC as forwardModel
	print("loading  C-C-O-I-IC-IC model")
	model_name ='model-5'

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
model = forwardModel.ForwardModel(protocol, temperature, sine_wave=True, logTransform=False)
#
# Define problem
#
problem = pints.SingleOutputProblem(model, time, current)
#
# Define log-posterior
#
log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
log_prior = forwardModel.LogPrior(lower_conductance, logTransform=False)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)
#
# Run
#
initial_parameters = forwardModel.fetch_parameters()
# Create sampler
nchains = 1
npar = log_prior.n_parameters()
x0 = [initial_parameters] * nchains #+ np.random.uniform(0,1e-5,npar)

mcmc = pints.MCMCSampling(log_posterior, nchains, x0, method=pints.PopulationMCMC)
#mcmc.set_log_to_file('log.txt')
mcmc.set_log_to_screen(False)
iterations = args.niter
mcmc.set_max_iterations(iterations)
mcmc.set_parallel(True)
# Run
with np.errstate(all='ignore'): # Tell numpy not to issue warnings
    
    trace = mcmc.run()
    if nchains > 1:
        print('R-hat:')
        print(pints.rhat_all_params(trace))
# save traces
root = os.path.abspath('mcmc_results')
param_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-mcmc_traces.p')
cPickle.dump(trace, open(param_filename, 'wb')) 

burnin = args.burnin
samples_all_chains = trace[:, burnin:, :]
sample_chain_1 = samples_all_chains[0]


plot = args.plot
print(plot)
# Plot 
if plot:
    
    root = os.path.abspath('figures/mcmc')
    ppc_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-mcmc_ppc.eps')
    pairplt_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-mcmc_pairplt.eps')
    traceplt_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-mcmc_traceplt.eps')
    

    new_values = []
    for ind in range(100):
        ppc_sol=model.simulate(sample_chain_1[ind,:npar], time)
        new_values.append(ppc_sol)
    new_values = np.array(new_values)
    mean_values = np.mean(new_values, axis=0)
    new_values.shape
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(time, voltage, color='orange', label='measured voltage')
    plt.xlim(0,8000)
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(time, current,'--', color='blue',lw=1.5, label='measured current')
    plt.plot(time, mean_values, color='SeaGreen', lw=1, label='mean of inferred current')
    plt.xlim(0,8000)
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(time[-40000:], current[-40000:],'--', color='blue',lw=1.5, label='measured current blow-up')
    plt.plot(time[-40000:], mean_values[-40000:], color='SeaGreen', lw=1, label='mean of inferred current blow-up')
    plt.xlim(4000,8000)
    plt.legend()
    plt.savefig(ppc_filename)   
    plt.close()

    pplot.pairwise(sample_chain_1[:,:npar], opacity=1)
    plt.savefig(pairplt_filename)   
    plt.close()

    pplot.trace(samples_all_chains)
    plt.savefig(traceplt_filename)   
    plt.close()
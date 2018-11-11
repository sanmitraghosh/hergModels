#!/usr/bin/env python2
#
# Fit Kylie's model to Cell 5 data using CMA-ES
#
from __future__ import division, print_function
import models_forward.pintsForwardModel as forwardModel
import models_forward.LogPrior as prior
import models_forward.Rates as Rates
import models_forward.util as util
import os
import sys
import pints
import pints.plot as pplot
import numpy as np
import scipy.stats
from collections import namedtuple
from scipy.special import logsumexp
import warnings
import cPickle
import myokit
import argparse
import matplotlib.pyplot as plt
# Load a hERG model and prior

# Check input arguments
class log_posterior_pointwise(object):
    def __init__(self, _problem, _current, _sigma):
        self._problem = _problem
        self._current = _current
        self._sigma = _sigma
    def __call__(self,x):
        means = self._problem.evaluate(x)
        log_pp = scipy.stats.norm.pdf(self._current,means,self._sigma) + log_prior(x)
        return log_pp

def waic(problem, samples, current, sigma):

    log_pp_obj = log_posterior_pointwise(problem, current, sigma)
    n_workers = pints.ParallelEvaluator.cpu_count()
    evaluator_log_pp = pints.ParallelEvaluator(log_pp_obj, n_workers=n_workers)
    log_pp = evaluator_log_pp.evaluate(samples)
    log_pp = np.asarray(log_pp)
    if log_pp.size == 0:
        raise ValueError('The model does not contain observed values.')

    lppd_i = logsumexp(log_pp, axis=0, b=1.0 / log_pp.shape[0])

    vars_lpd = np.var(log_pp, axis=0)
    warn_mg = 0
    if np.any(vars_lpd > 0.4):
        warnings.warn("""For one or more samples the posterior variance of the
        log predictive densities exceeds 0.4. This could be indication of
        WAIC starting to fail see http://arxiv.org/abs/1507.04544 for details
        """)
        warn_mg = 1

    waic_i = - 2 * (lppd_i - vars_lpd)

    waic_se = np.sqrt(len(waic_i) * np.var(waic_i))

    waic = np.sum(waic_i)

    p_waic = np.sum(vars_lpd)
    pointwise =False
    if pointwise:
        if np.equal(waic, waic_i).all():
            warnings.warn("""The point-wise WAIC is the same with the sum WAIC,
            please double check the Observed RV in your model to make sure it
            returns element-wise logp.
            """)
        WAIC_r = namedtuple('WAIC_r', 'WAIC, WAIC_se, p_WAIC, var_warn, WAIC_i')
        return WAIC_r(waic, waic_se, p_waic, warn_mg, waic_i)
    else:
        WAIC_r = namedtuple('WAIC_r', 'WAIC, WAIC_se, p_WAIC, var_warn')
        return WAIC_r(waic, waic_se, p_waic, warn_mg)
    
def loo(model, samples, current):

    pass

def aic_bic(samples, current):

    max_log_likelihood = np.max(samples[:,-1])

    return 2*npar - 2*max_log_likelihood, np.log(len(current))*npar - 2*max_log_likelihood

def aic_bic_opt(max_log_likelihood, current):

    return 2*npar - 2*max_log_likelihood, np.log(len(current))*npar - 2*max_log_likelihood

def rmse(model, samples, current, time):

    new_values = []
    for ind in xrange(len(samples)):
            ppc_sol=model.simulate(samples[ind,:npar], time)
            new_values.append(ppc_sol)
    new_values = np.array(new_values)
    mean_values = np.mean(new_values, axis=0)
    return np.sqrt(((current - mean_values) ** 2).mean())

def rmse_opt(model, parameters, current, time):

    mean_values = model.simulate(parameters, time)
    return np.sqrt(((current - mean_values) ** 2).mean())
    

def norm_const(samples):

    #ti=np.linspace(0,1,10)**5
    ti = np.array([1.69350878e-05, 5.41922810e-04, 4.11522634e-03,
       1.73415299e-02, 5.29221494e-02, 1.31687243e-01, 2.84628021e-01,
       5.54928957e-01, 1.00000000e+00])
    Eloglike_std = np.mean(samples[:,npar:],axis=0)
    E2loglike_std = np.mean(samples[:,npar:]**2,axis=0)
    Vloglike_std = E2loglike_std - (np.mean(samples[:,npar:],axis=0))**2

    I_MC = 0
    for i in xrange(len(ti)-1):

        I_MC += (Eloglike_std[i] + Eloglike_std[i+1])/2 * (ti[i+1]-ti[i]) \
                - (Vloglike_std[i+1] - Vloglike_std[i])/12 * (ti[i+1]-ti[i])**2
    return I_MC


parser = argparse.ArgumentParser(description='Fit all the hERG models to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N', \
      help='cell number : 1, 2, ..., 5' )
parser.add_argument('--points', type=int, default=5000, metavar='N', \
      help='number of samples to compute WAIC')
parser.add_argument('--mcmc', type=bool, default=False, metavar='N',
                    help='get model stats for mcmc or optimisation')
args = parser.parse_args()
#sys.path.append(os.path.abspath('models_forward'))

cell = args.cell
root = os.path.abspath('sine-wave-data')
print(root)
data_file_sine = os.path.join(root, 'cell-' + str(cell) + '.csv')
protocol_file_sine = os.path.join(root,'steps.mmt')

root = os.path.abspath('ap-data')
print(root)
data_file_ap = os.path.join(root, 'cell-' + str(cell) + '.csv')


protocol_sine = myokit.load_protocol(protocol_file_sine)

log_sine = myokit.DataLog.load_csv(data_file_sine).npview()
time_sine = log_sine.time()
current_sine = log_sine['current']
voltage_sine = log_sine['voltage']
del(log_sine)


log_ap = myokit.DataLog.load_csv(data_file_ap).npview()
time_ap = log_ap.time()
current_ap = log_ap['current']
voltage_ap = log_ap['voltage']
del(log_ap)
protocol_ap = [time_ap, voltage_ap]

model_ppc_tarces = []
#ikr_names = ['Beattie', 'C-O-I','C-C-O-I','C-C-C-O-I']
if args.mcmc:
    model_metrics = np.zeros((30,7))
else:
    model_metrics = np.zeros((30,8))
for i in xrange(30):
    model_name = 'model-'+str(i+1)
    root = os.path.abspath('models_myokit')
    myo_model = os.path.join(root, model_name + '.mmt')
    root = os.path.abspath('rate_dictionaries')
    rate_file = os.path.join(root, model_name + '-priors.p')
    rate_dict = cPickle.load(open(rate_file, 'rb'))
    sys.path.append(os.path.abspath('models_forward'))
    print("loading  model: "+str(i+1))
    model_name = 'model-'+str(i+1)
    temperature = forwardModel.temperature(cell)
    lower_conductance = forwardModel.conductance_limit(cell)
    print('Applying capacitance filtering')
    time, voltage, current = forwardModel.capacitance(
    protocol_sine, 0.1, time_sine, voltage_sine, current_sine)
    sigma_noise_sine = np.std(current_sine[:2000], ddof=1)
    sigma_noise_ap = np.std(current_ap[:2000], ddof=1)
    #
    # Create forward model
    #
    transform = 0
    model = forwardModel.ForwardModel(
        protocol_sine, temperature, myo_model, rate_dict, transform, sine_wave=True)
    model_ap = forwardModel.ForwardModel(
        protocol_ap, temperature, myo_model, rate_dict, transform, sine_wave=False)
    npar = model.n_params
    #
    # Define problem
    #
    problem_sine = pints.SingleOutputProblem(model, time_sine, current_sine)
    problem_ap = pints.SingleOutputProblem(model_ap, time_ap, current_ap)
    #
    # Define log-posterior
    #
    log_likelihood = pints.KnownNoiseLogLikelihood(problem_sine, sigma_noise_sine)
    log_likelihood_ap = pints.KnownNoiseLogLikelihood(problem_ap, sigma_noise_ap)
    log_prior = prior.LogPrior(
        rate_dict, lower_conductance, npar, transform)
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)
    log_posterior_ap = pints.LogPosterior(log_likelihood_ap, log_prior)
    rate_checker = Rates.ratesPrior(transform, lower_conductance)

    
    if args.mcmc:
        model_metrics = np.zeros((5,7))
        root = os.path.abspath('mcmc_results')
        param_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-mcmc_traces.p')
        trace = cPickle.load(open(param_filename, 'rb'))

        
        burnin = 70000
        points = burnin/args.points
        samples_all_chains = trace[:, burnin:, :]
        sample_chain_1 = samples_all_chains[0]
        samples_waic =sample_chain_1[::10,:npar]
        samples_rmse =sample_chain_1[::300,:npar]

        
        waic_train = waic(problem_sine, samples_waic, current_sine, sigma_noise_sine)[0]
        aic_ppc, bic_ppc =aic_bic(sample_chain_1, current_sine)
        log_Z = norm_const(sample_chain_1)
        rmse_sine = rmse(model, samples_rmse, current_sine, time_sine)
        #print(aic_ppc)
        
        waic_test = waic(problem_ap, samples_waic, current_ap, sigma_noise_ap)[0]
        rmse_ap = rmse(model_ap, samples_rmse, current_ap, time_ap)
        #print(bic_ppc)
        model_metrics[i,:] = [ waic_train, aic_ppc, bic_ppc, log_Z, rmse_sine, waic_test, rmse_ap]
    else:
        
        print(model_name+': stats written')
        parameters = forwardModel.fetch_parameters(model_name, cell)
        max_log_likelihood_train = log_likelihood(parameters)
        max_log_likelihood_ap = log_likelihood_ap(parameters)
        rmse_opt_train = rmse_opt(model, parameters, current_sine, time_sine)
        aic_opt_train, bic_opt_train =aic_bic_opt(max_log_likelihood_train, parameters)
        rmse_opt_ap = rmse_opt(model_ap, parameters, current_ap, time_ap)
        aic_opt_ap, bic_opt_ap =aic_bic_opt(max_log_likelihood_ap, parameters)
        model_metrics[i,:] = [ max_log_likelihood_train, aic_opt_train, bic_opt_train, rmse_opt_train, 
                               max_log_likelihood_ap, aic_opt_ap, bic_opt_ap, rmse_opt_ap]
    

outfile = './figures/model_metrics.txt'
np.savetxt(outfile, model_metrics,  fmt='%4.4e', delimiter=',')
    

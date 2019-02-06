import os
import pints
import numpy as np
import scipy.stats as stats

class discrepancyLogLikelihood(pints.ProblemLogLikelihood):
    """
    This class defines a custom loglikelihood which implements a
    discrepancy model where the noise is a time-varying function of
    the voltage and current

    """
    def __init__(self, problem, sigma, model_inputs):
        super(discrepancyLogLikelihood, self).__init__(problem)

        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._n_parameters = self._np + 2
        self._nt = problem.n_times() 
        self.voltage = model_inputs[0]
        self.current = model_inputs[1]
        self._nds = len(model_inputs)

        if np.isscalar(sigma):
            self.sigma = np.ones(self._no) * float(sigma)
        else:
            self.sigma = pints.vector(sigma)
            if len(sigma) != self._no:
                raise ValueError(
                    'Sigma must be a scalar or a vector of length n_outputs.')
        if np.any(self.sigma <= 0):
            raise ValueError('Standard deviation must be greater than zero.')

    def __call__(self, x):
 
        ds_params = x[-self._nds:]
        model_params = x[:-self._nds]

        sigma_t = np.exp(ds_params[0]*self.current  +  ds_params[1]*self.voltage) +  self.sigma        
        return np.sum(stats.norm.logpdf(self._values,loc=self._problem.evaluate(model_params),scale=sigma_t))

    
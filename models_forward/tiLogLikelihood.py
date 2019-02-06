import os
import pints
import numpy as np

class temperedLogLikelihood(pints.ProblemLogLikelihood):
    """
    This class defines a custom loglikelihood which is simply
    t * univariateNormalloglikelihood, where `t` is a temperature
    This implementation is needed to calculate the marginal likelihood 
    using thermodynamic integration

    """
    def __init__(self, problem, sigma, temp):
        super(temperedLogLikelihood, self).__init__(problem)

        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._nt = problem.n_times()
        self.temp = temp
        #self._n_parameters = problem.n_parameters()

        if np.isscalar(sigma):
            self.sigma = np.ones(self._no) * float(sigma)
        else:
            self.sigma = pints.vector(sigma)
            if len(sigma) != self._no:
                raise ValueError(
                    'Sigma must be a scalar or a vector of length n_outputs.')
        if np.any(self.sigma <= 0):
            raise ValueError('Standard deviation must be greater than zero.')

        self._offset = -0.5 * self._nt * np.log(2 * np.pi)
        self._offset -= self._nt * np.log(sigma)
        self._multip = -1 / (2.0 * sigma**2)        
    def __call__(self, x):
        
        error = self._values - self._problem.evaluate(x)
        return self.temp * np.sum(self._offset + self._multip * np.sum(error**2, axis=0))

    
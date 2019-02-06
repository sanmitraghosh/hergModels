import os
import pints
import numpy as np
import scipy.stats

class HalfCauchyLogPrior(pints.LogPrior):
    def __init__(self, location, scale):
        super(HalfCauchyLogPrior, self).__init__()
        # Test inputs
        if float(scale) <= 0:
            raise ValueError('Scale must be positive')
        self._location = float(location)
        self._scale = float(scale)
        self.n_params = 1

    def n_parameters(self):
        return self.n_params
    def __call__(self, x):
        return scipy.stats.halfcauchy.logpdf(x, self._location, self._scale) 
    def sample(self) :
        return np.array(scipy.stats.halfcauchy(self._location, self._scale).rvs(1)).reshape((1,1))

class halfNormalLogPrior(pints.LogPrior):
    def __init__(self, sd):
        super(halfNormalLogPrior, self).__init__()
        # Test inputs
        if float(sd) <= 0:
            raise ValueError('Scale must be positive')
        self._sd = float(sd)
        self.n_params = 1

    def n_parameters(self):
        return self.n_params
    def __call__(self, x):
        return scipy.stats.halfnorm.logpdf(x, scale=self._sd) 
    def sample(self) :
        return np.array(scipy.stats.halfnorm(scale=self._sd).rvs(1)).reshape((1,1))
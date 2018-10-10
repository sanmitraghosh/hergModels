#!/usr/bin/env python2
#
# Pints ForwardModel that runs simulations with Kylie's model.
# Sine waves optional
#
from __future__ import division
from __future__ import print_function
import os
import pints
import numpy as np
from Rates import ratesPrior

class LogPrior(pints.LogPrior):
    """
    Unnormalised prior with constraint on the rate constants.
    """
    def __init__(self, rate_dict_maker, lower_conductance, n_params, logTransform = False):
        super(LogPrior, self).__init__()
        
        self.lower_conductance = lower_conductance
        self.upper_conductance = 10 * lower_conductance
        self.minf = -float(np.inf)
        self.rate_dict_maker = rate_dict_maker
        self.n_params = n_params
        if logTransform:
            self.logParam = True
        else:
            self.logParam = False
    def n_parameters(self):
        return self.n_params

    def _get_rates(self, parameters):
        return self.rate_dict_maker(parameters)

    def __call__(self, parameters):
        if self.logParam:
            parameters = np.exp(parameters)

        if parameters[-1] < self.lower_conductance:
            return self.minf
        if parameters[-1] < self.upper_conductance:
            return self.minf

        rate_dict = self._get_rates(parameters)

        rate_checker = ratesPrior(self.lower_conductance)
        return rate_checker.check_rates(rate_dict)

    def sample(self):

        rate_checker = ratesPrior(self.lower_conductance)
        # dummy parameters passed to highlight rate directions (Fw/Bw)
        rate_dict = self._get_rates(np.zeros(self.n_params))
        
        params = rate_checker._sample_partial(rate_dict)

        return params
              

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
import json
import operator
import util
from Rates import ratesPrior


class LogPrior(pints.LogPrior):
    """
    Unnormalised prior with constraint on the rate constants.
    """

    def __init__(self, rate_dict, lower_conductance, n_params,  transform_type):
        super(LogPrior, self).__init__()

        self.lower_conductance = lower_conductance
        self.upper_conductance = 10 * lower_conductance
        self.minf = -float(np.inf)
        # sorted_rate_dict = sorted(d.items(), key=lambda x: x[1]) #sorted(rate_dict.items(), key=operator.itemgetter(0))
        # sorted_rate_dict
        self.rate_dict = rate_dict
        self.n_params = n_params
        self.transform_type = transform_type

    def n_parameters(self):
        return self.n_params

    """
    def _get_rates(self, parameters):
        i = 0
        for _, rate in self.rate_dict.iteritems():
            if rate[2] == 'vol_ind':
                rate[0] = parameters[i]
                i += 1

            elif rate[2] == 'positive' or rate[2] == 'negative':
                rate[0] = parameters[i]
                rate[1] = parameters[i+1]
                i += 2
        
        return self.rate_dict
    """

    def __call__(self, parameters):
        params = util.transformer(
            self.transform_type, parameters, self.rate_dict, False)

        if params[-1] < self.lower_conductance:
            return self.minf
        if params[-1] > self.upper_conductance:
            return self.minf

        #rate_dict = self._get_rates(params)

        rate_checker = ratesPrior(self.transform_type, self.lower_conductance)
        return rate_checker.check_rates(self.rate_dict, params)

    def sample(self):

        rate_checker = ratesPrior(self.transform_type, self.lower_conductance)
        # dummy parameters passed to highlight rate directions (Fw/Bw)
        #rate_dict = self._get_rates(np.zeros(self.n_params-1))

        params = rate_checker._sample_partial(self.rate_dict)

        return params

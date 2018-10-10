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

class ratesPrior(object):
    """
    Unnormalised prior with constraint on the rate constants.
    """
    def __init__(self, lower_conductance):
        

        self.lower_conductance = lower_conductance
        self.upper_conductance = 10 * lower_conductance

        self.lower_alpha = 1e-7              # Kylie: 1e-7
        self.upper_alpha = 1e3               # Kylie: 1e3
        self.lower_beta  = 1e-7              # Kylie: 1e-7
        self.upper_beta  = 0.4               # Kylie: 0.4

        self.minf = -float(np.inf)

        self.rmin = 1.67e-5
        self.rmax = 1000

        self.vmin = -120
        self.vmax =  60

    def check_rates(self, rates_dict):

        debug = False
        # Check parameter boundaries
        for names, rate in rates_dict.iteritems():

            if rate[2] == True:

                if rate[0] < self.lower_alpha:
                    if debug: print('Lower')
                    return self.minf

                if rate[1] > self.upper_alpha:
                    if debug: print('Lower')
                    return self.minf

                r = rate[0] * np.exp(rate[1] * self.vmax)
                if r < self.rmin or r > self.rmax:
                    if debug: print(names)
                    return self.minf

            elif rate[2] == False:

                if rate[0] < self.lower_beta:
                    if debug: print('Lower')
                    return self.minf

                if rate[1] > self.upper_beta:
                    if debug: print('Lower')
                    return self.minf

                r = rate[0] * np.exp(-rate[1] * self.vmin)
                if r < self.rmin or r > self.rmax:
                    if debug: print(names)
                    return self.minf
        return 0

    def _sample_rates(self, v):
        for i in xrange(100):
            a = np.exp(np.random.uniform(
                np.log(self.lower_alpha), np.log(self.upper_alpha)))
            b = np.random.uniform(self.lower_beta, self.upper_beta)
            r = a * np.exp(b * v)
            if r >= self.rmin and r <= self.rmax:
                return a, b
        raise ValueError('Too many iterations')

    def _sample_partial(self, rates_dict):

        p = np.zeros(2*len(rates_dict)+1)
        i = 0
        for _, rate in rates_dict.iteritems():
            # Sample forward rates
            if rate[2] == True:
                p[i], p[i+1] = self._sample_rates(self.vmax)
                i += 2
            # Sample backward rates
            else:
                p[i], p[i+1] = self._sample_rates(-self.vmin)
                i += 2
        # Sample conductance
        p[-1] = np.random.uniform(
            self.lower_conductance, self.upper_conductance)
        print(p)

        # Return
        return p
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
import operator
import util
from Rates import ratesPrior


class tiLogLikelihood(pints.ProblemLogLikelihood):
    """
    Unnormalised prior with constraint on the rate constants.
    """

    def __init__(self, problem, sigma, temperature):
        super(tiLogLikelihood, self).__init__(problem)

        # Store counts
        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._nt = problem.n_times()
        self.temperature = temperature
        self._n_parameters = problem.n_parameters() + 1

        # Check sigma
        if np.isscalar(sigma):
            self.sigma = np.ones(self._no) * float(sigma)
        else:
            self.sigma = pints.vector(sigma)
            if len(sigma) != self._no:
                raise ValueError(
                    'Sigma must be a scalar or a vector of length n_outputs.')
        if np.any(self.sigma <= 0):
            raise ValueError('Standard deviation must be greater than zero.')

        # Pre-calculate parts
        
        
        # Pre-calculate S1 parts
        self._isigma2 = sigma**-2

    def __call__(self, x):
        dicrp = x[-1] * self.temperature 
        sigma = self.sigma + dicrp
        error = self._values - (self._problem.evaluate(x) + dicrp)
        offset = -0.5 * self._nt * np.log(2 * np.pi)
        offset -= self._nt * np.log(sigma + dicrp)
        multip = -1 / (2.0 * sigma**2)

        return self.temperature * np.sum(offset + multip * np.sum(error**2, axis=0))

    
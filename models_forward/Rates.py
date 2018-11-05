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

    def __init__(self, transform, lower_conductance):

        self.lower_conductance = lower_conductance
        self.upper_conductance = 10 * lower_conductance
        self.transform = transform

        self.lower_alpha = 1e-7              # Kylie: 1e-7
        self.upper_alpha = 1e3               # Kylie: 1e3
        self.lower_beta = 1e-7              # Kylie: 1e-7
        self.upper_beta = 0.4               # Kylie: 0.4

        self.minf = -float(np.inf)

        self.rmin = 1.67e-5
        self.rmax = 1000

        self.vmin = -120
        self.vmax = 60
        n = 1e3
        a = np.exp(np.linspace(np.log(self.lower_alpha),
                               np.log(self.upper_alpha), n))
        f_bmin = (1 / self.vmax) * (np.log(self.rmin) - np.log(a))
        f_bmax = (1 / self.vmax) * (np.log(self.rmax) - np.log(a))
        self.f_bmin = np.maximum(f_bmin, self.lower_beta)
        self.f_bmax = np.minimum(f_bmax, self.upper_beta)

        r_bmin = (-1 / self.vmin) * (np.log(self.rmin) - np.log(a))
        r_bmax = (-1 / self.vmin) * (np.log(self.rmax) - np.log(a))
        self.r_bmin = np.maximum(r_bmin, self.lower_beta)
        self.r_bmax = np.minimum(r_bmax, self.upper_beta)
        # self.r_bmax[0] =  self.upper_beta

    def _get_boundaries(self, rates_dict):
        b_low = []  # np.zeros(2*len(rates_dict)+1)
        b_up = []
        for _, rate in rates_dict.iteritems():
            # Sample forward rates
            if self.transform == 0:
                if rate[2] == 'vol_ind':
                    b_low.append(self.lower_alpha)
                    b_up.append(self.upper_alpha)
                elif rate[2] == 'positive':
                    b_low.append(self.lower_alpha)
                    b_low.append(self.lower_beta)
                    b_up.append(self.upper_alpha)
                    b_up.append(self.f_bmax[0])
                elif rate[2] == 'negative':
                    b_low.append(self.lower_alpha)
                    b_low.append(self.lower_beta)
                    b_up.append(self.upper_alpha)
                    b_up.append(self.r_bmax[0])

            elif self.transform == 1:
                if rate[2] == 'vol_ind':
                    b_low.append(np.log(self.lower_alpha))
                    b_up.append(np.log(self.upper_alpha))
                elif rate[2] == 'positive':
                    b_low.append(np.log(self.lower_alpha))
                    b_low.append(self.lower_beta)
                    b_up.append(np.log(self.upper_alpha))
                    b_up.append(self.f_bmax[0])
                elif rate[2] == 'negative':
                    b_low.append(np.log(self.lower_alpha))
                    b_low.append(self.lower_beta)
                    b_up.append(np.log(self.upper_alpha))
                    b_up.append(self.r_bmax[0])

            elif self.transform == 2:
                if rate[2] == 'vol_ind':
                    b_low.append(np.log(self.lower_alpha))
                    b_up.append(np.log(self.upper_alpha))
                elif rate[2] == 'positive':
                    b_low.append(np.log(self.lower_alpha))
                    b_low.append(np.log(self.lower_beta))
                    b_up.append(np.log(self.upper_alpha))
                    b_up.append(np.log(self.f_bmax[0]))
                elif rate[2] == 'negative':
                    b_low.append(np.log(self.lower_alpha))
                    b_low.append(np.log(self.lower_beta))
                    b_up.append(np.log(self.upper_alpha))
                    b_up.append(np.log(self.r_bmax[0]))

        if self.transform <= 1:
            b_low.append(self.lower_conductance)
            b_up.append(self.upper_conductance)
        elif self.transform >= 2:
            b_low.append(np.log(self.lower_conductance))
            b_up.append(np.log(self.upper_conductance))

        return [np.array(b_low), np.array(b_up)]

    def check_rates(self, rates_dict, parameters):
        # Always operates on the model (untransformed) parameters.
        debug = True
        # Check parameter boundaries
        for names, rate in rates_dict.iteritems():

            if parameters[rate[0]] < self.lower_alpha:
                if debug:
                    print('Lower_alpha')
                return self.minf

            if parameters[rate[0]] > self.upper_alpha:
                if debug:
                    print('Upper_alpha')
                return self.minf

            if rate[2] == 'vol_ind':

                r = parameters[rate[0]]
                if r < self.rmin or r > self.rmax:
                    if debug:
                        print(names, 'vol_ind')
                    return self.minf

            elif rate[2] == 'positive':
                if parameters[rate[1]] < self.lower_beta:
                    if debug:
                        print('Lower_beta_pos')
                    return self.minf

                if parameters[rate[1]] > self.f_bmax[0]:
                    if debug:
                        print('Upper_beta_pos')
                    return self.minf

                r = parameters[rate[0]] * \
                    np.exp(parameters[rate[1]] * self.vmax)
                if r < self.rmin or r > self.rmax:
                    if debug:
                        print(names, 'pos')
                    return self.minf

            elif rate[2] == 'negative':
                if parameters[rate[1]] < self.lower_beta:
                    if debug:
                        print('Lower_beta_neg')
                    return self.minf

                if parameters[rate[1]] > self.r_bmax[0]:
                    if debug:
                        print('Upper_beta_neg')
                    return self.minf
                r = parameters[rate[0]] * \
                    np.exp(-parameters[rate[1]] * self.vmin)
                if r < self.rmin or r > self.rmax:
                    if debug:
                        print(names, 'neg')
                    return self.minf

        return 0

    def _sample_rates(self, v, rate_type):

        while True:
            if self.transform == 0:
                a = np.random.uniform(self.lower_alpha, self.upper_alpha)
            elif self.transform >= 1:
                a = np.exp(np.random.uniform(
                    np.log(self.lower_alpha), np.log(self.upper_alpha)))

            if rate_type == 'positive' or rate_type == 'negative':
                if rate_type == 'positive':
                    if self.transform <= 1:
                        b = np.random.uniform(self.lower_beta, self.f_bmax[0])
                    elif self.transform == 2:
                        b = np.exp(np.random.uniform(
                            np.log(self.lower_beta), np.log(self.f_bmax[0])))
                    else:
                        Exception('Unrecognised transform')
                    r = a * np.exp(b * v)
                elif rate_type == 'negative':
                    if self.transform <= 1:
                        b = np.random.uniform(self.lower_beta, self.r_bmax[0])
                    elif self.transform == 2:
                        b = np.exp(np.random.uniform(
                            np.log(self.lower_beta), np.log(self.r_bmax[0])))
                    else:
                        Exception('Unrecognised transform')
                    r = a * np.exp(-b * v)
                if r > self.rmin and r < self.rmax:
                    return a, b

            elif rate_type == 'vol_ind':
                r = a
                if r > self.rmin and r < self.rmax:
                    return a

        raise ValueError('Too many iterations')

    def _sample_partial(self, rates_dict):

        p = []  # np.zeros(2*len(rates_dict)+1)

        for _, rate in rates_dict.iteritems():
            # Sample forward rates

            if rate[2] == 'vol_ind':
                p.append(self._sample_rates(self.vmax, rate[2]))

            elif rate[2] == 'positive':
                a, b = self._sample_rates(self.vmax, rate[2])
                p.append(a)
                p.append(b)

            # Sample backward rates
            elif rate[2] == 'negative':
                a, b = self._sample_rates(self.vmin, rate[2])
                p.append(a)
                p.append(b)
            else:
                Exception(
                    'Unrecognised rate type - should be ''positive'', ''negative'', or ''vol_ind''.')

        # Sample conductance - linear for transforms=0 or 1, log for transform=2
        if self.transform <= 1:
            p.append(np.random.uniform(
                self.lower_conductance, self.upper_conductance))
        elif self.transform == 2:
            p.append(np.exp(np.random.uniform(
                np.log(self.lower_conductance), np.log(self.upper_conductance))))
        else:
            Exception('Unrecognised transform')

        p = np.asarray(p)
        print(p)

        # Return
        return p

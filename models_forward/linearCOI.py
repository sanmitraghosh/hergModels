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
import myokit
import myokit.pacing as pacing
#from util import erev, Pr1Error, conductance_limit, capacitance, fold_plot
import util

# Beattie model in Myokit
root = os.path.abspath('models_myokit')
model_file = os.path.join(root, 'linearCOI.mmt')


class ForwardModel(pints.ForwardModel):
    """
    Pints ForwardModel that runs simulations with Kylie's model.
    Sine waves or data protocol optional.

    Arguments:

        ``protocol``
            A myokit.Protocol or a tuple (times, voltage)
        ``temperature``
            The temperature in deg C
        ``sine_wave``
            Set to True if sine-wave protocol is being used.

    """
    parameters = [
        'ikr.p1',
        'ikr.p2',
        'ikr.p3',
        'ikr.p4',
        'ikr.p5',
        'ikr.p6',
        'ikr.p7',
        'ikr.p8',
        'ikr.p9',
    ]

    def __init__(self, protocol, temperature, sine_wave=False, logTransform=False):

        # Load model
        model = myokit.load_model(model_file)

        # Set reversal potential
        model.get('nernst.EK').set_rhs(erev(temperature))

        # Add sine-wave equation to model
        if sine_wave:
            model.get('membrane.V').set_rhs(
                'if(engine.time >= 3000.1 and engine.time < 6500.1,'
                + ' - 30'
                + ' + 54 * sin(0.007 * (engine.time - 2500.1))'
                + ' + 26 * sin(0.037 * (engine.time - 2500.1))'
                + ' + 10 * sin(0.190 * (engine.time - 2500.1))'
                + ', engine.pace)')

        # Create simulation
        self.simulation = myokit.Simulation(model)

        # Add protocol
        if isinstance(protocol, myokit.Protocol):
            self.simulation.set_protocol(protocol)
        else:
            # Apply data-clamp
            times, voltage = protocol
            self.simulation.set_fixed_form_protocol(times, voltage)

            # Set max step size
            self.simulation.set_max_step_size(0.1)

        # Set solver tolerances to values used by Kylie
        self.simulation.set_tolerance(1e-8, 1e-8)

	# Don't log transform params unless specified
	if logTransform:
		self.logParam = True
	else:
		self.logParam = False

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters)`. """
        return len(self.parameters)

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.outputs()`. """
        return 1

    def simulate(self, parameters, times):

        # Note: Kylie doesn't do pre-pacing!
        if self.logParam:
            parameters = np.exp(parameters)
            # Update model parameters
            for i, name in enumerate(self.parameters):
                self.simulation.set_constant(name, parameters[i])

            # Run
            self.simulation.reset()
            try:
                d = self.simulation.run(
                    np.max(times + 0.5 * times[1]),
                    log_times = times,
                    log = ['engine.time', 'ikr.IKr', 'membrane.V'],
                    ).npview()
            except myokit.SimulationError:
                return times * float('inf')

            # Store membrane potential for debugging
            self.simulated_v = d['membrane.V']

            # Return
            return d['ikr.IKr']

class LogPrior(pints.LogPrior):
    """
    Unnormalised prior with constraint on the rate constants.
    """
    def __init__(self, lower_conductance, logTransform=False):
        super(LogPrior, self).__init__()

        self.lower_conductance = lower_conductance
        self.upper_conductance = 10 * lower_conductance

        self.lower_alpha = 1e-7              # Kylie: 1e-7
        self.upper_alpha = 1e3               # Kylie: 1e3
        self.lower_beta  = 1e-7              # Kylie: 1e-7
        self.upper_beta  = 0.4               # Kylie: 0.4

        self.lower = np.array([
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_conductance,
        ])
        self.upper = np.array([
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_conductance,
        ])

        self.minf = -float(np.inf)

        self.rmin = 1.67e-5
        self.rmax = 1000

        self.vmin = -120
        self.vmax =  60

	# Don't log transform params unless specified
	if logTransform:
		self.logParam = True
	else:
		self.logParam = False
    def n_parameters(self):
        return 9

    def __call__(self, parameters):

        debug = False
	if self.logParam:
		parameters = np.exp(parameters)
        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return self.minf

        # Calculate area of forward and reverse priors
        n = 1e3
        a = np.linspace(self.lower_alpha, self.upper_alpha, n)
        f_bmin = (1 / self.vmax) * (np.log(self.rmin) - np.log(a))
        f_bmax = (1 / self.vmax) * (np.log(self.rmax) - np.log(a))
        f_bmin = np.maximum(f_bmin, self.lower_beta)
        f_bmax = np.minimum(f_bmax, self.upper_beta)

        r_bmin = (-1 / self.vmin) * (np.log(self.rmin) - np.log(a))
        r_bmax = (-1 / self.vmin) * (np.log(self.rmax) - np.log(a))
        r_bmin = np.maximum(r_bmin, self.lower_beta)
        r_bmax = np.minimum(r_bmax, self.upper_beta)       

        adiff = a[1:] - a[:-1]
        f_bdiff = (f_bmax - f_bmin)[:-1]
        r_bdiff = (r_bmax - r_bmin)[:-1]
        f_area = np.sum(f_bdiff * adiff)
        r_area = np.sum(r_bdiff * adiff)

        # Check rate constant boundaries
        p1, p2, p3, p4, p5, p6, p7, p8, p9 = parameters

        # Check forward rates
        r = p1 * np.exp(p2 * self.vmax)  
        if r < self.rmin or r > self.rmax:
            if debug: print('r1')
            return self.minf
        r = p5 * np.exp(p6 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug: print('r3')
            return self.minf
        f_lp = 2*np.log(1/f_area)

        # Check backward rates
        r = p3 * np.exp(-p4 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug: print('r2')
            return self.minf
        r = p7 * np.exp(-p8 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug: print('r4')
            return self.minf
        r_lp = 2*np.log(1/r_area)

        return f_lp + r_lp

    def _sample_partial(self, v):
        for i in xrange(100):
            a = np.exp(np.random.uniform(
                np.log(self.lower_alpha), np.log(self.upper_alpha)))
            b = np.random.uniform(self.lower_beta, self.upper_beta)
            r = a * np.exp(b * v)
            if r >= self.rmin and r <= self.rmax:
                return a, b
        raise ValueError('Too many iterations')


    def sample(self):

        p = np.zeros(9)

        # Sample forward rates
        p[0:2] = self._sample_partial(self.vmax)
        p[4:6] = self._sample_partial(self.vmax)

        # Sample backward rates
        p[2:4] = self._sample_partial(-self.vmin)
        p[6:8] = self._sample_partial(-self.vmin)

        # Sample conductance
        p[8] = np.random.uniform(
            self.lower_conductance, self.upper_conductance)

        # Return
        return p

class Pr1Error(pints.ErrorMeasure):
    def __init__(self, problem):
        self._problem = problem

    def dimension(self):
        return self._model.dimension()

    def __call__(self, x):
        # Run simulation
        values = self._problem.evaluate()

        # Extract points of interest
        raise NotImplementedError

def fetch_parameters(MCMC=None):
	root = os.path.abspath('cmaes_results')
	
	filename = os.path.join(root,'model-2-cell-5-cmaes.txt')
	print(filename)
	with open(filename, 'r') as f:
		obtained_parameters = [float(x) for x in f.readlines()]
	return obtained_parameters

def erev(temperature):
	return util.erev(temperature)

def conductance_limit(cell):
	return util.conductance_limit(cell)

def temperature(cell):
	return util.temperature(cell)

def capacitance(protocol, dt, *signals):
	return util.capacitance(protocol, dt, *signals)

def fold_plot(protocol, time, voltage, currents, labels=None):
	return util.fold_plot(protocol, time, voltage, currents, labels=None)





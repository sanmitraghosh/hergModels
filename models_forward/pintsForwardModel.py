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
import util



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
    
    def __init__(self, protocol, temperature, myo_model, n_params, sine_wave=False, logTransform=False):

        # Load model
        model = myo_model

        # Set reversal potential
        model.get('ikr.E').set_rhs(erev(temperature))

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

        self.model = model
        self.n_params = int(n_params)
  
    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters)`. """
        return self.n_params

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.outputs()`. """
        return 1

    def simulate(self, parameters, times):
        # Note: Kylie doesn't do pre-pacing!
        if self.logParam:
            parameters = np.exp(parameters)
            
        # Update model parameters
        for i in xrange(int(self.n_params)):

            self.simulation.set_constant('ikr.p'+str(i+1), parameters[i])
        
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


def erev(temperature):
	return util.erev(temperature)

def conductance_limit(cell):
	return util.conductance_limit(cell)

def temperature(cell):
	return util.temperature(cell)

def capacitance(protocol, dt, *signals):
	return util.capacitance(protocol, dt, *signals)

def fetch_parameters(model_name, MCMC=None):
	root = os.path.abspath('cmaes_results')
	
	filename = os.path.join(root,model_name+'-cell-5-cmaes.txt')
	print(filename)
	with open(filename, 'r') as f:
		obtained_parameters = [float(x) for x in f.readlines()]
	return obtained_parameters
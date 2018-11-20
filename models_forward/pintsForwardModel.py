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

    def __init__(self, protocol, temperature, myo_model, rate_dict, transform_type, sine_wave=0):

        # Load model
        model = myokit.load_model(myo_model)
        n_params = int(model.get('ikr.n_params').value())
        parameters = np.zeros(n_params)
        for i in xrange(n_params):
            parameters[i] = model.get('ikr.p'+str(i+1)).value()

        self.parameters = parameters
        self.n_params = len(parameters)
        self.func_call = 0
        self.rate_dict = rate_dict

        # Set reversal potential
        model.get('nernst.EK').set_rhs(erev(temperature))

        # Add sine-wave equation to model
        if sine_wave==1:
            model.get('membrane.V').set_rhs(
                'if(engine.time >= 3000.1 and engine.time < 6500.1,'
                + ' - 30'
                + ' + 54 * sin(0.007 * (engine.time - 2500.1))'
                + ' + 26 * sin(0.037 * (engine.time - 2500.1))'
                + ' + 10 * sin(0.190 * (engine.time - 2500.1))'
                + ', engine.pace)')
        #elif sine_wave==2:
        #    model.get('membrane.V').set_rhs(
        #        'if(engine.time >= 3000.1 and engine.time < 8000.1,'
        #        + ' + 57 * sin(0.195* (engine.time - 2500.1))'
        #        + ' + 28 * sin(0.503 * (engine.time - 2500.1))'
        #        + ' + 18 * sin(0.7037 * (engine.time - 2500.1))'
        #        + ', engine.pace)')

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

        # Store the parameter transform type 0=none, 1=log(A)Linear(B), 2= log everything
        self.transform_type = transform_type

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
        self.func_call += 1

        parameters = util.transformer(
            self.transform_type, parameters, self.rate_dict, False)

        # Update model parameters
        for i in xrange(int(self.n_params)):
            self.simulation.set_constant('ikr.p'+str(i+1), parameters[i])

         # Run
        self.simulation.reset()
        try:
            d = self.simulation.run(
                np.max(times + 0.5 * times[1]),
                log_times=times,
                log=['engine.time', 'ikr.IKr', 'membrane.V','ikr.O']
                #log = ['engine.time', 'ikr.IKr', 'membrane.V', 'ikr.m_inf', 'ikr.h_inf'],
            ).npview()
        except myokit.SimulationError as e:
            print('Myokit error: ',e)
            return times * float('inf')

        # Store membrane potential for debugging
        self.simulated_v = d['membrane.V']
        self.simulated_o = d['ikr.O']
        #self.simulated_minf = d['ikr.m_inf']
        #self.simulated_hinf = d['ikr.h_inf']

        # Return
        """
        counter = self.simulation.last_number_of_evaluations()
        outfile = 'func_calls.txt'
        func_calls = d['ikr.counter']
        np.savetxt(outfile, func_calls)

        """
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


def fetch_parameters(model_name, cell, MCMC=None):
    root = os.path.abspath('cmaes_results')

    filename = os.path.join(
        root, model_name+'-cell-' + str(cell) + '-cmaes.txt')
    print(filename)
    with open(filename, 'r') as f:
        obtained_parameters = [float(x) for x in f.readlines()]
    return obtained_parameters

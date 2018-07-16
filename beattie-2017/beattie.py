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


# Beattie model in Myokit
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_file = os.path.join(root, 'models', 'beattie-2017-ikr-markov.mmt')


def erev(temperature):
    T = 273.15 + temperature
    F = 96485.0
    R = 8314.0
    K_i = 130.0
    k_o = 4.0
    return ((R*T)/F) * np.log(k_o/K_i)


class BeattieModel(pints.ForwardModel):
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

    def __init__(self, protocol, temperature, sine_wave=False):

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

    def dimension(self):
        return len(self.parameters)

    def simulate(self, parameters, times):
        # Note: Kylie doesn't do pre-pacing!

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


class BeattieLogPrior(pints.LogPrior):
    """
    Unnormalised prior with constraint on the rate constants.
    """
    def __init__(self, lower_conductance):
        super(BeattieLogPrior, self).__init__()

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

        self.minf = -float('inf')

        self.rmin = 1.67e-5
        self.rmax = 1000

        self.vmin = -120
        self.vmax =  60


    def dimension(self):
        return 9

    def __call__(self, parameters):

        debug = False

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return self.minf

        # Check rate constant boundaries
        p1, p2, p3, p4, p5, p6, p7, p8, p9 = parameters

        # Check forward rates
        r = p1 * np.exp(p2 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug: print('r1')
            return self.minf
        r = p5 * np.exp(p6 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug: print('r2')
            return self.minf

        # Check backward rates
        r = p3 * np.exp(-p4 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug: print('r3')
            return self.minf
        r = p7 * np.exp(-p8 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug: print('r4')
            return self.minf

        return 0

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


def conductance_limit(cell):
    """
    Returns a lower limit for the conductance of the cell with the
    given integer index ``cell``.
    """
    #
    # Guesses for lower conductance
    #
    lower_conductances = {
        5: 0.0612,  # 16713110
    }
    #if strcmp(exp_ref,'16708118')==1    lower_conductance = 0.0170;
    #if strcmp(exp_ref,'16704047')==1    lower_conductance = 0.0434;
    #if strcmp(exp_ref,'16704007')==1    lower_conductance = 0.0886;
    #if strcmp(exp_ref,'16707014')==1    lower_conductance = 0.0203;
    #if strcmp(exp_ref,'16708060')==1    lower_conductance = 0.0305;
    #if strcmp(exp_ref,'16708016')==1    lower_conductance = 0.0417;
    #if strcmp(exp_ref,'16713003')==1    lower_conductance = 0.0478;
    #if strcmp(exp_ref,'16715049')==1    lower_conductance = 0.0255;
    #if strcmp(exp_ref,'average')==1     lower_conductance = 0.0410;

    return lower_conductances[cell]


def temperature(cell):
    """
    Returns the temperature (in degrees Celsius) for the given integer index
    ``cell``.
    """
    temperatures = {
        5 : 21.4,   # 16713110
    }
    #if strcmp(exp_ref,'16708016')==1    temperature = 21.8;
    #if strcmp(exp_ref,'16708060')==1    temperature = 21.7;
    #if strcmp(exp_ref,'16704047')==1    temperature = 21.6;
    #if strcmp(exp_ref,'16704007')==1    temperature = 21.2;
    #if strcmp(exp_ref,'16713003')==1    temperature = 21.3;
    #if strcmp(exp_ref,'16715049')==1    temperature = 21.4;
    #if strcmp(exp_ref,'16707014')==1    temperature = 21.4;
    #if strcmp(exp_ref,'16708118')==1    temperature = 21.7;

    return temperatures[cell]


def capacitance(protocol, dt, *signals):
    """
    Create and apply a capacitance filter
    """
    cap_duration = 5    # Same as Kylie
    fcap = np.ones(len(signals[0]), dtype=int)
    steps = [step for step in protocol]
    for step in steps[1:]:
        i1 = int(step.start() / dt)
        i2 = i1 + int(cap_duration / dt)
        fcap[i1:i2] = 0
    fcap = fcap > 0

    # Apply filter
    return [x[fcap] for x in signals]


def fold_plot(protocol, time, voltage, currents, labels=None):
    """
    Create a folded plot of a given protocol
    """
    # Points to split signal at
    splits = {
        'pr1-activation-kinetics-1' : [
            # Numbers 2, 3, and 4 are a bit off for some reason
            0, 51770, 103519, 155269, 207019, 258770, 310520,
        ],
        'pr2-activation-kinetics-2' : [
            0, 51770, 103519, 155269, 207019, 258770, 310520,
        ],
        'pr3-steady-activation' : [
            0, 82280, 164609, 246889, 329169, 411449, 493729, 576010,
        ],
        'pr4-inactivation' : [
            0, 28657, 57363, 86019, 114674, 143331, 171987, 200642, 229299,
            257955, 286611, 315267, 343922, 372578, 401235, 429891, 458546,
        ],
        'pr5-deactivation' : [
            0, 102974, 205897, 308822, 411746, 514670, 617593, 720518, 823442,
            926366,
        ],
    }

    # Find split points
    try:
        split = splits[protocol]
        split = zip(split[:-1], split[1:])
    except KeyError:
        # Try to auto-detect!
        repeats = {
            'pr1-activation-kinetics-1' : 6,
            'pr2-activation-kinetics-2' : 6,
            'pr3-steady-activation' : 7,
            'pr4-inactivation' : 16,
            'pr5-deactivation' : 9,
        }
        period = (time[-1] + 0.1) / repeats[protocol]
        print('Period: ' + str(period))

        print('Guessed splits:')
        splits = [int(i * period * 10) for i in range(repeats)]
        splits.append(len(time))
        splits = zip(splits[:-1], splits[1:])
        first = np.where(voltage != voltage[0])[0][0]
        for lower, upper in splits:
            step = np.where(
                np.abs(voltage[lower:upper] - voltage[lower]) > 10)[0][0]
            print(lower + step - first)
        print(len(time))

        # Will need customisation, so halt!
        raise

    # Define zoom points
    zoom = {
        #'pr1-activation-kinetics-1' : (),
        #'pr2-activation-kinetics-2' : (),
        'pr3-steady-activation' : ((500, 6500), (-0.1, 1.75)),
        'pr4-inactivation' : ((1180, 1500), (-3.2, 6.5)),
        'pr5-deactivation' : ((2300, 8000), (-4, 2)),
    }

    # Load matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec

    # Create colormap
    #cmap = matplotlib.cm.get_cmap('Dark2')
    cmap = matplotlib.cm.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(0, len(split) - 1)

    # Create plot
    plt.figure()
    matplotlib.gridspec.GridSpec(4, 1)

    plt.subplot2grid((4, 1), (0, 0))
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')
    for i, bounds in enumerate(split):
        lower, upper = bounds
        plt.plot(time[lower:upper] - time[lower], voltage[lower:upper],
                 color=cmap(norm(i)))

    plt.subplot2grid((4, 1), (1, 0), rowspan=3)
    plt.xlabel('Time (ms)')
    plt.ylabel('I (nA)')
    styles = ['-', '--', ':']
    for i, bounds in enumerate(split):
        lower, upper = bounds
        for j, current in enumerate(currents):
            label = labels[j] if labels and i == 0 else None
            plt.plot(time[lower:upper] - time[lower], current[lower:upper],
                     styles[j], color=cmap(norm(i)), label=label)
    if labels:
        plt.legend()

    try:
        xlim, ylim = zoom[protocol]
        plt.xlim(*xlim)
        plt.ylim(*ylim)
    except KeyError:
        pass


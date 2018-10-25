#!/usr/bin/env python2
#
# Show traditional protocol data plus simulation
#
from __future__ import division
from __future__ import print_function
import os
import sys
import pints
import numpy as np
import myokit
import cPickle
import argparse
import matplotlib.pyplot as plt

def run_model(model, cell, protocol, time, voltage, current, plot='unfold', label=None, axes=None):

    # Select protocol file
    protocol_file = os.path.join(root, protocol + '.mmt')
    print(protocol_file)
    myokit_protocol = myokit.load_protocol(protocol_file)

    # Estimate noise from start of data
    sigma_noise = np.std(current[:2000], ddof=1)

    # fetch cmaes parameters
    obtained_parameters = model.fetch_parameters()


    # Cell-specific parameters
    temperature = model.temperature(cell)
    lower_conductance = model.conductance_limit(cell)

    # Apply capacitance filter based on protocol
    print('Applying capacitance filtering')
    time, voltage, current = model.capacitance(myokit_protocol, 0.1, time, voltage, current)

    forward_model = model.ForwardModel(myokit_protocol, temperature, sine_wave=False)
    problem = pints.SingleOutputProblem(forward_model, time, current)
    log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
    log_prior = model.LogPrior(lower_conductance)
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    # Show obtained parameters and score
    obtained_log_posterior = log_posterior(obtained_parameters)
    print('Kylie sine-wave parameters:')
    for x in obtained_parameters:
        print(pints.strfloat(x))
    print('Final log-posterior:')
    print(pints.strfloat(obtained_log_posterior))

    # Simulate
    simulated = forward_model.simulate(obtained_parameters, time)

    if plot == 'unfold':
        axes[0].plot(time, voltage, color ='red')#, label='voltage')
        #axes[0].legend(loc='upper right')
    	axes[1].plot(time, current, alpha= 0.3, color ='red')#, label='measured current')


        if label == 0:
            model_name = 'circularCOIIC'
            axes[1].plot(time, simulated, alpha= 1, color= 'blue', label=model_name)
        elif label == 1:
            model_name = 'linearCOI'
            axes[1].plot(time, simulated, alpha= 1, color= 'magenta', label=model_name)
        elif label == 2:
            model_name = 'linearCCOI'
            axes[1].plot(time, simulated, alpha= 1, color= 'seagreen', label=model_name)
        elif label == 3:
            model_name = 'linearCCCOI'
            axes[1].plot(time, simulated, alpha= 1, color= 'seagreen', label=model_name)
    	#axes.subplot(2,1,1)

    else:

        IkrModel.fold_plot(protocol, time, voltage, [current, simulated])


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Plot traditional protocols for given model')
    parser.add_argument('--model', type=int, default=1, metavar='N', \
      help='model number : 1 for C-O-I-IC, 2 for C-O and so on' )
    parser.add_argument('--cell', type=int, default=5, metavar='N', \
      help='cell number : 1, 2, ..., 5' )
    parser.add_argument('--compare', type=bool, default=False, metavar='N', \
      help='compare : if True thencompare all modelsin a single plot' )
    args = parser.parse_args()
    sys.path.append(os.path.abspath('models_forward'))

    if args.model == 1 and not(args.compare):
        import circularCOIIC as IkrModel
        print("loading  C-O-I-IC model")
        # Load earlier result
        """
        filename = './cmaes_results/cell-5-circularCOIIC.txt'
        print
        with open(filename, 'r') as f:
            obtained_parameters = [float(x) for x in f.readlines()]
        """

    elif args.model == 2 and not(args.compare):
        import linearCO as IkrModel
        print("loading  C-O model")
        """
        filename = './cmaes_results/cell-5-linearCO.txt'
        with open(filename, 'r') as f:
            obtained_parameters = [float(x) for x in f.readlines()]
        """

    elif args.model == 3 and not(args.compare):
        import linearCOI as IkrModel
        print("loading  C-O-I model")
        """
        filename = './cmaes_results/cell-5-linearCOI.txt'
        with open(filename, 'r') as f:
            obtained_parameters = [float(x) for x in f.readlines()]
        """

    cell = args.cell

    # Select protocol
    protocol = 'pr3-steady-activation'

    # Select data file
    root = os.path.abspath('traditional-data')
    data_file = os.path.join(root, protocol + '-cell-' + str(cell) + '.csv')

    # Load data
    log = myokit.DataLog.load_csv(data_file).npview()
    time = log.time()
    current = log['current']
    voltage = log['voltage']
    del(log)

    if not(args.compare):

        run_model(IkrModel, cell, protocol, time, voltage, current)

    else:
        #sys.path.append(os.path.abspath('models_forward'))
        import circularCOIIC as IkrModel1
        import linearCOI as IkrModel2
        import linearCCOI as IkrModel3
        import linearCCCOI as IkrModel4


        models = [IkrModel1,IkrModel2,IkrModel3,IkrModel4]
        fig, ax = plt.subplots(nrows=2, ncols=1)
        for label, model_ikr in enumerate(models):
            print(model_ikr)
            IkrModel = model_ikr

            print(label)
            if label !=1:
                run_model(IkrModel, cell, protocol, time, voltage, current, 'unfold', label, ax)
        ax[1].legend(loc='lower right')
        plt.show()

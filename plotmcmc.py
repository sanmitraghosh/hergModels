#!/usr/bin/env python2
#
# Show traditional protocol data plus simulation
#
from __future__ import division
from __future__ import print_function
import os
import sys
import pints
import pints.plot as pplot
import numpy as np
import myokit
import cPickle
import argparse
import matplotlib.pyplot as plt

def run_model(model, cell, protocol, time, voltage, current, temperature, plot='unfold', label=None, axes=None, colour='SeaGreen'):

    forward_model = model.ForwardModel(protocol, temperature, sine_wave=False, logTransform=False)
    
    
    model_name = label#'model-'+str(label+1)
    print(model_name)
    
    root = os.path.abspath('mcmc_results')
    param_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-mcmc_traces.p')
    trace = cPickle.load(open(param_filename, 'rb'))
    burnin = 50000
    npar = forward_model.n_parameters()
    samples_all_chains = trace[:, burnin:, :]
    sample_chain = samples_all_chains[0]
    
    #indices = int(np.random.choice(50000,size=1))
    

    root = os.path.abspath('figures/mcmc')
    ppc_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-mcmc_ppc.png')
    pairplt_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-mcmc_pairplt.png')
    traceplt_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-mcmc_traceplt.eps')
    

    new_values = []
    for ind in range(100):
        ppc_sol=forward_model.simulate(sample_chain[ind,:npar], time)
        new_values.append(ppc_sol)
    new_values = np.array(new_values)
    mean_values = np.mean(new_values, axis=0)
    new_values.shape
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(time, voltage, color='orange', label='measured voltage')
    plt.xlim(0,8000)
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(time, current,'--', color='blue',lw=1.5, label='measured current')
    plt.plot(time, mean_values, color='SeaGreen', lw=1, label='mean of inferred current')
    plt.xlim(0,8000)
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(time[-40000:], current[-40000:],'--', color='blue',lw=1.5, label='measured current blow-up')
    plt.plot(time[-40000:], mean_values[-40000:], color='SeaGreen', lw=1, label='mean of inferred current blow-up')
    plt.xlim(4000,8000)
    plt.legend()
    plt.savefig(ppc_filename)   
    plt.close()

    pplot.pairwise(sample_chain[:,:npar], opacity=1)
    plt.savefig(pairplt_filename)   
    plt.close()

    pplot.trace(samples_all_chains)
    plt.savefig(traceplt_filename)   
    plt.close()


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
    

    if args.model == 1:	
        import circularCOIIC as forwardModel
        model_name ='model-1'
        print("loading  C-O-I-IC model")
        
    elif args.model == 2:
        import linearCOI as forwardModel
        model_name ='model-2'
        print("loading  C-O-I model")

    elif args.model == 3:
        import linearCCOI as forwardModel
        print("loading  C-C-O-I model")
        model_name ='model-3'

    elif args.model == 4:
        import linearCCCOI as forwardModel
        print("loading  C-C-C-O-I model")
        model_name ='model-4'


        
    cell = args.cell

    root = os.path.abspath('sine-wave-data')
    #
    # Select protocol file
    #
    protocol_file = os.path.join(root,'steps.mmt')
    #
    # Load protocol
    #
    protocol = myokit.load_protocol(protocol_file)

    #
    # Select data file
    #
    data_file = os.path.join(root, 'cell-' + str(cell) + '.csv')
    #

    # Load data
    log = myokit.DataLog.load_csv(data_file).npview()
    time = log.time()
    current = log['current']
    voltage = log['voltage']
    del(log)

    # Cell-specific parameters
    temperature = forwardModel.temperature(cell)
    
    # Apply capacitance filter based on protocol
    print('Applying capacitance filtering')
    time, voltage, current = forwardModel.capacitance(protocol, 0.1, time, voltage, current)

    run_model(forwardModel, cell, protocol, time, voltage, current, temperature, label=model_name)
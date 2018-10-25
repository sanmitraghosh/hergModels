#!/usr/bin/env python2
#
# Fit Kylie's model to Cell 5 data using CMA-ES
#
from __future__ import division, print_function
import os
import sys
import pints
import pints.plot as pplot
import numpy as np
import cPickle
import myokit
import argparse
import matplotlib.pyplot as plt
# Load a hERG model and prior

# Check input arguments

parser = argparse.ArgumentParser(description='Fit all the hERG models to sine wave data')
parser.add_argument('--cell', type=int, default=5, metavar='N', \
      help='cell number : 1, 2, ..., 5' )
parser.add_argument('--burnin', type=int, default=50000, metavar='N', \
      help='number of burn-in samples')
parser.add_argument('--train', type=bool, default=False, metavar='N', \
      help='plot training fits' )
args = parser.parse_args()
sys.path.append(os.path.abspath('models_forward'))

cell = args.cell
root = os.path.abspath('sine-wave-data')
print(root)
data_file = os.path.join(root, 'cell-' + str(cell) + '.csv')
protocol_file = os.path.join(root,'steps.mmt')

protocol = myokit.load_protocol(protocol_file)
log = myokit.DataLog.load_csv(data_file).npview()
time = log.time()
current = log['current']
voltage = log['voltage']
del(log)

root_ap = os.path.abspath('ap-data')
print(root)
data_file_ap = os.path.join(root_ap, 'cell-' + str(cell) + '.csv')

log_ap = myokit.DataLog.load_csv(data_file_ap).npview()
time_ap = log_ap.time()
current_ap = log_ap['current']
voltage_ap = log_ap['voltage']
del(log_ap)
protocol_ap = [time_ap, voltage_ap]

model_ppc_traces = []
model_ppc_traces_ap = []
mppc_cols =['gold','lightseagreen']
plt.rc('font', size=20)
fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(25, 15))
ikr_names = ['Beattie', 'Wang']
for i in xrange(len(ikr_names)):

    if i ==0:
        import circularCOIIC as forwardModel
        model_name ='model-1'
        print("loading  C-O-I-IC model")


    elif i ==1:
        import linearCCCOI as forwardModel
        print("loading  C-C-C-O-I model")
        model_name ='model-4'

    temperature = forwardModel.temperature(cell)
    lower_conductance = forwardModel.conductance_limit(cell)
    time, voltage, current = forwardModel.capacitance(
        protocol, 0.1, time, voltage, current)
    #
    # Create forward model
    #
    model = forwardModel.ForwardModel(protocol, temperature, sine_wave=True, logTransform=False)

    root = os.path.abspath('mcmc_results')
    param_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-mcmc_traces.p')
    trace = cPickle.load(open(param_filename, 'rb'))

    burnin = args.burnin
    samples_all_chains = trace[:, burnin:, :]
    sample_chain_1 = samples_all_chains[0]
    rand_samples =sample_chain_1[np.random.choice(50000,5),:]
    #print(rand_samples.shape)
    npar = model.n_parameters()
    
    root = os.path.abspath('figures/mcmc/Gary_disc')
    ppc_filename = os.path.join(root, model_name +'-cell-' + str(cell) + '-mcmc_ppc.png')
    model_ppc_filename = os.path.join(root, 'all models '+ '-cell-' + str(cell) + '-mcmc_ppc.png')
    
    Train = args.train
    print(Train)
    if Train == True:

        
        new_values = []
        
        for ind in xrange(len(rand_samples)):
            
            ppc_sol=model.simulate(rand_samples[ind,:npar], time)
            new_values.append(ppc_sol)
        new_values = np.array(new_values)
        
        mean_values = np.mean(new_values, axis=0)
        model_ppc_traces.append(mean_values)        
    
    else:

        model_ppc_filename = os.path.join(root, 'all models '+ '-cell-' + str(cell) + '-test-mcmc_ppc.png')
        model_ap = forwardModel.ForwardModel(protocol_ap, temperature, sine_wave=False, logTransform=False)
        
        new_values_ap = []
        for ind in xrange(len(rand_samples)):
            
            ppc_sol_ap=model_ap.simulate(rand_samples[ind,:npar], time_ap)
            new_values_ap.append(ppc_sol_ap)
        new_values = np.array(new_values_ap)
        time=time_ap
        current=current_ap
        voltage=voltage_ap

  
    if i == 0:
        axes[0].plot(time, voltage, color='orange', lw=1.8,label='measured voltage')
    else:
        axes[0].plot(time, voltage, lw=1.8,color='orange')
    
    axes[0].set_xlim(0,7960)
    axes[0].legend()
    """
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(40)
    """
    if i == 0:
        axes[1].plot(time, current,'--', color='blue',lw=0.5, label='measured current')
    else:
        axes[1].plot(time, current,'--', color='blue',lw=0.5,)
    axes[1].plot(time, new_values[0,:], color=mppc_cols[i], lw=0.5, label=ikr_names[i])
    for ind in xrange(len(new_values)-1):
        
        axes[1].plot(time, new_values[ind+1,:], color=mppc_cols[i], lw=0.5)
    
    axes[1].set_xlim(0,7960)
    #box = axes[1].get_position()
    #axes[1].set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.98])
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=6)
    
    if i ==1:


        mpd_fit = os.path.join(root, 'wang_sine_fit_mpd.png')
        mpd_ind = np.argsort(sample_chain_1[:,npar+9])
        
        mpd_mode_1= sample_chain_1[mpd_ind[-1],:npar]
        #print(sample_chain_1[mpd_ind[-1],-1])
        outfile_1 =os.path.join(root, 'mpd_1.txt')
        np.savetxt(outfile_1,mpd_mode_1)
        mpd_sol_1=model.simulate(mpd_mode_1, time)

        mpd_mode_2= sample_chain_1[mpd_ind[-1000],:npar]
        #print(sample_chain_1[mpd_ind[-100],-1])
        outfile_2 =os.path.join(root, 'mpd_2.txt')
        np.savetxt(outfile_2,mpd_mode_2)
        mpd_sol_2=model.simulate(mpd_mode_2, time)

        plt.figure()
        plt.rc('font', size=12)
        plt.subplot(3,1,1)
        plt.plot(time, voltage, color='orange', label='measured voltage')
        plt.xlim(0,8000)
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(time, current,'--', color='blue',lw=1.5)
        plt.plot(time, mpd_sol_1, color='SeaGreen', lw=1)
        plt.plot(time, mpd_sol_2, color='brown', lw=1)
        plt.xlim(0,8000)
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(time[-40000:], current[-40000:],'--', color='blue',lw=1.5, label='measured current')
        plt.plot(time[-40000:], mpd_sol_1[-40000:], color='SeaGreen', lw=1, label='At mode 1')
        plt.plot(time[-40000:], mpd_sol_2[-40000:], color='brown', lw=1, label='At mode 2')
        plt.xlim(4000,8000)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=6)
    
        plt.savefig(mpd_fit)   
        plt.close()
plt.show()
fig.savefig(model_ppc_filename)  

#root = os.path.abspath('figures/mcmc/Gary_disc')


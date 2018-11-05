# Prior for Kylie's model
#
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

def go(log, parameters):
    
    
    
    lower_alpha = 1e-7              # Kylie: 1e-7
    upper_alpha = 1e3               # Kylie: 1e3
    lower_beta  = 1e-7              # Kylie: 1e-7
    upper_beta  = 0.4               # Kylie: 0.4

    n = 10000
    b = np.linspace(lower_beta, upper_beta, n)
    if log:
        a = np.exp(np.linspace(np.log(lower_alpha), np.log(upper_alpha), n))
    else:
        a = np.linspace(lower_alpha, upper_alpha, n)

    rmin = 1.67e-5
    rmax = 1000

    vmin = -120
    vmax = 58.25
    
    plt.subplot(1, 2, 1)
    plt.xlabel('p1')
    plt.ylabel('p2')
    plt.axhline(lower_beta, color='k', alpha=0.25)
    plt.axhline(upper_beta, color='k', alpha=0.25)
    plt.axvline(np.log(lower_alpha), color='k', alpha=0.25)
    plt.axvline(np.log(upper_alpha), color='k', alpha=0.25)
    
    
    bmin = (1 / vmax) * (np.log(rmin) - np.log(a))
    bmax = (1 / vmax) * (np.log(rmax) - np.log(a))
    bmin = np.maximum(bmin, lower_beta)
    bmax = np.minimum(bmax, upper_beta)
   
    plt.plot(np.log(a), bmin, label='Lower bound')
    plt.plot(np.log(a), bmax, label='Upper bound')
    plt.plot(parameters[0], parameters[1], 'x', label='Wang Cell-5 p1, p2')
    plt.plot(parameters[4], parameters[5], 'x', label='Wang 2018 Cell-5 p5, p6')
    plt.plot(parameters[8], parameters[9], 'x', label='Wang 2018 Cell-5 p9, p10')
    
    #
    # Figure 2
    #
    plt.subplot(1, 2, 2)
    plt.xlabel('p3')
    plt.ylabel('p4')
   
    plt.axhline(lower_beta, color='k', alpha=0.25)
    plt.axhline(upper_beta, color='k', alpha=0.25)
    plt.axvline(np.log(lower_alpha), color='k', alpha=0.25)
    plt.axvline(np.log(upper_alpha), color='k', alpha=0.25)
    bmin = (-1 / vmin) * (np.log(rmin) - np.log(a))
    bmax = (-1 / vmin) * (np.log(rmax) - np.log(a))
    bmin = np.maximum(bmin, lower_beta)
    bmax = np.minimum(bmax, upper_beta)
    plt.plot(np.log(a), bmin, label='Lower bound')
    plt.plot(np.log(a), bmax, label='Upper bound')
    plt.plot(parameters[2], parameters[3], 'x', label='Wang Cell-5 p3, p4')
    plt.plot(parameters[6], parameters[7], 'x', label='Wang Cell-5 p7, p8')
    plt.plot(parameters[10], parameters[11], 'x', label='Wang Cell-5 p11, p12')
 
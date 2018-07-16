# Prior for Kylie's model
#
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

def go(log, parameters):
    """
    parameters = [
        2.26026076650526e-004,
        6.99168845608636e-002,
        3.44809941106440e-005,
        5.46144197845311e-002,
        8.73240559379590e-002,
        8.91302005497140e-003,
        5.15112582976275e-003,
        3.15833911359110e-002,
        1.52395993652348e-001,
    ]
    """
    lower_alpha = 1e-7              # Kylie: 1e-7
    upper_alpha = 1e3               # Kylie: 1e3
    lower_beta  = 1e-7              # Kylie: 1e-7
    upper_beta  = 0.4               # Kylie: 0.4

    n = 1000
    b = np.linspace(lower_beta, upper_beta, n)
    if log:
        a = np.exp(np.linspace(np.log(lower_alpha), np.log(upper_alpha), n))
    else:
        a = np.linspace(lower_alpha, upper_alpha, n)

    rmin = 1.67e-5
    rmax = 1000

    vmin = -120
    vmax = 58.25

    title = 'r = p1 * np.exp(p2 * v)'
    title += ' v in [' + str(vmin) + ', ' + str(vmax) + ']'
    title += ' r in [' + str(rmin) + ', ' + str(rmax) + ']'

    #
    # Figure 1
    #
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.xlabel('p1')
    plt.ylabel('p2')
    plt.axvline(lower_alpha, color='k', alpha=0.25)
    plt.axvline(upper_alpha, color='k', alpha=0.25)
    plt.axhline(lower_beta, color='k', alpha=0.25)
    plt.axhline(upper_beta, color='k', alpha=0.25)
    if log:
        plt.xscale('log')
	plt.yscale('log')
 #       plt.xlim(lower_alpha * 0.1, upper_alpha * 10)
#	plt.xlim(lower_beta * 0.1, upper_beta * 10)
    else:
        plt.xlim(lower_alpha - 50, upper_alpha + 50)
    plt.ylim(lower_beta - 0.05, upper_beta + 0.05)
    bmin = (1 / vmax) * (np.log(rmin) - np.log(a))
    bmax = (1 / vmax) * (np.log(rmax) - np.log(a))
    bmin = np.maximum(bmin, lower_beta)
    bmax = np.minimum(bmax, upper_beta)
    plt.fill_between(a, bmin, bmax, color='k', alpha=0.1, label='Prior')
    plt.plot(a, bmin, label='Lower bound')
    plt.plot(a, bmax, label='Upper bound')
    plt.plot(
        parameters[0], parameters[1], 'x', label='Beattie 2018 Cell-5 p1, p2')
    plt.plot(
        parameters[4], parameters[5], 'x', label='Beattie 2018 Cell-5 p5, p6')
    plt.legend(loc='upper right').get_frame().set_alpha(1)

    adiff = a[1:] - a[:-1]
    bdiff = (bmax - bmin)[:-1]
    area1 = np.sum(bdiff * adiff)
    area2 = (upper_beta - lower_beta) * (upper_alpha - lower_alpha)
    print('Estimated area under prior 2: ' + str(area1))
    print('Area within parameter bounds: ' + str(area2))
    print('Ratio: ' + str(area1 / area2))


    #
    # Figure 2
    #
    plt.subplot(1, 2, 2)
    plt.xlabel('p3')
    plt.ylabel('p4')
    plt.axvline(lower_alpha, color='k', alpha=0.25)
    plt.axvline(upper_alpha, color='k', alpha=0.25)
    plt.axhline(lower_beta, color='k', alpha=0.25)
    plt.axhline(upper_beta, color='k', alpha=0.25)
    if log:
        plt.xscale('log')
	plt.yscale('log')
#        plt.xlim(lower_alpha * 0.1, upper_alpha * 10)
#	plt.xlim(lower_beta * 0.1, upper_beta * 10)
    else:
        plt.xlim(lower_alpha - 50, upper_alpha + 50)
    plt.ylim(lower_beta - 0.05, upper_beta + 0.05)
    bmin = (-1 / vmin) * (np.log(rmin) - np.log(a))
    bmax = (-1 / vmin) * (np.log(rmax) - np.log(a))
    bmin = np.maximum(bmin, lower_beta)
    bmax = np.minimum(bmax, upper_beta)
    plt.fill_between(a, bmin, bmax, color='k', alpha=0.1, label='Prior')
    plt.plot(a, bmin, label='Lower bound')
    plt.plot(a, bmax, label='Upper bound')
    plt.plot(
        parameters[2], parameters[3], 'x', label='Beattie 2018 Cell-5 p3, p4')
    plt.plot(
        parameters[6], parameters[7], 'x', label='Beattie 2018 Cell-5 p7, p8')
    plt.legend(loc='upper right').get_frame().set_alpha(1)

    adiff = a[1:] - a[:-1]
    bdiff = (bmax - bmin)[:-1]
    area1 = np.sum(bdiff * adiff)
    area2 = (upper_beta - lower_beta) * (upper_alpha - lower_alpha)
    print('Estimated area under prior 2: ' + str(area1))
    print('Area within parameter bounds: ' + str(area2))
    print('Ratio: ' + str(area1 / area2))
    plt.show()

#
# Show
#
#go(True)
#go(False)
#plt.show()

#!/usr/bin/env python2
#
# Fit Kylie's model to Cell 5 data using CMA-ES
#
from __future__ import division, print_function
import os
import sys
import pints
import numpy as np
import myokit
from boundaries import go
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
# Load beattie model and prior
sys.path.append(os.path.abspath('models_forward'))
import circularCOIIC as forwardModel
model_name ='CCOIIC'

# Check input arguments

#
# Select cell
#
cell = 5


#
# Select data file
#
root = os.path.abspath('sine-wave-data')
print(root)
data_file = os.path.join(root, 'cell-' + str(cell) + '.csv')


#
# Select protocol file
#
protocol_file = os.path.join(root,'steps.mmt')


#
# Cell-specific parameters
#
temperature = forwardModel.temperature(cell)
lower_conductance = forwardModel.conductance_limit(cell)


#
# Load protocol
#
protocol = myokit.load_protocol(protocol_file)


#
# Load data
#
log = myokit.DataLog.load_csv(data_file).npview()
time = log.time()
current = log['current']
voltage = log['voltage']
del(log)


#
# Estimate noise from start of data
# Kylie uses the first 200ms, where I = 0 + noise
#
sigma_noise = np.std(current[:2000], ddof=1)


#
# Apply capacitance filter based on protocol
#
print('Applying capacitance filtering')
time, voltage, current = forwardModel.capacitance(
    protocol, 0.1, time, voltage, current)


#
# Create forward model
#
model = forwardModel.ForwardModel(protocol, temperature, sine_wave=True)


#
# Define problem
#
problem = pints.SingleOutputProblem(model, time, current)


#
# Define log-posterior
#
log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
log_prior = forwardModel.LogPrior(lower_conductance)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)

parameters = [
        2.26026076650526e-4,
	6.99168845608636e-2,
	3.44809941106440e-5,
	5.46144197845311e-2,
	8.73240559379590e-2,
	8.91302005497140e-3,
	5.15112582976275e-3,
	3.15833911359110e-2,
	1.52395993652348e-1
	]

lower_alpha = 1e-7              # Kylie: 1e-7
upper_alpha = 1e3               # Kylie: 1e3
lower_beta  = 1e-7              # Kylie: 1e-7
upper_beta  = 0.4               # Kylie: 0.4

run_sim=False
outfile='./figures/meshResultsp5p6.txt'
nx=ny=100
p1 = np.linspace(np.log(lower_alpha), np.log(upper_alpha), nx)
#p2 = np.linspace(np.log(lower_beta), np.log(upper_beta), ny)
p2 = np.linspace(lower_beta, upper_beta, ny)
A, B = np.meshgrid(p1, p2)

if run_sim:
		
	AB=[]
	counter=0
	for i in range(nx):
	     for j in range(ny):
		     params=parameters
		     
		     params[4], params[5]=np.exp(A[i,j]), B[i,j] #np.exp(B[i,j])#np.exp(B[i,j])
	
		     lp=log_likelihood(params)
		     if np.isnan(lp):
			lp = -np.inf #AB[counter-1]
		     	print(lp)		
		     AB.append(lp)
		     counter += 1
		     #AB.append(params)
	print(np.array(AB))
	logP=np.array(AB)
	np.savetxt(outfile, logP)
else:
	print('loading earlier simulation ')
	outfile1='./figures/meshResultsp5p6.txt'
	outfile2='./figures/meshResultsBothLogp5p6.txt'
	logP=np.loadtxt(outfile1)
	logPbothL=np.loadtxt(outfile2)

rmin = 1.67e-5
rmax = 1000

vmin = -120
vmax = 58.25
a=np.exp(np.linspace(np.log(lower_alpha), np.log(upper_alpha),10000))#p1np.linspace(np.log(lower_alpha), np.log(upper_alpha)
bmin = (1 / vmax) * (np.log(rmin) - np.log(a))
bmax = (1 / vmax) * (np.log(rmax) - np.log(a))
bmin = np.maximum(bmin, lower_beta)
bmax = np.minimum(bmax, upper_beta)
"""
Z=logP
scale=float(1)
Z=Z.reshape((100,100))
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
origin = 'lower'


CS=plt.contourf(A, B, Z ,1000, origin=origin)
cbar = plt.colorbar(CS)
plt.plot(a, bmin, label='Lower bound')
plt.plot(a, bmax, label='Upper bound')
plt.xlabel('log(p1)')
plt.ylabel('p2')
plt.xscale('log')
plt.xlim(lower_alpha * 0.1, 1e3 * 10)
plt.ylim(lower_beta , upper_beta *1)
plt.title('Log(p1), p2')
plt.legend(loc='upper right').get_frame().set_alpha(1)	
plt.subplot(1, 2, 2)
origin = 'lower'
#levels = [-1.5, -1, -0.5, 0, 0.5, 1]

CS=plt.contourf(A, B, Z ,10, origin=origin)
cbar = plt.colorbar(CS, label='LogLikelihood')
plt.plot(a, bmin, label='Lower bound')
plt.plot(a, bmax, label='Upper bound')
plt.xlabel('log(p1)')
plt.ylabel('log(p2)')

plt.xscale('log')
plt.yscale('log')
plt.xlim(lower_alpha * 0.1, 1e3 * 10)
plt.ylim(lower_beta , upper_beta)
plt.title('Log(p1), log(p2)')
"""

import matplotlib.colors as colors
from copy import copy
from matplotlib import ticker, cm
from matplotlib.colors import LinearSegmentedColormap

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
palette = copy(plt.cm.gray)
palette.set_over('r', 1.0)
palette.set_under('g', 1.0)
palette.set_bad('b', 1.0)
Z=logP
scale=float(1)
Z=Z.reshape((100,100))
V=Z
locs=np.where(V==-np.inf)
V[locs]=1e12
V=np.log(abs(V))
V=-1*V
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)

boundary=np.linspace(V.min(),V.max(),20)
m = plt.imshow(V,interpolation='none',
					
					norm=colors.BoundaryNorm(boundaries=boundary,ncolors=palette.N),#palette.N#
					#norm=colors.LogNorm(vmin=V.max(), vmax=V.min()),                			
					aspect='auto',
					cmap=parula_map,
                			origin='lower', extent=[np.log(1e-7), np.log(1e3), 1e-7, 0.4]
                )
plt.text(0.5, 0.3, 'Masked \n here', fontdict=font)
plt.plot(np.log(parameters[4]),parameters[5],'x', color='k',alpha=1.0,label='best fits')
#plt.xscale('log')
#plt.yscale('log')
cbar = plt.colorbar(m, extend='both', shrink=1)
plt.plot(np.log(a), bmin, label='Lower bound', color ='k')
plt.plot(np.log(a), bmax, label='Upper bound', color ='r')
plt.xlabel('log(p5)')
plt.ylabel('p6')
plt.legend(loc='upper right').get_frame().set_alpha(1)
plt.title('Log(p5), p6')


plt.subplot(1,2,2)

L=logPbothL
L=L.reshape((100,100))
V=L
locs=np.where(V==-np.inf)
V[locs]=1e12
V=np.log(abs(V))
V=-1*V

boundary=np.linspace(V.min(),V.max(),20)
m = plt.imshow(V,interpolation='none',
					
					norm=colors.BoundaryNorm(boundaries=boundary,ncolors=palette.N),#palette.N#
					
                			aspect='auto',
					cmap=parula_map,
                			origin='lower', extent=[np.log(1e-7), np.log(1e3), np.log(1e-7), np.log(0.4)]
                )

plt.plot(np.log(parameters[4]),np.log(parameters[5]),'x',color='k',alpha=1.0)
#plt.xscale('log')
#plt.yscale('log')
cbar = plt.colorbar(m, extend='both', shrink=1,label='Log(LogLikelihood)')
plt.plot(np.log(a), np.log(bmin), label='Lower bound', color ='k')
plt.plot(np.log(a), np.log(bmax+1e-7), label='Upper bound', color ='r')
plt.xlabel('log(p5)')
plt.ylabel('log(p6)')

plt.title('Log(p5), log(p6)')

plt.show()
print(log_likelihood(parameters))
print(abs(L))
"""
delta=0.25
p1=p2=np.arange(0.1, 3.01, delta)
X, Y = np.meshgrid(np.log(p1), np.log(p2))
X, Y=np.exp([X,Y])
print(X,Y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

interior = np.sqrt((X**2) + (Y**2)) > 0.5
Z[interior] = np.ma.masked

outfile = TemporaryFile()
np.save(outfile, Z)

#X,Y=np.log([X,Y])
origin = 'lower'
plt.contourf(X, Y, Z, 10, cmap=plt.cm.bone, origin=origin)
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(lower_alpha * 0.1, upper_alpha * 10)
#plt.ylim(lower_beta *0, upper_beta *10)
plt.show()


outfile.seek(0)
LZ=np.load(outfile)
plt.contourf(X, Y, LZ, 10, cmap=plt.cm.bone, origin=origin)
plt.show()
"""





























# coding: utf-8

# # Marginal likelihood of linear regression
# 
# I am trying to see whether TI gives the same marg likelihood as Analytical. The model is
# $$y = ax +m + \epsilon$$
# 
# $\epsilon\sim\mathcal{N}(0,1)$ and the priors are $a\sim \mathcal{N}(2,1)$ and $m\sim \mathcal{N}(3,1)$.
# 
# $x=(1,2,\ldots,20)$
# 
# We can write this in matrix form as 
# $$Y=X\beta + \boldsymbol{\epsilon}$$, where $X=\begin{bmatrix} x_1 &1 \\ \vdots & \vdots \\ x_{20} & 1\end{bmatrix}$
# and $\beta=[a,m]^T$. the bivariate Gaussian prior is then $\mathcal{N}(\hat{\beta},C_{uu})=\mathcal{N}([2,3],diag(1,1))$
# 
# According to [this paper](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2014WR016062) the analytical likelihood is $p(Y) = \mathcal{N}(Y|X\hat{\beta},C_{yy} +R)$, where $C_{yy}=X C_{uu} X^T$. $R$ is the diagonal noise covariance, which in this problem is identity, thus $R=\mathbb{I}$.
# 
# ## Thermodynamic Integration Formula
# 
# We write the Expected log likelihood of a heated chain $j$, with temperatures $0<t(j)<1$, as
# $$ y(j) = 1/Ns\sum_{i=1}^{Ns} t(j)\log p(Y|X,\beta_{i})$$, where $Ns$ is the number of post-burnin (amd thinned) samples.
# 
# The marginal likelihood is then given by
# $$p(Y)=\exp\Bigg(\sum_{j=2}^{J}\Big( \frac{y(j)+y(j-1)}{2}\Big)\big(t(j)-t(j-1)\big)\Bigg)$$, where $J$ is the number of chains.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy
#get_ipython().run_line_magic('matplotlib', 'inline')

import pints

class LinearModel(pints.ForwardModel):
    
    def simulate(self, parameters, times):
        a = parameters[0]
        m = parameters[1]
        y = a*times + m
        return y.reshape(times.shape)
    
    def n_parameters(self):
        # Return the dimension of the parameter vector
        return 2


# Then create an instance of our new model class
model = LinearModel()


# In[ ]:


true_parameters = [2., 3.]
times = np.arange(1, 20)
# Run a simulation to get test data
values = model.simulate(true_parameters, times)

# Add some noise
values += np.random.normal(0., 1., values.shape)

# Plot the results
"""
plt.figure()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.plot(times, values)
plt.show()
"""


# In[ ]:


problem = pints.SingleOutputProblem(model, times, values)


# In[ ]:


class tiLogLikelihood(pints.ProblemLogLikelihood):
    def __init__(self, problem, sigma, temp):
        super(tiLogLikelihood, self).__init__(problem)

        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._nt = problem.n_times()
        self.temp = temp
        #self._n_parameters = problem.n_parameters()

        if np.isscalar(sigma):
            self.sigma = np.ones(self._no) * float(sigma)
        else:
            self.sigma = pints.vector(sigma)
            if len(sigma) != self._no:
                raise ValueError(
                    'Sigma must be a scalar or a vector of length n_outputs.')
        if np.any(self.sigma <= 0):
            raise ValueError('Standard deviation must be greater than zero.')

        self._offset = -0.5 * self._nt * np.log(2 * np.pi)
        self._offset -= self._nt * np.log(sigma)
        self._multip = -1 / (2.0 * sigma**2)        
    def __call__(self, x):
        
        error = self._values - self._problem.evaluate(x)
        return self.temp * np.sum(self._offset + self._multip * np.sum(error**2, axis=0))


# In[ ]:


sigma_noise = 1.
log_prior_a = pints.NormalLogPrior(2.,1.)
log_prior_b = pints.NormalLogPrior(3.,1.)
log_prior = pints.ComposedLogPrior(log_prior_a,log_prior_b)


# In[ ]:


import mcmcsampling
from joblib import Parallel, delayed
import multiprocessing
niter = 30000
xs = log_prior.sample(1)
def mcmc_runner(temps):

    nchains = 1
    #print('temperature', temps)
    tempered_log_likelihood = tiLogLikelihood(problem, sigma_noise, temps)
    tempered_log_posterior = pints.LogPosterior(tempered_log_likelihood, log_prior)
    
    mcmc = mcmcsampling.MCMCSampling(tempered_log_posterior, nchains, xs,
                                        method=pints.AdaptiveCovarianceMCMC)
    #mcmc.set_log_to_file('log.txt')
    mcmc.set_log_to_screen(False)
    mcmc.set_max_iterations(niter)
    mcmc.set_parallel(False)
    chains, LL = mcmc.run(returnLL=True)
    return chains, LL

temperature = np.unique(np.sort(np.hstack((np.linspace(1e-5,1e-3,10),np.linspace(1e-3,0.01,15),np.linspace(0.01,0.1,9),np.linspace(1e-5,1,11)))))
print(temperature)
with np.errstate(all='ignore'):  # Tell numpy not to issue warnings
    
    num_cores = multiprocessing.cpu_count()
     
    results = Parallel(n_jobs=num_cores-2)(delayed(mcmc_runner)(t) for t in temperature)


# In[ ]:


burnin = niter/2

param_chains = np.reshape(results[len(temperature)-1][0][:,burnin:,:],(burnin,2))
tempered_LLs = np.array([results[i][1] for i in range(len(temperature))]).reshape((len(temperature),niter))
untempered_LLs = np.array([tempered_LLs[i,:]/temperature[i] for i in range(len(temperature))])
loglike = untempered_LLs[:,burnin:].T

#plt.scatter(param_chains[:,0],param_chains[:,1])


# In[ ]:


def thermo_int(inp):

    ti=temperature
    print('schedule is :',ti)
    Eloglike_std = np.mean(inp,axis=0)
    E2loglike_std = np.mean(inp**2,axis=0)
    Vloglike_std = E2loglike_std - (np.mean(inp,axis=0))**2
    I_MC = []
    """
    for i in xrange(len(ti)-1):
        I_MC.append( (0.5*(Eloglike[i] + Eloglike[i+1]))* (ti[i+1]-ti[i]) )
    """
    for i in xrange(len(ti)-1):

        I_MC.append( (Eloglike_std[i] + Eloglike_std[i+1])/2 * (ti[i+1]-ti[i]) \
                - (Vloglike_std[i+1] - Vloglike_std[i])/12 * (ti[i+1]-ti[i])**2  ) 
    
    return np.exp(np.sum(I_MC)), Eloglike_std


# In[ ]:
estimated_marginal_likelihood , yks = thermo_int(loglike)
plt.plot(temperature,yks)
plt.ylabel('E|Log Likelihood|')
plt.xlabel('temperature')
plt.show()


# In[ ]:

from numpy.linalg import inv
from numpy import matmul
import scipy.stats as stats

H=np.vstack((times,np.ones(len(times)))).T
Cyy = matmul(matmul(H,np.diag([1.,1])),H.T)
R = np.diag(np.ones(len(times)))
Cov=Cyy + R
mu = np.dot(H,np.array([2.,3.]))
v=model.simulate(true_parameters, times)
vv=H.dot(np.array([2.,3.]))
np.testing.assert_array_equal(v,vv)
true_marginal_likelihood = stats.multivariate_normal(mu,Cov).pdf(values)
print('Estimated Marginal likelihood for model is:', estimated_marginal_likelihood)
print('True Marginal likelihood for model is:', true_marginal_likelihood)


Cuhuh = inv(matmul(matmul(H.T,inv(R)),H)) 
Cuu = np.diag([1.,1])
um = np.array([2.,3.])
ud = inv(H.T.dot(H)).dot(matmul(H.T,values))
Cudud = inv(inv(Cuhuh) + inv(Cuu))
uh = Cudud.dot((inv(Cuhuh).dot(ud)) + (inv(Cuu).dot(um)) )
true_samples = stats.multivariate_normal(uh, Cudud).rvs(burnin)
plt.subplot(1,2,1)
plt.title('True')
plt.scatter(true_samples[:,0],true_samples[:,1])
plt.xlim([1.5 ,2.5])
plt.ylim([1 ,5])
plt.subplot(1,2,2)
plt.title('MCMC')
plt.scatter(param_chains[:,0],param_chains[:,1])
plt.xlim([1.5, 2.5])
plt.ylim([1, 5])
plt.show()
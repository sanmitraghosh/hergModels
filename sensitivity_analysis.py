from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import scipy.io
from scipy.integrate import odeint
import numpy as np
import cPickle
import argparse

def loadData(numFiles=2):
    ozoneData=[]
    for matfile in range(numFiles):
            ozoneData.append(scipy.io.loadmat('ozone'+str(matfile+1)+'_newModel.mat'))
    return ozoneData

def preProcessData(data, subSampleRate=500, numEpisode=10, timeSteps=144):
    Voltage=[]
    for matfile in range(len(data)):
        channel1=np.asarray(data[matfile]["stch1"])
        channel2=np.asarray(data[matfile]["stch2"])
        DiffVoltage=np.subtract(channel1,channel2)
        #subsample by averaging every 500 samples
        Voltage.append(np.asarray([np.mean(DiffVoltage[1:,window].reshape(timeSteps, subSampleRate), axis=1)
                                   for window in range(numEpisode)]).T)
    return Voltage
#define the model of ozone induced clacium current
def deriv_ozone(y, t, param):
    #Define parameters
    koz,alph1,alph2,K1,K2,delta1,delta2,G,Voff,B,Jion,hx,l,tau,rho,dcal,Kcat,hca=param

    #define states
    Ot, I1t, I2t, V1t, V2t, Q1t, Q2t=y
    #define derivatives
    dOt_dt=-koz*Ot
    dI1t_dt=(alph1*Ot)/(K1+Ot) -delta1*I1t
    dI2t_dt=(alph2*Ot)/(K2+Ot) -delta2*I2t
    dV1t_dt=G*V1t/(np.exp(-B*(V1t-Voff))) + (Jion*((I1t)**hx))/(l+((I1t)**hx))-V1t/tau
    dV2t_dt=G*V2t/(np.exp(-B*(V2t-Voff))) + (Jion*((I2t)**hx))/(l+((I2t)**hx))-V2t/tau
    dQ1t_dt=I1t - (dcal*((Q1t)**hca))/(Kcat+((Q1t)**hca)) + V1t/rho
    dQ2t_dt=I2t - (dcal*((Q2t)**hca))/(Kcat+((Q2t)**hca)) + V2t/rho

    return dOt_dt, dI1t_dt, dI2t_dt, dV1t_dt, dV2t_dt, dQ1t_dt, dQ2t_dt
# here we define the objective function nonlinear least squares
def squareDist(param, data, time):
    #objective function for a single episode
    y0 = [10,0,0,0,0,param[18],param[19]]#I am trying to learn the init charge of calcium

    solution = odeint(deriv_ozone, y0, time, args=(param[0:18],))
    velocities= np.asarray([deriv_ozone(solution[times,:],time,param[0:18]) for times in range(np.size(time))])
    trace=velocities[:,5]-velocities[:,6]
    dist=np.sum(np.square(data-trace))#need to verify with matlab
    return dist
def squareDistMulti(param, data, time, objective, numEpisode=10):#shall use lambda
    #bieng ambitious trying to fit all episodes with a single parameter set
    if objective=='multiEpisodeMultiParam':
        param=np.reshape(param,(20,10))
        dist=np.sum([squareDist(param[:,episode], data[:,episode], time) for episode in range(numEpisode)])
        colDist=np.ones((numEpisode,numEpisode))
        for colA in range(numEpisode):
            for colB in range(numEpisode):
                colDist[colA,colB]=np.sum(scipy.spatial.distance.euclidean(param[:,colA], param[:,colB]))
        dist=dist + 0.0*np.sum(colDist)
    else:
        dist=np.sum([squareDist(param, data[:,episode], time) for episode in range(numEpisode)])

    return dist
def plotEpisode(ax, y_exp, x, y_model, legend='OFF'):
    # plots a single episode given axex
    ax.plot(x, y_exp, 'r--',label='Observed response')
    ax.plot(x, y_model, 'b--',label='Model response')
    ax.set_xlim([0 ,2])
    if legend=='ON':
        ax.legend(loc='upper right',fontsize='10')
    ax.set_xlabel('time (hours)',fontsize='16')
    ax.set_ylabel(r'$Voltage$ (mV)',rotation='vertical',fontsize='16')
    plt.tight_layout()
def plotExperiments(data, timeAxis, model, row=5, column=2, title='some experiment'):
    # pass an experiment to plot episodes
    fig, axes = plt.subplots(nrows=row, ncols=column,figsize=(10, 15))
    fig.suptitle(title, fontsize=24, fontweight='bold')
    #
    legenCount=0
    for row in axes:
        for ax in row:
            if legenCount==0:
                plotEpisode(ax, data[:,legenCount], timeAxis, model[:,legenCount], 'ON')
                legenCount+=1
            else:
                plotEpisode(ax, data[:,legenCount], timeAxis, model[:,legenCount])
                legenCount+=1
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return ax


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sensitivity Analysis Ozone')
    parser.add_argument('--dataset', type=int, default=1, metavar='N',
                    help='input dataset number : 1 for ozone 7, 2 for ozone 9')
    parser.add_argument('--direction', type=str, default='stiff', metavar='M',
                    help='mention the direction of analysis: stiff or sloppy (default: stiff)')
    args = parser.parse_args()



    # here we load two experiments at first, we also load the pickled parameters learnt previously
    oz=loadData()
    Voltage=preProcessData(oz)
    ts = np.linspace(0, 2, 144)
    plot_experiment_number=args.dataset-1#see the fitting for experiment number #
    data=np.asarray(Voltage[plot_experiment_number])
    paramsSE= np.array(cPickle.load(open('paramsSE.p', 'rb')))



    # First we plot the respective fits episode wise
    model=[]
    numEpisodes=10
    for episode in range(numEpisodes):
        parplot=paramsSE[plot_experiment_number][:,episode]
        y0 = [10,0,0,0,0,parplot[18],parplot[19]]
        states = odeint(deriv_ozone, y0, ts, args=(parplot[0:18],))
        velocities= np.asarray([deriv_ozone(states[i,:],ts,parplot[0:18]) for i in range(np.size(ts))])
        model.append(velocities[:,5]-velocities[:,6])
    model=np.asarray(model).T
    plotExperiments(data, ts, model, row=5, column=2, title='Fit for Dataset/experiment number '+str(plot_experiment_number+1))




    # Now we calculate the sensitivities as well as other geomtric patterns
    h=0.001
    episodeSens=[]
    episodeUeigVec=[]
    episodeVeigVec=[]
    episodeSingVal=[]
    episodeHess=[]
    # We define the numerical derivative as a vector function: f(\theta+h)-f(\theta-h)/2h. N.B. We use f(x):(data-model(\theta))^2
    dError=lambda y,xr,xl: (np.square(y-xr)-np.square(y-xl))/2*h
    vErrorfunc = np.vectorize(dError)

    for episode in range(numEpisodes):
        dEdparam=np.ones((data.shape[0],paramsSE.shape[1]))
        Hess=np.ones((paramsSE.shape[1],paramsSE.shape[1]))
        for th in range(paramsSE.shape[1]):
            # Calculate f(\theta+h) by perturbing each by perturbing eqach element of \theta
            param=paramsSE[plot_experiment_number][:,episode]
            drparam=param
            drparam[th]+=h
            y0 = [10,0,0,0,0,drparam[18],drparam[19]]
            states = odeint(deriv_ozone, y0, ts, args=(drparam[0:18],))
            velocities= np.asarray([deriv_ozone(states[i,:],ts,drparam[0:18]) for i in range(np.size(ts))])
            modelright=velocities[:,5]-velocities[:,6]
            # Calculate f(\theta-h) by perturbing each by perturbing eqach element of \theta
            dlparam=param
            dlparam[th]-=h
            y0 = [10,0,0,0,0,dlparam[18],dlparam[19]]
            states = odeint(deriv_ozone, y0, ts, args=(dlparam[0:18],))
            velocities= np.asarray([deriv_ozone(states[i,:],ts,dlparam[0:18]) for i in range(np.size(ts))])
            modelleft=velocities[:,5]-velocities[:,6]
            # Now Calculate sensitivity as dError/d\theta
            dEdparam[:,th]=vErrorfunc(data[:,episode],modelright,modelleft)
        # perform SVD of the sensitvity matrix [dError/d\theta]
        U, s, V = np.linalg.svd(dEdparam, full_matrices=True)
        # Also calculate the Hessian. N.B we are not using it now.
        Hess=dEdparam.T.dot(dEdparam)
        # Save the episode wise sensitivity matrix
        episodeSens.append(dEdparam)
        if args.direction=='sloppy':
            # Save eigenvectors acording to direction specified by user. stiff: 0-th, sloppy:19-th. N.B. \theta in R^20
            episodeUeigVec.append(U[:,19])
            episodeVeigVec.append(V[19,:])
            direction='sloppiest'
        else:

            episodeUeigVec.append(U[:,0])
            episodeVeigVec.append(V[0,:])
            direction='stiffest'
        # save the singluar values and the hessian
        episodeSingVal.append(s)
        episodeHess.append(Hess)


    # Plot the episode wise Sensitivities
    fig, axes = plt.subplots(nrows=5, ncols=2, facecolor='white')#figsize=(10, 12)
    fig.suptitle('Dataset '+str(plot_experiment_number+1)+' Sensitivities',
                 fontsize=24, fontweight='bold')
    Count=0
    for row in axes:
        for ax in row:
            ax.plot(ts, episodeSens[Count])
            ax.set_xlabel('time (hours)',fontsize='16')
            ax.set_ylabel('Episode '+str(Count+1),fontsize='16')
            ax.set_xlim([0,2])
            Count+=1
    plt.subplots_adjust(top=0.95)
    plt.figure()



    # Plot the singular value spectrum episode wise
    x=[]
    tix=[]
    for epi in range(10):
        plt.semilogy(np.ones(20)*(0.9 + 0.01*epi),episodeSingVal[epi]/episodeSingVal[epi][0],'*', markersize=25)
        x.append(0.9 + 0.01*epi)
        tix.append('Episode '+str(epi+1))
    plt.xlim([0.899,0.991])
    plt.xticks(x, tix, rotation=-45,fontsize='16')
    plt.ylabel(r'$log(s/s_1)$',rotation='vertical',fontsize='16')    #plt.ion()
    plt.suptitle('Singular value spectrums for '+'Dataset number '+str(plot_experiment_number+1),fontsize=24, fontweight='bold')



    # Plot the right Eigenvectors: this shows the parameter combination in stiff/sloppy directions
    tix=['koz','alph1','alph2','K1','K2','delta1','delta2','G','Voff','B','Jion','hx','l',
         'tau','rho','dcal','Kcat','hca','Qca1_0','Qca2_0']
    fig, axes = plt.subplots(nrows=5, ncols=2,figsize=(10, 12))
    fig.suptitle('Dataset number '+str(plot_experiment_number+1)+' Right Eigenvectors in '+direction+ ' direction',
                 fontsize=24, fontweight='bold')
    Count=0
    for row in axes:
        for ax in row:
            ax.bar(np.arange(20), episodeVeigVec[Count])
            ax.set_xticks([])
            ax.set_ylabel('Episode '+str(Count+1))
            ax.set_xlim([0,19.5])
            Count+=1
    plt.xticks(np.arange(20), tix, rotation=45,fontsize='16')
    plt.tight_layout()
    plt.subplots_adjust(top=0.98)



    # # Plot the Left Eigenvectors: this shows the change in error for parameter combination in stiff/sloppy directions
    fig, axes = plt.subplots(nrows=5, ncols=2,figsize=(10, 12))
    fig.suptitle('Dataset number '+str(plot_experiment_number+1)+' Left Eigenvectors in '+direction+ ' direction',
                 fontsize=24, fontweight='bold')
    Count=0
    for row in axes:
        for ax in row:
            ax.plot(ts, episodeUeigVec[Count])
            ax.set_xlabel('time (hours)',fontsize='16')
            ax.set_ylabel('Episode '+str(Count+1),fontsize='16')
            ax.set_xlim([0,2])
            Count+=1
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

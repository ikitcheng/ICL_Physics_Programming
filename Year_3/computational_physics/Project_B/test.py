# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 00:26:05 2017

@author: Matthew Cheng
"""
# In[]:
from integrator import NC, MC
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time as t
import pandas as pd
plt.rcParams['axes.grid'] = True # always plot with grids
start_time = t.clock()

def integrand(z):
    """
    Returns value of integrand f at z
    Input: z = value of integration variable
    Output: eval(f) = value of integrand evaluated at z
    """
    #f = 1/np.sqrt(np.pi)*np.exp(-z**2)
    return eval(f)

def pdf(shape, za, zb, z):
    """ 
    Sampling pdf for MC integration
    Input: shape = type of pdf wanted, za=lower limit, zb= upper limit, z= independent variable
    Output: P = pdf value at z
    """
    if shape=='uniform':
        P = 1/(zb-za)                                   # uniform pdf (normalised based on limits)
    if shape=='linear':
        A = -48/25/(zb-za)**2
        B = 1/25*(49*zb-za)/(zb-za)**2
        P = A*z+B                                       # linear pdf (normalised based on limits)
    if shape=='cos2':
        P = 1/(1+np.sin(4)/4)*np.cos(z)**2
    if shape=='psi_sq':
        P = 1/np.sqrt(np.pi)*np.exp(-z**2)
    return P        

def validation(integrand,lim,nt, ns): 
    '''
    Input: nt = # fn evaluations for trapzInt, ns= # fn evaluations for simpsInt
    Output: table of integrals using numpy and scipy functions 
    '''
    za, zb = lim[0], lim[1]
    global aa
    t1 = t.clock()
    aa = sp.integrate.quad(integrand, za,zb)
    tend1 = t.clock()-t1
    
    t2=t.clock()
    az = np.trapz(integrand(np.linspace(za,zb,nt)),x=np.linspace(za,zb,nt))
    tend2 = t.clock()-t2
    
    t3 = t.clock()
    ay = sp.integrate.trapz(integrand(np.linspace(za,zb,nt)),x=np.linspace(za,zb,nt))
    tend3 = t.clock()-t3
    
    t4 = t.clock()
    ax = sp.integrate.simps(integrand(np.linspace(za,zb,ns)),x=np.linspace(za,zb,ns))
    tend4 = t.clock()-t4
    
    print(pd.DataFrame({'Integral':[aa[0],az,ay,ax], 'Error':[aa[1],'','',''], 'Time/s':[tend1,tend2,tend3,tend4]},index=['sp quad', 'np trapz','sp trapz','sp simps'],
                        columns=['Integral', 'Error', 'Time/s'])) 
# In[]:
################################ User Inputs ################################## 

f =  '1/np.sqrt(np.pi)*np.exp(-z**2)'
lim = [0.,2.]
eps = 1e-5

print('\n','\t\tQ1. Newton-Coates integration','\n')  
# Trapz
integ1 = NC.NC('trapz')
integ1.Int(integrand,eps,lim)
integ1.result()

# Simps
integ2 = NC.NC('simps')
integ2.Int(integrand,eps,lim)
integ2.result()

print('\n','\t\tQ2. Monte Carlo Methods')  

# Basic MC - uniform pdf 
integ3 = MC.MC(1000.,adapt=False)
integ3.MCInt(integrand,1e-3,lim, S='uniform') # max eps = 1e-3 (based on 5 minute time window)
integ3.result()

# Basic MC - linear pdf
integ4 = MC.MC(1000.,adapt=False)
integ4.MCInt(integrand,1e-3,lim, S='linear') # max eps = 1e-4
integ4.result()

# Adaptive MC - uniform pdf
integ5 = MC.MC(1000.,adapt=True) 
integ5.MCInt(integrand,1e-4,lim,S='uniform') # max eps = 1e-6 
integ5.result()

# metropolis - integrand pdf 
integ6 = MC.MC(1000.,adapt=False)
integ6.metropInt(integrand,1e-3,integrand,lim) # max eps = 1e-4
integ6.result()

integ1.plot()
integ2.plot()
integ3.plot()
integ4.plot()
integ5.plot()
integ6.plot()
    
print('\n','############### Validation ###############')
validation(integrand,lim,integ1.neval, integ2.neval)
print('##########################################')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("--- %s seconds ---" % (t.clock() - start_time))

########################### PLOTS #############################################

## NC err plots (Fig.2 in report)
#fig = plt.figure()
#plt.title('Relative error vs.  No. of function evaluations')
#plt.loglog(integ1.nList, integ1.errList,'bo-',label='trapz')
#plt.loglog(integ1.nList, 1/(np.array(integ1.nList))**2,'r--',label=('N^-2'))
#plt.loglog(integ2.nList[1:], integ2.errList[1:],'yo-',label='simps')
#plt.loglog(integ2.nList, 1/(np.array(integ2.nList))**4,'m--',label=('N^-4'))
#plt.xlabel('No. of fn eval, N') # N is the total number of points
#plt.ylabel('rel. error')
#plt.legend()
#plt.show()


### MC integral plot (Fig. 5 in report)
#fig1 = plt.figure()
#plt.title('Integral estimate vs.  No. of function evaluations')
#plt.semilogx(np.linspace(integ3.N,integ3.neval,integ3.niter), integ3.wgtIList,'b-',label='MC(uniform)')
#plt.semilogx(np.linspace(integ4.N,integ4.neval,integ4.niter), integ4.wgtIList,'r-',label='MC(linear)')
#plt.semilogx(np.linspace(integ5.N,integ5.neval,integ5.niter), integ5.IList,'-',color='purple',label='AMC')
#plt.semilogx(np.linspace(integ6.N,integ6.neval,integ6.niter), integ6.wgtIList,'-', color='saddlebrown',label='Metropolis')
#plt.xlabel('No. of fn eval, N') # N is the total number of points
#plt.ylabel('Integral estimate')
#plt.legend()
#plt.show()


### MC distribution plot (Fig.7 in report)
#fig2, ax = plt.subplots(4, sharex=True, figsize=(7, 7))
#fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=5)
#ax[0].set_title('z-sampling distribution')
#ax[0].hist(integ3.zList,bins='auto', edgecolor='black', linewidth=0.2,color='blue', normed=True, alpha=0.5, label='MC (uniform)')
#ax[0].plot(np.linspace(lim[0],lim[1],100), integrand(np.linspace(integ5.lim[0],integ5.lim[1],100)), label='integrand')
#ax[0].plot([x for x in np.linspace(lim[0],lim[1],10)], [1/(lim[1]-lim[0])]*10,color='red', label='pdf=%.2f'%(1/(lim[1]-lim[0])), alpha=1)
#ax[1].hist(integ4.zList,bins='auto', edgecolor='black', linewidth=0.2,color='blue', normed=True, alpha=0.5, label='MC (linear)')
#ax[1].plot(np.linspace(lim[0],lim[1],100), integrand(np.linspace(lim[0],lim[1],100)), label='integrand')
#ax[1].plot([x for x in np.linspace(lim[0],lim[1],10)], -0.48*np.linspace(lim[0],lim[1],10)+0.98,color='red', label='pdf=-0.48z+0.98', alpha=1)
#ax[2].hist(integ5.zList,bins='auto', edgecolor='black', linewidth=0.2,color='blue', normed=True, alpha=0.5, label='Adaptive MC')
#ax[2].plot(np.linspace(lim[0],lim[1],100), integrand(np.linspace(lim[0],lim[1],100)), label='integrand')
#ax[3].hist(integ6.zList,bins='auto', edgecolor='black', linewidth=0.2,color='blue', normed=True, alpha=0.5, label='Metropolis')
#ax[3].plot(np.linspace(lim[0],lim[1],100), integrand(np.linspace(lim[0],lim[1],100)), label='integrand')
#ax[3].plot([x for x in np.linspace(lim[0],lim[1],10)], integrand(np.linspace(lim[0],lim[1],10))/0.49766,color='red', label='pdf=integrand (normed)')
#if integ5.adapt:
#    for i in range(len(integ5._A)):
#        ax[2].plot([integ5._A[i],integ5._A[i]],[0, integrand(integ5._A[i])] ,'r-')
#    ax[2].plot([integ5._B[-1],integ5._B[-1]],[0, integrand(integ5._B[-1])], 'r-',label='sub-region boundaries')  
#plt.xlabel('z') # N is the total number of points
#for i in range(len(ax)):
#    ax[i].legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)
#    ax[i].set_ylabel('rel. freq')
#plt.show()

## MC #fn eval vs eps (Fig. 6 in report)
#neval1, neval2,neval3,neval4, eps1, eps2, eps3,eps4= [], [], [],[],[],[],[],[]
#for i in np.arange(1,3,0.2):
#    integ3.MCInt(integrand, 10**(-i), lim, S='uniform')
#    integ4.MCInt(integrand, 10**(-i), lim, S='linear')
#    integ5.MCInt(integrand, 10**(-i), lim)
#    integ6.metropInt(integrand, 10**(-i), integrand,lim)
#    
#    neval1.append(integ3.neval)
#    eps1.append(10**(-i))
#    
#    neval2.append(integ4.neval)
#    eps2.append(10**(-i))
#    
#    neval3.append(integ5.neval)
#    eps3.append(10**(-i))
#    
#    neval4.append(integ6.neval)
#    eps4.append(10**(-i))
#plt.title('No. of function evaluations vs. relative accuracy')    
#plt.loglog(eps1, neval1, label='MC(uniform)') 
#plt.loglog(eps1, 1/np.array(eps1)**2,'--', label='eps^-2')
#plt.loglog(eps2, neval2, label='MC(linear)')
#plt.loglog(eps3, neval3, label='AMC')
#plt.loglog(eps4, neval4, label='Metropolis')
#plt.xlabel('eps')
#plt.ylabel('No. fn eval')
#plt.legend()

#fig.savefig('NCerr.jpg', dpi=150, bbox_inches='tight')

     

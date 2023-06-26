# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 00:21:46 2017

@author: ikc15
"""
import numpy as np
import scipy as sp
import time as t
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True                                        # always plot with grids

class MC:
    """
       MC objects are integrators that uses Monte Carlo (MC) methods.
       User can choose between basic MC, adaptive MC (vegas)
    """
    
    def __init__(self, N, adapt=True):
        ''' Constructor for this class. '''
        self.adapt = adapt                                              # True or False: implement adaptive MC 
        self.pdf = None                                                 # sampling pdf used
        self.__f = None                                                 # integrand fn    
        self._A,self._B = None,None                                   # sub-region boundaries: A contains lower boundary, B contains upper boundaries
        self.lim=[]                                                     # integration limits
        self.acc = None                                                 # acceptance rate
        self.rej = None                                                 # rejection rate
        self.errList = []                                               # stores the relative errors of each iteration
        self.zList= []                                                  # sampled z values
        self.IList = []                                                 # integral estimates 
        self.wgtIList = []                                              # weighted estimates of I
        self.N = N                                                      # no.random pts in each new region (or per iteration if adapt=False) 
        self.T = 1e6                                                    # max no. of iterations                
        self.I = None                                                   # final estimate of integral
        self.err = None                                                 # final relative error
        self.neval = None                                               # final number of function evaluations (niter*N)
        self.niter = None                                               # final number of interations
        self.t = None                                                   # final time taken to perform integral
    
    def MCInt(self, f, eps,domain=[], S='uniform'):
        """
        Int method performs numerical integration on a user specified 
        function f (in 1d) using MC estimates. 
        
        The basic MC algorithm randomly samples the integrand according to a
        smapling pdf (which is hard-coded into this routine using 
        transformation method). 
        
        The adaptive MC (AMC) algorithm uses basic MC with a uniform
        sampling pdf. The difference is that vegas divides the integration 
        domain after each iteration in order to reduce the standard deviation 
        of the integral estimate in the most efficient way. 
        
        Methods: uniform sampling, linear sampling, 
                fixed importance sampling, adpative importance sampling
        
        Input:  f=integrand fn, eps= relative accuracy, 
                domain = integration domain, S= sampling pdf (default= uniform)
                adapt= True or False (default: adapt=True, use S = uniform 
                when adapt=True)
        Output: Value of integral, rel. error, # fn eval, time
        """
        tstart = t.clock()
        if (self.adapt) and (S=='linear'): 
            raise ValueError ('if adapt= True -> S= uniform')
        self.pdf,self.__f,self.lim,T,N = S,f,domain,self.T,self.N       # set parameters                    
        np.random.seed(15)                                              # seed the random number generator to produce the same sequence of random no in each run.
        A,B,TM,TE, = np.empty(1),np.empty(1),np.empty(1),np.empty(1),   # A and B = lower and upper bounds for each subregion, TM and TE = list of estimated integral and err in each subregion
        za, zb = domain[0], domain[1]                                   # lower and upper integration limits     
        niter, inverr2, wgtQ, wgterr,E,wgtQmean,I= 0,0,0,1e6,1e6,1,1    # initialise variables      
        
        while (niter<T) and (wgterr/wgtQmean>eps):                      # relative err
            Q1, Q2, f1, f2 = 0,0,0,0
            for i in range(int(N)):                                     # fixed no. of random walks from above starting pt
                x = np.random.uniform(0,1)
                if S=='uniform':
                    z = (zb-za)*x + za                                  # choose random z position from a uniform pdf mapping -> uniform(0,2)
                    P = 1/(zb-za)                                       # uniform pdf (normalised based on limits)
                if S=='linear':
                    A = -48./25/(zb-za)**2                              # coefficients for linear pdf: normalised in terms of limits
                    B = 1./25*(49*zb-za)/(zb-za)**2
                    b = 2.*B/A 
                    c = -(za*za + b*za + 2*x/A)
                    z = (-b - np.sqrt(b*b - 4*c))/2                     # choose random z position from a uniform pdf mapping -> linear pdf
                    P = A*z+B                                           # linear pdf (normalised based on limits)
                fz = f(z)
                f1 += fz                                                # add value of integrand to running sum
                f2 += fz**2                                             # add squared value of integrand to running sum
                Q1 += fz/P                                              # add value of integrand weighted by pdf at z to running sum
                Q2 += (fz/P)**2                                         # add sqaured value of integrand weighted by pdf at z to running sum
                self.zList.append(z)                                    # add sampled z to zList
            Qmean = Q1/N                                                # est. of integral from N samples (i.e. Qmean is an estimate of I)
            err = np.sqrt(abs(1/(N-1)*(Q2- N*Qmean**2)/N))              # error of integral using sqrt(unbiased variance of Qi/ N)
            wgtQ += Qmean/err**2                                        # weighted Qmean by its error
            inverr2 += 1/err**2                                         # inverse square of the error
            wgtQmean = wgtQ/inverr2                                     # weighted average of integral 
            wgterr = np.sqrt(1/inverr2)                                 # error on wgtQmean
            self.IList.append(Qmean)                                    # add integral to Ilist
            self.wgtIList.append(wgtQmean)                              # add wgt. integral to wgtIList
            self.errList.append(wgterr)                                 # add error on wgtQmean to list
            if not self.adapt:  niter+=1                                # if adapt=False, go to next iteration
            
            if self.adapt:                                              # if adapt=True: Do
                A[0], B[0] = za, zb                                     # Step 1: lower and upper bound of whole integration domain (no subregions)
                ai=0                                                    # ai = index of region with max estimated error 
                V = B[0] - A[0]                                         # V = width of integration domain 
                Ihat = V/N*f1                                           # Step 2: Get est. of integral 
                Ehat = V*np.sqrt((f2/N - (f1/N)**2)/N)                  #         Get est. of error      
                TM[0]= Ihat                                             # store these estimates in TM and TE arrays
                TE[0]= Ehat                                                              
                while(niter<T) and (E/I >eps):                          # E/I = relative error (i.e. absolute err divided by integral value) 
                    c = (A[ai] + B[ai]) / 2                             # Step 3: divide region with max error into two (for iter=0 the region is the whole integration domain)
                    A = np.insert(A,ai+1,0)                             # increase array size by 1 at index (ai+1) to accomodate 1 extra region due to division in Step 3. 
                    B = np.insert(B,ai,0)                               # same as above but at index ai
                    TM = np.insert(TM,ai,0)                             # same as above but at index ai
                    TE = np.insert(TE,ai,0)                             # same as above but at index ai
                    for k in range(niter+2):                            # niter+2 is the new total number of regions in integration domain
                        if k == (ai+1): A[k] = c                        # calculate new A and B by adding in new bounds for the sub-regions    
                        if k == ai:     B[k] = c
                    for j in range(ai, ai+2):                           # for each new region:
                        V = abs(B[j] - A[j])                            #   V = width of integration domain 
                        f1, f2 = 0,0                                    #   vars to store sum of sampled integrands
                        for k in range(int(N)):                         #   take N samples of z
                            z = np.random.uniform(A[j],B[j])            #   random z within subregion j       
                            fz = f(z)                                   #   evaluate integrand at z    
                            f1 += fz                                    #   add value of integrand to running sum
                            f2 += fz**2                                 #   add squared value of integrand to running sum    
                            self.zList.append(z)                        #   add sampled z to zList
                        Ihat = V/N*f1                                   #   Get new est. of integral 
                        Ehat = V*np.sqrt((f2/N - (f1/N)**2)/N)          #   Get new est. of error
                        TM[j] = Ihat                                    #   store est. of int in TM array
                        TE[j] = Ehat                                    #   sotre est. of err in TE array
                    ai = np.argmax(TE)                                  # Step 4: Find new ai = index of region with max error     
                    niter += 1                                          #         increase count by 1 (each iteration gives 1 indep. est. of integral)        
                    I = TM.sum()                                        # Step 5: calculate total estimated integral 
                    E = np.sqrt(np.sum(TE**2))                          #         calculate total estimated error
                    self.IList.append(I)                                # add total integral to list
                    self.errList.append(E)                              # add total err to list    
                    
                if (niter>=T): raise ValueError('niter exceeded %d.'%T)    
                self.I, self.err, self.neval, = I, E/I, (niter+1)*N     # update final integration results
                self.niter, self.t, self.acc, self.rej = niter+1, t.clock()-tstart, 100, 0
                self._A, self._B = A, B
                return (self.I,self.err)
        
        if (niter>=T): raise ValueError('niter exceeded %d.'%T)    
        self.I, self.err, self.neval, = wgtQmean, wgterr/wgtQmean, niter*N
        self.niter, self.t, self.acc, self.rej = niter, t.clock()-tstart, 100,0
        return (self.I,self.err)

    def metropInt(self,f, eps,P,domain=[],S='integrand'):
        """
        metropInt method performs numerical integration on a user specified 
        function f (in 1d) using a MCMC (specifically metropolis algorithm). 
        
        The metropolis algorithm samples the integrand according to a user-
        specified smapling pdf (can be un-normalised). It estimates the 
        integral by finding the mean of the integrand samples weighted by the
        pdf. 
        
        Methods: fixed importance sampling 
        
        Input:  f=integrand fn, eps= relative accuracy, P = sampling pdf fn
                domain = integration domain, S= sampling pdf (default= integrand)
        Output: Value of integral, rel. error, # fn eval, time
        """
        tstart = t.clock()
        self.pdf,self.__f,self.lim,T,N = S,f,domain,self.T,self.N       # set parameters                    
        np.random.seed(15)                                              # seed the random number generator to produce the same sequence of random no in each run.
        za, zb = domain[0], domain[1]                                   # lower and upper integration limits     
        niter, rejected,inverr2, wgtQ, wgterr,wgtQmean= 0,0,0,0,1e6,1   # initialise variables   
        
        ## use metrop alg. to generate samples from integrand directly 
        while(niter<T and wgterr/wgtQmean>eps):
            niter +=1
            Q1, Q2,zzList= 0,0,[]
            z = np.random.uniform(za, zb)                               # randomly pick a starting pt z for each iteration 
            for i in range(1,int(N+1)):                                 # fixed no. of random walks from above starting pt
                znew = np.random.normal(z,0.5)                          # take a random step from normal distribution centered on z (if low acceptance-> decrease variance (smaller steps))            
                while(znew>zb): znew = znew-(zb-za)                     # implement periodic BC
                while(znew<za): znew = znew+(zb-za)       
                if P(znew) < P(z) :                                     # if prob at znew is smaller
                    if (P(znew)/P(z))<np.random.uniform(0,1):           # accept new step with prob P(znew) compared to a random no. bet 0 and 1
                        znew = z                                        # step rejected -> go back to z
                        rejected +=1
                z = znew
                zzList.append(z)                                        # add sampled z to zzList (for plotting final histogram)
                self.zList.append(z)                                    # add sampled z to zList (for normalising in this iteration)
            
            ## produce a normalised pdf using a normed histogram of the samples
            hist, bin_edges = np.histogram(self.zList,bins='auto', 
                                           range=(za,zb),normed=True)   # extract density values in each bin, and values of bin edges
            for z in zzList:                                            # for each z in zList:
                for i in range(len(bin_edges)):                         # find which bin z belongs in (N.B. bin_edges in ascending order)
                    if z == bin_edges[i]:                               # if z is one of the bin edges 
                        Pz = hist[i]                                    # return corresponding density   
                        break
                    elif bin_edges[i] < z:                              # otherse loop through data until we find a bin edge > z
                        continue
                    elif bin_edges[i] > z:                              # if bin edge > z: 
                        Pz = hist[i-1]                                  # hist[i-1] is the density at z   
                        break
                fz = f(z)                                               # evaluate integrand at z
                Q1 += fz/Pz                                             # add value of integrand weighted by density at z to running sum
                Q2 += (fz/Pz)**2                                        # add sqaured value of integrand weighted by density at z to running sum
            Qmean = Q1/N                                                # est. of integral from N samples
            err = np.sqrt(abs(1/(N-1)*(Q2- N*Qmean**2)/N))              # error of integral using sqrt(unbiased variance of Qi/ N)
            if err==0: err=1
            wgtQ += Qmean/err**2                                        # weighted Qmean by its error
            inverr2 += 1/err**2                                         # inverse square of the error
            wgtQmean = wgtQ/inverr2                                     # weighted average of integral 
            wgterr = np.sqrt(1/inverr2)                                 # error on wgtQmean
            self.IList.append(Qmean)                                    # add integral to Ilist
            self.wgtIList.append(wgtQmean)                              # add wgt. integral to wgtIList
            self.errList.append(wgterr)                                 # add error on wgtQmean to list  
        
        if (niter>=T): raise ValueError('niter exceeded %d.'%T)    
        self.I, self.err, self.neval, = wgtQmean, wgterr/wgtQmean, niter*N 
        self.niter, self.t = niter, t.clock()-tstart
        self.rej = rejected/self.neval*100         
        self.acc = (100-self.rej) 
        return (self.I,self.err)        
        
    def result(self):
        ''' Returns a summary of the integration results from Int using Pandas. '''
        if self.adapt: index = 'AMC(uniform)'
        else: index = 'MC(%s) '%(self.pdf)    
        if self.pdf == 'integrand': index = 'Metrop(integrand)'
        print(pd.DataFrame({'Integral':[self.I], 'Rel. Err':[self.err], 
                             '# fn eval':[int(self.neval)], 'Time/s':[self.t],
                             'Acceptance/%':[self.acc]}, 
                             index=[index],
                             columns=['Integral', 'Rel. Err', '# fn eval', 'Time/s','Acceptance/%']))

    def plot(self):
        ''' Returns a plot of the integration results. '''
        if self.adapt: index = 'AMC'
        if self.pdf =='integrand': index = 'metrop(integrand)'
        else: index = 'MC(%s)'%self.pdf
        fig, ax = plt.subplots(3,)                               
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=3)
        ax[0].set_title('Integral estimate vs. No. of function evaluations')
        if self.adapt: ax[0].semilogx(np.linspace(self.N,self.neval,self.niter), self.IList, 'b-', label=index)
        else: ax[0].semilogx(np.linspace(self.N,self.neval,self.niter), self.wgtIList, 'b-', label=index)
        ax[0].set_ylabel('Integral estimate')
        ax[1].set_title('Relative integration error vs.  No. of function evaluations')
        ax[1].loglog(np.linspace(self.N,self.neval,self.niter), self.errList,'b-',label=index)
        ax[1].loglog(np.linspace(self.N,self.neval,self.niter), 1/np.sqrt(np.linspace(self.N,self.neval,self.niter)),'r--',label=('N^-0.5'))
        ax[1].set_ylabel('Rel. error')
        ax[1].set_xlabel('No. of fn eval, N')                                 
        ax[2].set_title('Sampling distribution')
        ax[2].hist(self.zList,bins='auto', edgecolor='black', linewidth=0.2,color='blue', normed=True, alpha=0.5)
        ax[2].plot(np.linspace(self.lim[0],self.lim[1],100), self.__f(np.linspace(self.lim[0],self.lim[1],100)), label='integrand')
        if self.adapt:
            for i in range(len(self._A)):
                ax[2].plot([self._A[i],self._A[i]],[0, self.__f(self._A[i])] ,'r-')
            ax[2].plot([self._B[-1],self._B[-1]],[0, self.__f(self._B[-1])], 'r-', label='sub-region boundaries')
        ax[2].set_xlabel('z')
        ax[2].set_ylabel('rel. freq')
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.show() 
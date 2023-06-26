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
plt.rcParams['axes.grid'] = True # always plot with grids

class NC:
    """
       NC objects are integrators that uses Newton-Coates rules.
       User can choose between trapezoidal and Simpson's rule.
    """
    
    def __init__(self, method='trapz or simps'):
        ''' Constructor for this class. '''
        self.method = method                                            # prints the NC method used
        self.errList = []                                               # stores the relative errors of each iteration
        self.nList = []                                                 # stores the no. of points of each iteration
        self.IList = []                                                 # stores the integral estimate of each iteration
        self.I = None                                                   # final estimate of integral
        self.err = None                                                 # final relative error
        self.neval = None                                               # final number of function evaluations
        self.t = None                                                   # final time taken to perform integral
        
    def Int(self, f,eps,domain=[]):
        """
        NCInt method performs numerical integration on a user specified 
        function f (in 1d) using extended trapezoidal rule or extended Simpson's rule.
        Input: f=integrand fn, eps= reltaive accuracy, domain = integration doamin.
        Output: Value of integral, rel. error, # fn eval, time
        """
        tstart= t.clock()
        T = 1e3                                                         # maximum no. of iterations
        za, zb = domain[0], domain[1]                                   # integration limits
        h = zb-za                                                       # range of integration limits
        counter,err = 0, 1e6                                            # counter = no. of iterations, err= relative error of integral estimates
        S_prev, f_prev, h_prev, n_prev = 1, 0, 0, 0                  # vars to store previous iteration quantities
        while (counter < T) and (err>eps):                              # set max limit on no. of iterations and relative accuracy criteria
            counter += 1                                                # increase no. iteration by 1    
            if counter==1:                                              # In 1st interation 
                f_prev = (f(za) + f(zb))/2                              # Evaluate first estimate of integral with 1 trapezoid    
                h_prev, n_prev = h, 2                                   # store step size and no. of points
            n = 2*n_prev - 1                                            # new no. of points   
            h2 = h_prev/2                                               # new step size (half of h_prev)
            i = np.arange(n)                                            # labels for the n points
            z = np.array(za + i[1::2]*h2)                               # select every other pt to avoid evaluating at same z from previous iteration
            f2 = sum(f(z)) + f_prev                                     # evaluate and sum integrand at new points plus the previous sum

            if self.method == 'trapz':
                err = abs((h2*f2-h_prev*f_prev)/(h_prev*f_prev))        # relative error of integral estimates
                self.IList.append(f2*h2)                                # add trapz estimate to IList   

            if self.method == 'simps':
                S = 4./3*h2*f2 - 1./3*h_prev*f_prev                     # Evaulate Simpson's Rule using two sucessive trapezoidal iterations
                err = abs((S-S_prev)/S_prev)                            # relative error
                S_prev = S                                              # store Simpson's estmiate
                self.IList.append(S)                                    # add Simpson's estimate to IList

            f_prev,h_prev,n_prev = f2,h2,n                              # store trapezoid integral, step size, no. of points
            self.errList.append(err)                                    # add error to errList
            self.nList.append(n)                                        # add no. of points to nList
            
        if (counter>=T): raise ValueError('counter exceeded %d.'%T)                 
        if self.method == 'trapz': 
            self.I, self.err, self.neval,self.t = f2*h2, err, n, t.clock()-tstart 
            return (self.I, self.err)
        if self.method == 'simps': 
            self.I, self.err, self.neval,self.t = S, err, n, t.clock()-tstart  
            return (self.I, self.err)
    
    def result(self):
        ''' Returns a summary of the integration results from Int. '''
        print(pd.DataFrame({'Integral':[self.I], 'Rel. Err':[self.err], 
                             '# fn eval':[self.neval], 'Time/s':[self.t]}, 
                             index=[self.method],
                             columns=['Integral', 'Rel. Err', '# fn eval', 'Time/s']))
        
    def plot(self):
        ''' Returns a plot of the integration results. '''
        if self.method =='trapz': e = 2.
        if self.method =='simps': e = 4.
        fig, ax = plt.subplots(2,sharex=True)                               
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=3)
        ax[1].set_title('Relative integration error vs.  No. of function evaluations')
        if self.method=='simps': ax[1].loglog(self.nList[1:], self.errList[1:],'bo-',label=self.method)
        if self.method=='trapz': ax[1].loglog(self.nList, self.errList,'bo-',label=self.method)
        ax[1].loglog(self.nList, 1/(np.array(self.nList))**e,'r--',label=('1/N^%d'%e))
        ax[1].set_ylabel('Rel. error')
        ax[0].set_title('Integral estimate vs. No. of function evaluations')
        ax[0].semilogx(self.nList, self.IList, 'bo-', label=self.method)
        ax[0].set_ylabel('Integral estimate')
        plt.xlabel('No. of fn eval, N')                                 
        ax[0].legend()
        ax[1].legend()
        plt.show()
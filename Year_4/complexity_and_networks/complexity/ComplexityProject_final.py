"""
Created on Thu Jan 10 17:33:31 2019

Complexity and Network Project

@author: ikc15
##############################################################################
Oslo Model (d=1): 
The system is driven by adding grains, one-by-one, to the boundary site i = 1.
A site relaxes when z(i) > zth(i) by letting grain topple from site i -> i+1.
recalculate zth(i)

Algorithm:
    1. Initialisation: Prepare empty config with z(i)=0 for all i. Choose zth(i)
                        for all i.
    2. Drive: Add 1 grain at site i=1
                z(1)->z(1)+1
    3. Relaxation: if z(i) > zth(i), relax site i. Choose new threshold slope 
                    for relaxed site only.
    4. Iteration: return to 2    
    
"""
import numpy as np
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import time
import pandas as pd
from logbin622018 import logbin
np.random.seed(1000)

class OsloModel:
    def __init__(self,L=10,p=0.5):
        # 1. Initialisation 
        self.L      = L     # size of the system
        self.p      = p     # probability that zth(i) = 1
        self.ngrains= 0     # number of grains in system
        self.s      = 0     # avalanche size (number of topples)
        self.h      = [0]*self.L # prepare empty config for each site
        self.z      = [0]*self.L # slope at each site
        self.zth    = [0]*self.L # threshold slope at each site
        self.out    = 0          # out=1 means grain flow out of the system.
       # self.d      = 0          # number of grains that left the system.
        self.tc     = 0          # cross-over time = no. of grains in the system 
                                 # before an added grain induces a grain to leave the system for the first time
        for i in range(self.L):  # choose random threshold slope for each site
            self.zth[i] = self.calc_zth()
        self.h1avg  = 0
        self.h1list = [] 
        self.h1sqavg= 0
        self.h1stddev=0 
        self.htilde  =[] 
        self.Lfn     =0
        self.savg    = 0
        self.slist   = [] 
        self.tcavg   =0
        self.tclist  = []
        self.zavg    = 0
        self.zavglist=[]
    def drive(self):
        # add 1 grain to site i = 1 (drive system from left-most boundary)
        self.ngrains += 1
        self.h[0]   += 1
        self.z[0]   += 1
    
    def relax(self,plot=True):
        relax  = True
        self.s = 0
        while relax==True:
            relax = False
            # plot current state of ricepile
            if plot == True:
                self.plotRicepile(0)
            for i in range(self.L): # for each site
                if self.z[i]>self.zth[i]: # site topples if z(i) BIGGER than zth
                    relax = True
                    # change the height of the ricepile
                    if i<self.L-1:
                        self.h[i]   -= 1
                        self.h[i+1] += 1
                    else: self.h[i] -= 1
    
                    # change the slope of the pile
                    if i == 0: # left boundary site. 
                        self.z[0] -= 2
                        self.z[1] += 1
                    elif i == self.L-1: # right boundary site
                        self.z[i]   -= 1
                        self.z[i-1] += 1
                        if self.out == 0: # Only update tc once when a grain leaves the system for the first time
                            self.tc  = self.ngrains - 1 # -1 because tc is the no. of grains before an added grain leaves the system
                            self.out = 1
                    else: # all sites in between
                        self.z[i]   -= 2
                        self.z[i-1] += 1
                        self.z[i+1] += 1
                    
                    #plot the toppled ricepile
                    if i<self.L-1 and plot==True: 
                        self.plotRicepile(self.i+1) 
#                         argument here is i+1, because want to check if the site i+1 
#                         would topple next. If yes, we plot the top grain red.
                    
                    # recalculate the threshold slope of site i.    
                    self.zth[i] = self.calc_zth()    
                    self.s+=1
    
    def runPile(self,N,animate=True):
        self.h1list    = []    # height of site i = 1 for each grain added
        self.slist     = []    # avalanche sizes for each grain added
        self.zavglist  = []    # average slope of all sites at each timestep
        t1 = t()
        while self.ngrains<N:#self.tc+N:
            self.drive()
            self.relax(animate)
            self.h1list.append(self.h[0])
            self.slist.append(self.s)
            self.zavglist.append(np.mean(self.z))  
            dt = t()-t1
            if self.ngrains%10000==0:
                print('ngrains = %d. Estimated time left: %.1f s'%(self.ngrains,dt*(N/self.ngrains-1)))
                
    def calc_zth(self):
        """
        calculates threshold slope for each site randomly.
        """
        randno = np.random.random(1)[0] # random number between 0 and 1.
        if randno>self.p or self.p==0:   zthi = 2
        else: zthi = 1
        return zthi
    
    def calc_avg(self):
        "Calculate parameter averages after steady state reached at t>tc"
        self.h1avg   = np.mean(self.h1list[self.tc+1:])   # avg height at i=1
        self.h1sqavg = np.mean(np.array(self.h1list[self.tc+1:])**2) # avg height^2 at i=1
        self.zavg    = np.mean(self.zavglist[self.tc+1:]) # avg slope across whole pile
        self.zsqavg  = np.mean(np.array(self.zavglist[self.tc+1:])**2)
        self.savg    = np.mean(self.slist[self.tc+1:])    # avg avalanche size
    
    def calc_htilde(self,M):
        """
        Smooth the h(t;L) function by averaging over M different piles of size L 
        (each with a unique set of threshold slope).
        input: M = number of piles of size L
               N = number of grains
        output: htilde, tcavg
        """
        hprev  = np.zeros(N)
        tclist = [] 
        for i in range(M): # for each pile 
            self.__init__(self.L,self.p)
            self.runPile(N,animate=False)
            tclist.append(self.tc)
            hprev += np.array(self.h1list)
        self.htilde = hprev/M
        self.tclist = tclist
        
    def calc_tcavg(self):
        self.tcavg = np.mean(self.tclist)
        self.Lfn   = self.L**2*(1+1/self.L)/2 # L(L+1)/2

    def calc_h1stddev(self):
        self.h1stddev = np.sqrt(self.h1sqavg-self.h1avg**2)
    
    def calc_zstddev(self):
        self.zstddev = np.sqrt(self.zsqavg-self.zavg**2)
        
    def calc_h1prob(self):
        """
        Calculates the probability that h(i=1;L) has a height h in steady state.
        Each grain added after tc is a stable config. 
        Input: h = height
        """
        #return self.h1list[self.tc+1:].count(h)/len(self.h1list[self.tc+1:])
        unq,counts = np.unique(self.h1list[self.tc+1:],return_counts=True) # MUCH faster way than .count()
        unq        = np.concatenate(([min(unq)-1],unq,[max(unq)+1]))
        prob       = np.concatenate(([0],counts/len(self.h1list[self.tc+1:]),[0]))
        return unq, prob
        
    def calc_sprob(self):
        """
        Calculates the probability of a size s avalanche in steady state.
        Each grain added after tc is a stable config.
        Input: s = avalanche size
        """
        #return self.slist[self.tc+1:].count(s)/len(self.slist[self.tc+1:])
        unq,counts = np.unique(self.slist[self.tc+1:],return_counts=True)
        return unq, counts/len(self.slist[self.tc+1:])
        
    def calc_kmoment(self,k):
        """
        Calculates the kth moment of avalanche size <s^k>
        Input: k = kth moment
        """
        slist = np.array(self.slist[self.tc+1:])
        sklist = slist.astype(np.float64)**k # must be a float array otherwise overflow error i.e. not enough memory to store the large number
        return np.sum(sklist,axis=1)/len(self.slist[self.tc+1:])
    
    def plotRicepile(self,i):
        """
        plot the ricepile.
        input: i = current site of interest
        """
        plt.figure('ricepile')
        ax = plt.gca()
        ax.clear()
        ax.set_title('Oslo Model: grains = %d'%self.ngrains)
        plt.bar(np.arange(1,self.L+1), self.h)   
        if self.z[i]>self.zth[i]: # if the current site of interest is about to topple (z > zth)
            plt.bar(i+1,1,bottom=self.h[i]-1,color='r')  # make the top grain at this site red (stacked barchart)
        plt.xlabel('Sites (i)')
        plt.ylabel('Height (h)')
        plt.xticks(np.arange(1,self.L+1))
        plt.yticks(np.arange(0, max(self.h)+1))
        plt.grid(True)
        plt.pause(0.05)
        #plt.savefig('size=%d p=%s ricepile.png'%(self.L,self.p))
    
    def ploth1(self,htilde=False):
        """
        plot the height at site i=1 vs grains added.
        """
        plt.figure('h1')
        if htilde==False:
            plt.plot(np.arange(1,self.ngrains+1), self.h1list,label='L=%d'%self.L)  
        else:
            plt.plot(np.arange(1,self.ngrains+1), self.htilde,label='L=%d'%self.L)  
        plt.xlabel('Time t (number of grains)')
        plt.ylabel('Height at site $i=1$ h(t;L)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        #plt.savefig('size=%d p=%s- h1.png'%(self.L,self.p))
        
    def plotDataCollapse(self):
        plt.figure('Data collapse')
        plt.plot(np.arange(1,self.ngrains+1)/(self.L**2), self.htilde/self.L, label='L=%d'%self.L)
        plt.xlabel(r'Normalised time $t/L^2$ ')
        plt.ylabel(r'Normalised height at site $i=1$ $\tilde{h}(t;L)/L$')
        plt.legend(loc='best')
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
    
    def printResults(self):
        self.results = pd.DataFrame({'L':[self.L],'p':[self.p],'Grains':[self.ngrains], 
                                     'tc':[self.tc],'<h(i=1)>':[self.h1avg], 
                                     '<z>':[self.zavg],'<s>':[self.savg]},
                                     index=[''],
                                     columns=['L','p','Grains','tc','<h(i=1)>','<z>','<s>'])
        print(self.results)
    
    def reload(self,file):
        self.h, self.h1avg, self.h1list, self.h1sqavg, self.h1stddev = file[0], file[1], file[2],file[3], file[4]
        self.htilde, self.L, self.Lfn, self.ngrains, self.out, self.p= file[5], file[6], file[7],file[8], file[9], file[10]
        self.s, self.savg, self.slist, self.tc, self.tcavg, self.tclist= file[11], file[12],file[13], file[14],file[15],file[16]
        self.z,self.zavg,self.zavglist,self.zth= file[17],file[18], file[19], file[20]
# In[]: General functions used for plotting and fitting
def t():
    return time.clock()

def saveData(filename, data):
    outfile = open(filename,'wb')
    pickle.dump(data,outfile)
    outfile.close()
    
def loadData(filename):
    infile = open(filename,'rb')
    outfile = pickle.load(infile)
    infile.close()
    return outfile

def Gaussian(x,mu,sigma):
    return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

def linearfit(x,y):
    (m,b)=np.polyfit(x ,y ,1)
    yfit = np.polyval([m,b],x)
    eqn = 'y = ' + str(round(m,4)) + 'x' ' + ' + str(round(b,4))
    return m, b, yfit, eqn

def plotloglog(x,y,xlabel='',ylabel='',legend='',log=True, newplot=True,fit=True,marker='x-',color='',Alpha=1, i1=0,i2=None):
    """
    Input: i1 = starting index for regression of data array.
    """
    regression_stats=[]
    if newplot==True:
        plt.figure()
    if color:
        plt.plot(x,y,color+marker,alpha=Alpha,label=legend)
    else:
        plt.plot(x,y,marker,alpha=Alpha,label=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log == True:
        plt.xscale('log')
        plt.yscale('log')
        if fit==True:
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(np.array(x[i1:i2])),np.log(np.array(y[i1:i2])))
            (m,b,yfit,eqn) = linearfit(np.log(np.array(x[i1:i2])),np.log(np.array(y[i1:i2]))) # find power law exponent from slope
            plt.plot(x[i1:i2],np.exp(yfit),label= eqn)
            regression_stats=[slope, intercept, r_value, p_value, std_err]
    elif log==False and fit == True:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[i1:i2],y[i1:i2])
        (m,b,yfit,eqn) = linearfit(x[i1:i2],y[i1:i2]) # find power law exponent from slope
        plt.plot(x[i1:i2],yfit,label=eqn)
        regression_stats=[slope, intercept, r_value, p_value, std_err]
    plt.grid(True)
    plt.legend()
    plt.show()
    return regression_stats 
##############################################################################    
# In[]: COLLECT DATA
print('COLLECTING DATA...')
t1 = t()
N = int(1e6) # set N=1e6 for decent results.
p4   = OsloModel(L=4,p=0.5)
p8   = OsloModel(L=8,p=0.5)
p16a = OsloModel(L=16,p=0)
p16b = OsloModel(L=16,p=1)
p16  = OsloModel(L=16,p=0.5)
p32  = OsloModel(L=32,p=0.5)
p64  = OsloModel(L=64,p=0.5)
p128 = OsloModel(L=128,p=0.5)
p256 = OsloModel(L=256,p=0.5)
p512 = OsloModel(L=512,p=0.5)

p4.runPile(N,animate=False)
p8.runPile(N,animate=False)
p16a.runPile(N,animate=False)
p16b.runPile(N,animate=False)
p16.runPile(N,animate=False)
p32.runPile(N,animate=False)
p64.runPile(N,animate=False)
p128.runPile(N,animate=False)
p256.runPile(N,animate=False)
p512.runPile(N,animate=False)
#
dt = t()-t1
print('    Time Taken= %.5f'%dt)
print('Completed collecting data..')
# In[]:
"""
Load Data
"""
print('LOADING DATA...')

t1 = t()
load = False
if load:
    p4file = loadData('pile4tilde')    # used 60 piles
    p8file = loadData('pile8tilde')    # used 30 piles
    p16file = loadData('pile16tilde')  # used 20 piles
    p32file = loadData('pile32tilde')  # used 10 piles
    p64file = loadData('pile64tilde')  # used 10 piles
    p128file = loadData('pile128tilde')# used 10 piles
    p256file = loadData('pile256tilde')# used 10 piles
    p512file = loadData('pile512tilde')# used 5 piles
    
    p4.reload(p4file)
    p8.reload(p8file)
    p16.reload(p16file)
    p32.reload(p32file)
    p64.reload(p64file)
    p128.reload(p128file)
    p256.reload(p256file)
    p512.reload(p512file)
dt = t()-t1
print('    Time Taken= %.5f'%dt)
print('Completed Loading Data...')
###############################################################################
# In[]:
"""
Task1: Test results
"""
print('\nPerforming Task 1...')
t1 = t()
## calculate averages
p4.calc_avg() 
p8.calc_avg()
p16a.calc_avg()
p16b.calc_avg()
p16.calc_avg()
p32.calc_avg()
p64.calc_avg()
p128.calc_avg()
p256.calc_avg()
p512.calc_avg()

################ print results################
p4.printResults()
p8.printResults()
p16a.printResults()
p16b.printResults()
p16.printResults()
p32.printResults()
dt = t()-t1
print('    Time Taken= %.5f'%dt)
print('Completed Task 1...')
# In[]:
"""
Task 2a: plot height vs time
"""
print('\nPerforming Task 2a...')
t1 = t()
p4.ploth1()
p8.ploth1()
p16.ploth1()
p32.ploth1()
p64.ploth1()
p128.ploth1()
p256.ploth1()
p512.ploth1()
plt.plot(np.arange(1,p512.ngrains+1), np.arange(1,p512.ngrains+1)**0.5,'k--',label='y=0.5x')
plt.legend()
dt = t()-t1
plt.savefig('Task2a-h1.png', bbox_inches='tight',dpi=200)
print('    Time Taken= %.5f'%dt)
print('Completed Task 2a...')
# In[]:
"""
Task 2c: Smoothing h(t;L) to htilde(t;L) by averaging over using M different piles of the same size L.
Then produce data collapse. 
"""
print('\nPerforming Task 2c DATA SMOOTHING & DATA COLLAPSE...')
t1 = t()
M = 10
p4.calc_htilde(M)
p8.calc_htilde(M)
p16.calc_htilde(M)
p32.calc_htilde(M)
p64.calc_htilde(M)
p128.calc_htilde(M)
p256.calc_htilde(M)
p512.calc_htilde(M)
####
p4.ploth1(htilde=True)
p8.ploth1(htilde=True)
p16.ploth1(htilde=True)
p32.ploth1(htilde=True)
p64.ploth1(htilde=True)
p128.ploth1(htilde=True)
p256.ploth1(htilde=True)
p512.ploth1(htilde=True)
plt.legend()
#plt.savefig('Task2c-h1tilde.png', bbox_inches='tight',dpi=200)
####
p4.plotDataCollapse()
p8.plotDataCollapse()
p16.plotDataCollapse()
p32.plotDataCollapse()
p64.plotDataCollapse()
p128.plotDataCollapse()
p256.plotDataCollapse()
p512.plotDataCollapse()
plt.plot(np.arange(1,p512.ngrains+1)/(p512.L**2), np.e**0.6181*(np.arange(1,p512.ngrains+1)/(p512.L**2))**0.5063,'k--',label='y=0.5063x+0.6181')
plt.legend()

#plotloglog(np.arange(1,p512.ngrains+1)/(p512.L**2), p512.htilde/512,'','', '',True,True,True,i1=0,i2=p512.tc)

plt.savefig('Task2c-h1tildeCollapse.png', bbox_inches='tight',dpi=200)

####
dt = t()-t1
print('    Time Taken= %.5f'%dt)
print('Completed Task 2c...')
# In[]:
"""
Task 2d: Estimate scaling of average cross-over time with system size (<tc(L)> vs L(L+1)/2)
"""
print('\nPerforming Task 2d...')
t1 = t()
####

p4.calc_tcavg()
p8.calc_tcavg()
p16.calc_tcavg()
p32.calc_tcavg()
p64.calc_tcavg()
p128.calc_tcavg()
p256.calc_tcavg()
p512.calc_tcavg()

zavg = 1.72
Lfnlist = [p4.Lfn, p8.Lfn, p16.Lfn, p32.Lfn, p64.Lfn, p128.Lfn, p256.Lfn, p512.Lfn]
Llist     = [p4.L, p8.L, p16.L, p32.L, p64.L, p128.L, p256.L, p512.L]
tcavglist = [p4.tcavg, p8.tcavg, p16.tcavg, p32.tcavg, p64.tcavg, p128.tcavg, p256.tcavg, p512.tcavg]
tcavgtheory= [p4.zavg*p4.Lfn, p8.zavg*p8.Lfn,p16.zavg*p16.Lfn,p32.zavg*p32.Lfn,
              p64.zavg*p64.Lfn,p128.zavg*p128.Lfn,p256.zavg*p256.Lfn,p512.zavg*p512.Lfn]
tcavgtheory1= [zavg*p4.Lfn, zavg*p8.Lfn,zavg*p16.Lfn,zavg*p32.Lfn,
              zavg*p64.Lfn,zavg*p128.Lfn,zavg*p256.Lfn,zavg*p512.Lfn]
tcavgstats=plotloglog(Lfnlist,tcavglist,r'System size function $L(L+1)/2$',r'Average cross-over time $<t_c(L)>$',
           log=False,newplot=True,fit=True,marker='x',i1=0)
print(tcavgstats)
#plt.savefig('Task2d-tcavg_vs_Lfn.png', bbox_inches='tight',dpi=200)
####
plotloglog(Llist, tcavglist, r'System size function $L(L+1)/2$',r'Average cross-over time $<t_c(L)>$','Experimental',False,True,False, marker='x')
#plotloglog(Llist, tcavgtheory,r'System size $L$',r'Average cross-over time $<t_c(L)>$','Theoretical',False,False,False,marker='-')
plotloglog(Llist, tcavgtheory1,r'System size $L$',r'Average cross-over time $<t_c(L)>$','Theoretical',False,False,False,marker='-')
ax2 = plt.gca().twinx()
ax2.plot(Llist, abs(np.array(tcavgtheory1)-np.array(tcavglist))/np.array(tcavglist)*100, color='C2',label='Error')
ax2.set_ylabel('Relative error %', color='C2')
ax2.set_ylim([-1,14])
ax2.tick_params('y', colors='C2')
####
#plt.savefig('Task2d-tcavg_vs_L.png', bbox_inches='tight',dpi=200)
#zavgdata1 = (p32.zavg + p64.zavg +  p128.zavg +  p256.zavg + p512.zavg)/5
# the slope value of the tcavg vs Lfn is indeed very similar to <z>. The slope value has 0.35% error from actual value of <z>. 
# Therefore, the data corroborate with theoretical prediction. N.B. line of best fit to systems with L>32 to avoid corrections to scaling.
####
dt = t()-t1
print('    Time Taken= %.5f'%dt)
print('Completed Task 2d...')
# In[]:
"""
Task 2e: Average height scaling with system size L and corrections to scaling
"""
# Calculate average height using h(t;L) not htilde(t;L) for t>=tc+1
# plot (1-<h>/(a0*L)) vs L in log-log to reveal corrections to scaling 
# Line equation: log(1-<h>/a0*L) = -w1*log(L) + log(a1)
# To show corrections to scaling: plot <h>/L vs L in loglog
# To find a0: plot (a0-<h>/L) vs. L in loglog and adjust a0. a0 -> <h>/L as L-> infty
# To find w1: slope of above plot is -w1. 
print('\nPerforming Task 2e...')
t1 = t()
a0 = 1.738
def h1avgfn1(h1avg,L,a0):
    return 1- h1avg/a0/L

def h1avgfn2(h1avg,L,a0):
    return a0-h1avg/L
    
Llist       = [p4.L, p8.L, p16.L, p32.L, p64.L, p128.L, p256.L, p512.L]
h1avglist   = [p4.h1avg, p8.h1avg, p16.h1avg, p32.h1avg, p64.h1avg, p128.h1avg, p256.h1avg, p512.h1avg]
h1avgfn1list= [h1avgfn1(p4.h1avg,p4.L,a0), h1avgfn1(p8.h1avg,p8.L,a0),
               h1avgfn1(p16.h1avg,p16.L,a0), h1avgfn1(p32.h1avg,p32.L,a0),
               h1avgfn1(p64.h1avg,p64.L,a0),h1avgfn1(p128.h1avg,p128.L,a0),
               h1avgfn1(p256.h1avg,p256.L,a0),h1avgfn1(p512.h1avg,p512.L,a0)]
h1avgfn2list= [h1avgfn2(p4.h1avg,p4.L,a0), h1avgfn2(p8.h1avg,p8.L,a0),
               h1avgfn2(p16.h1avg,p16.L,a0),h1avgfn2(p32.h1avg,p32.L,a0),
               h1avgfn2(p64.h1avg,p64.L,a0),h1avgfn2(p128.h1avg,p128.L,a0),
               h1avgfn2(p256.h1avg,p256.L,a0),h1avgfn2(p512.h1avg,p512.L,a0)]
h1avgLlist  = [p4.h1avg/p4.L, p8.h1avg/p8.L, p16.h1avg/p16.L, p32.h1avg/p32.L,
               p64.h1avg/p64.L,p128.h1avg/p128.L,p256.h1avg/p256.L,p512.h1avg/p512.L]

plotloglog(Llist,h1avglist,'System size L', 'Average height <h>',marker='x')
#plt.savefig('Task2e-havg_vs_L.png', bbox_inches='tight',dpi=200)

plotloglog(Llist,h1avgLlist,'System size L', 'Average height/System size <h>/L',log=False,fit=False)
plt.hlines(a0,Llist[0],Llist[-1])
#plt.ylim([1,2])
#plt.savefig('Task2e-havgoverL_vs_L.png', bbox_inches='tight',dpi=200)
plotloglog(np.array(Llist),np.array(h1avgfn2list),'System size L', 'Average height function a0-<h>/L',marker='x')
#plt.savefig('Task2e-havgfn_vs_L.png', bbox_inches='tight',dpi=200)
dt = t()-t1
# By tunning a0 such that the third plot shows a straight line (use pearson's r), we can extract -w1 from the slope. 
print('    Time Taken= %.5f'%dt)
print('Completed Task 2e...')
# In[]:
"""
Task 2f: standard deviation of height scaling with system size
"""
print('\nPerforming Task 2f...')
t1 = t()

# In the limit L->infty, average slope <z> = a0 = <h>/L. i.e. a delta function with 0 standard deviation.
p4.calc_h1stddev()
p8.calc_h1stddev()
p16.calc_h1stddev()
p32.calc_h1stddev()
p64.calc_h1stddev()
p128.calc_h1stddev()
p256.calc_h1stddev()
p512.calc_h1stddev()

p4.calc_zstddev()
p8.calc_zstddev()
p16.calc_zstddev()
p32.calc_zstddev()
p64.calc_zstddev()
p128.calc_zstddev()
p256.calc_zstddev()
p512.calc_zstddev()

Llist        = [p4.L, p8.L, p16.L, p32.L, p64.L, p128.L, p256.L, p512.L]
h1stddevlist = [p4.h1stddev, p8.h1stddev, p16.h1stddev, p32.h1stddev, p64.h1stddev, p128.h1stddev, p256.h1stddev, p512.h1stddev]
zstddevlist  = [p4.zstddev, p8.zstddev, p16.zstddev, p32.zstddev,p64.zstddev,p128.zstddev,p256.zstddev,p512.zstddev]
print(plotloglog(Llist,h1stddevlist,'System size L', r'Standard deviation of height $\sigma_h$',marker='x',i1=4))
print(plotloglog(Llist,zstddevlist,'System size L', r'Standard deviation of height $\sigma_z$',marker='x',i1=4))
#plt.savefig('Task2f-h1stddev_vs_L.png', bbox_inches='tight',dpi=200)
dt = t()-t1
print('    Time Taken= %.5f'%dt)
print('Completed Task 2f...')

# In[]:
#Save data using pickle
print('\nSaving Data...')

t1 = t()
save = False
if save:
    p4data = [p4.h, p4.h1avg, p4.h1list, p4.h1sqavg, p4.h1stddev, p4.htilde, p4.L, 
              p4.Lfn, p4.ngrains, p4.out, p4.p, p4.s, p4.savg, p4.slist, p4.tc, p4.tcavg, 
              p4.tclist,p4.z,p4.zavg,p4.zavglist,p4.zth]
    
    p8data = [p8.h, p8.h1avg, p8.h1list, p8.h1sqavg, p8.h1stddev, p8.htilde, p8.L, 
              p8.Lfn, p8.ngrains, p8.out, p8.p, p8.s, p8.savg, p8.slist, p8.tc, p8.tcavg, 
              p8.tclist,p8.z,p8.zavg,p8.zavglist,p8.zth]
    
    p16data = [p16.h, p16.h1avg, p16.h1list, p16.h1sqavg, p16.h1stddev, p16.htilde, p16.L, 
              p16.Lfn, p16.ngrains, p16.out, p16.p, p16.s, p16.savg, p16.slist, p16.tc, p16.tcavg, 
              p16.tclist,p16.z,p16.zavg,p16.zavglist,p16.zth]
    
    p32data = [p32.h, p32.h1avg, p32.h1list, p32.h1sqavg, p32.h1stddev, p32.htilde, p32.L, 
              p32.Lfn, p32.ngrains, p32.out, p32.p, p32.s, p32.savg, p32.slist, p32.tc, p32.tcavg, 
              p32.tclist,p32.z,p32.zavg,p32.zavglist,p32.zth]
    
    p64data = [p64.h, p64.h1avg, p64.h1list, p64.h1sqavg, p64.h1stddev, p64.htilde, p64.L, 
              p64.Lfn, p64.ngrains, p64.out, p64.p, p64.s, p64.savg, p64.slist, p64.tc, p64.tcavg, 
              p64.tclist,p64.z,p64.zavg,p64.zavglist,p64.zth]
    
    p128data = [p128.h, p128.h1avg, p128.h1list, p128.h1sqavg, p128.h1stddev, p128.htilde, p128.L, 
              p128.Lfn, p128.ngrains, p128.out, p128.p, p128.s, p128.savg, p128.slist, p128.tc, p128.tcavg, 
              p128.tclist,p128.z,p128.zavg,p128.zavglist,p128.zth]
    
    p256data = [p256.h, p256.h1avg, p256.h1list, p256.h1sqavg, p256.h1stddev, p256.htilde, p256.L, 
              p256.Lfn, p256.ngrains, p256.out, p256.p, p256.s, p256.savg, p256.slist, p256.tc, p256.tcavg, 
              p256.tclist,p256.z,p256.zavg,p256.zavglist,p256.zth]
    
    p512data = [p512.h, p512.h1avg, p512.h1list, p512.h1sqavg, p512.h1stddev, p512.htilde, p512.L, 
              p512.Lfn, p512.ngrains, p512.out, p512.p, p512.s, p512.savg, p512.slist, p512.tc, p512.tcavg, 
              p512.tclist,p512.z,p512.zavg,p512.zavglist,p512.zth]
    
    saveData('pile4'  ,p4data)
    saveData('pile8'  ,p8data)
    saveData('pile16' ,p16data)
    saveData('pile32' ,p32data)
    saveData('pile64' ,p64data)
    saveData('pile128',p128data)
    saveData('pile256',p256data)
    saveData('pile512',p512data)
dt = t()-t1
print('    Time Taken= %.5f'%dt)
print('Data Saved.')
# In[]:
"""
#Task 2g: Probability of height given system size L (P(h;L)) scaling with h for different L.
"""
print('\nPerforming Task 2g...')
t1 = t()
#
probhlist   ={'L=4':[],'L=8':[],'L=16':[],'L=32':[],'L=64':[],'L=128':[],'L=256':[],'L=512':[]}
probhscaled ={'L=4':[],'L=8':[],'L=16':[],'L=32':[],'L=64':[],'L=128':[],'L=256':[],'L=512':[]}
#
probhlist['L=4']  =p4.calc_h1prob()
probhlist['L=8']  =p8.calc_h1prob()
probhlist['L=16'] =p16.calc_h1prob()
probhlist['L=32'] =p32.calc_h1prob()
probhlist['L=64'] =p64.calc_h1prob()
probhlist['L=128']=p128.calc_h1prob()
probhlist['L=256']=p256.calc_h1prob()
probhlist['L=512']=p512.calc_h1prob()

# check probabilities are correctly normalised
print('sum of P(h;L) = %.1f'%np.sum(probhlist['L=4'][1]))

plotloglog(probhlist['L=4'][0],   probhlist['L=4'][1],  'Height h','Probability of height P(h;L)','L=4',  False,True, False,marker='-')
plotloglog(probhlist['L=8'][0],   probhlist['L=8'][1],  'Height h','Probability of height P(h;L)','L=8',  False,False,False,marker='-')
plotloglog(probhlist['L=16'][0],  probhlist['L=16'][1], 'Height h','Probability of height P(h;L)','L=16', False,False,False,marker='-')
plotloglog(probhlist['L=32'][0],  probhlist['L=32'][1], 'Height h','Probability of height P(h;L)','L=32', False,False,False,marker='-')
plotloglog(probhlist['L=64'][0],  probhlist['L=64'][1], 'Height h','Probability of height P(h;L)','L=64', False,False,False,marker='-')
plotloglog(probhlist['L=128'][0], probhlist['L=128'][1],'Height h','Probability of height P(h;L)','L=128',False,False,False,marker='-')
plotloglog(probhlist['L=256'][0], probhlist['L=256'][1],'Height h','Probability of height P(h;L)','L=256',False,False,False,marker='-')
plotloglog(probhlist['L=512'][0], probhlist['L=512'][1],'Height h','Probability of height P(h;L)','L=512',False,False,False,marker='-')
#plt.savefig('Task2g-P(h,L)_vs_h.png', bbox_inches='tight',dpi=200)

# Data collpase for P(h)
probhscaled['L=4']   = p4.h1stddev*probhlist['L=4'][1]
probhscaled['L=8']   = p8.h1stddev*probhlist['L=8'][1]
probhscaled['L=16']  = p16.h1stddev*probhlist['L=16'][1]
probhscaled['L=32']  = p32.h1stddev*probhlist['L=32'][1]
probhscaled['L=64']  = p64.h1stddev*probhlist['L=64'][1]
probhscaled['L=128'] = p128.h1stddev*probhlist['L=128'][1]
probhscaled['L=256'] = p256.h1stddev*probhlist['L=256'][1]
probhscaled['L=512'] = p512.h1stddev*probhlist['L=512'][1]

plotloglog((probhlist['L=4'][0]-p4.h1avg)/p4.h1stddev,      probhscaled['L=4'],  r'Normalised Height $(h-<h>)/\sigma_h$',r'Normalised Probability of height $\sigma_hP(h;L)$','L=4',False,True,False,marker='x')
plotloglog((probhlist['L=8'][0]-p8.h1avg)/p8.h1stddev,      probhscaled['L=8'],  r'Normalised Height $(h-<h>)/\sigma_h$',r'Normalised Probability of height $\sigma_hP(h;L)$','L=8',False,False,False,marker='x')
plotloglog((probhlist['L=16'][0]-p16.h1avg)/p16.h1stddev,   probhscaled['L=16'], r'Normalised Height $(h-<h>)/\sigma_h$',r'Normalised Probability of height $\sigma_hP(h;L)$','L=16',False,False,False,marker='x')
plotloglog((probhlist['L=32'][0]-p32.h1avg)/p32.h1stddev,   probhscaled['L=32'], r'Normalised Height $(h-<h>)/\sigma_h$',r'Normalised Probability of height $\sigma_hP(h;L)$','L=32',False,False,False,marker='x')
plotloglog((probhlist['L=64'][0]-p64.h1avg)/p64.h1stddev,   probhscaled['L=64'], r'Normalised Height $(h-<h>)/\sigma_h$',r'Normalised Probability of height $\sigma_hP(h;L)$','L=64',False,False,False,marker='x')
plotloglog((probhlist['L=128'][0]-p128.h1avg)/p128.h1stddev,probhscaled['L=128'],r'Normalised Height $(h-<h>)/\sigma_h$',r'Normalised Probability of height $\sigma_hP(h;L)$','L=128',False,False,False,marker='x')
plotloglog((probhlist['L=256'][0]-p256.h1avg)/p256.h1stddev,probhscaled['L=256'],r'Normalised Height $(h-<h>)/\sigma_h$',r'Normalised Probability of height $\sigma_hP(h;L)$','L=256',False,False,False,marker='x')
plotloglog((probhlist['L=512'][0]-p512.h1avg)/p512.h1stddev,probhscaled['L=512'],r'Normalised Height $(h-<h>)/\sigma_h$',r'Normalised Probability of height $\sigma_hP(h;L)$','L=512',False,False,False,marker='x')

plt.plot(np.linspace(-6,6,100),Gaussian(np.linspace(-6,6,100),0,1),'k-',label=r'Gaussian($\mu=0,\sigma=1$)')
plt.xlim([-6,6])
plt.legend(loc='best')
#plt.savefig('Task2g-Pscaled(h,L)_vs_hscaled.png', bbox_inches='tight',dpi=200)

dt = t()-t1
print('    Time Taken= %.5f'%dt)
print('Completed Task 2g...')
#
# In[]:
"""
Task 3a: Avalanche-size probability P(s;L) vs Avalanche size s for all system sizes L.
"""
print('\nPerforming Task 3a...')
t1 = t()
# raw data
probslist   ={'L=4':[],'L=8':[],'L=16':[],'L=32':[],'L=64':[],'L=128':[],'L=256':[],'L=512':[]}
probsbinned ={'L=4':[],'L=8':[],'L=16':[],'L=32':[],'L=64':[],'L=128':[],'L=256':[],'L=512':[]}

probslist['L=4']=p4.calc_sprob()
probslist['L=8']=p8.calc_sprob()
probslist['L=16']=p16.calc_sprob()
probslist['L=32']=p32.calc_sprob()
probslist['L=64']=p64.calc_sprob()
probslist['L=128']=p128.calc_sprob()
probslist['L=256']=p256.calc_sprob()
probslist['L=512']=p512.calc_sprob()
# check probabilities are correctly normalised
print('sum of P(s;L) = %.1f'%np.sum(probslist['L=4'][1]))

plotloglog(probslist['L=4'][0],   probslist['L=4'][1],  'Avalanche size s','Probability of avalanche-size P(s;L)','L=4',  True,True, False,marker='x',Alpha=0.25)
plotloglog(probslist['L=8'][0],   probslist['L=8'][1],  'Avalanche size s','Probability of avalanche-size P(s;L)','L=8',  True,False,False,marker='x',Alpha=0.25)
plotloglog(probslist['L=16'][0],  probslist['L=16'][1], 'Avalanche size s','Probability of avalanche-size P(s;L)','L=16', True,False,False,marker='x',Alpha=0.25)
plotloglog(probslist['L=32'][0],  probslist['L=32'][1], 'Avalanche size s','Probability of avalanche-size P(s;L)','L=32', True,False,False,marker='x',Alpha=0.25)
plotloglog(probslist['L=64'][0],  probslist['L=64'][1], 'Avalanche size s','Probability of avalanche-size P(s;L)','L=64', True,False,False,marker='x',Alpha=0.25)
plotloglog(probslist['L=128'][0], probslist['L=128'][1],'Avalanche size s','Probability of avalanche-size P(s;L)','L=128',True,False,False,marker='x',Alpha=0.25)
plotloglog(probslist['L=256'][0], probslist['L=256'][1],'Avalanche size s','Probability of avalanche-size P(s;L)','L=256',True,False,False,marker='x',Alpha=0.25)
plotloglog(probslist['L=512'][0], probslist['L=512'][1],'Avalanche size s','Probability of avalanche-size P(s;L)','L=512',True,False,False,marker='x',Alpha=0.25)

# binned data
binscale = 1.26
probsbinned['L=4'] = logbin(p4.slist[p4.tc+1:],binscale)
probsbinned['L=8'] = logbin(p8.slist[p8.tc+1:],binscale)
probsbinned['L=16'] = logbin(p16.slist[p16.tc+1:],binscale)
probsbinned['L=32'] = logbin(p32.slist[p32.tc+1:],binscale)
probsbinned['L=64'] = logbin(p64.slist[p64.tc+1:],binscale)
probsbinned['L=128'] = logbin(p128.slist[p128.tc+1:],binscale)
probsbinned['L=256'] = logbin(p256.slist[p256.tc+1:],binscale)
probsbinned['L=512'] = logbin(p512.slist[p512.tc+1:],binscale)

plotloglog(probsbinned['L=4'][0],  probsbinned['L=4'][1],  'Avalanche size s','Probability of avalanche size P(s;L)','L=4 binned',True,False,False,marker='-',color='C0')
plotloglog(probsbinned['L=8'][0],  probsbinned['L=8'][1],  'Avalanche size s','Probability of avalanche size P(s;L)','L=8 binned',True,False,False,marker='-',color='C1')
plotloglog(probsbinned['L=16'][0], probsbinned['L=16'][1], 'Avalanche size s','Probability of avalanche size P(s;L)','L=16 binned',True,False,False,marker='-',color='C2')
plotloglog(probsbinned['L=32'][0], probsbinned['L=32'][1], 'Avalanche size s','Probability of avalanche size P(s;L)','L=32 binned',True,False,False,marker='-',color='C3')
plotloglog(probsbinned['L=64'][0], probsbinned['L=64'][1], 'Avalanche size s','Probability of avalanche size P(s;L)','L=64 binned',True,False,False,marker='-',color='C4')
plotloglog(probsbinned['L=128'][0],probsbinned['L=128'][1],'Avalanche size s','Probability of avalanche size P(s;L)','L=128 binned',True,False,False,marker='-',color='C5')
plotloglog(probsbinned['L=256'][0],probsbinned['L=256'][1],'Avalanche size s','Probability of avalanche size P(s;L)','L=256 binned',True,False,False,marker='-',color='C6')
plotloglog(probsbinned['L=512'][0],probsbinned['L=512'][1],'Avalanche size s','Probability of avalanche size P(s;L)','L=512 binned',True,False,False,marker='-',color='C7')
#plt.savefig('Task3a-P(s,L)_vs_s.png', bbox_inches='tight',dpi=200)
# Estimate the slope of the loglog plot to find exponent of power -law scaling: get tau_s = 1.55
plotloglog(probsbinned['L=512'][0],probsbinned['L=512'][1],'Avalanche size s',
           'Probability of avalanche size P(s;L)','L=512 binned',True,True,True,
           marker='-',color='C7',i1=7,i2=-12)
#plt.savefig('Task3a-P(s,512)_vs_s.png', bbox_inches='tight',dpi=200)
dt = t()-t1
print('    Time Taken= %.5f'%dt)
print('Completed Task 3a...')
# In[]:
"""
Task 3b: Data collapse: Scaled Avalanche-size probability s^tau_s*P(s;L) vs Scaled avalanche size s/L^D for all system sizes L.
"""
print('\nPerforming Task 3b...')
t1 = t()
# initial tau_s value from task 3a- fine-tune by aligning horizontally the bump (i.e. peak of bump at same height)
# Tune D by aligning vertically the bump (i.e. at the same x-value)
tau_s = 1.557
D = 2.25

plotloglog(probsbinned['L=4'][0]/p4.L**D,  probsbinned['L=4'][0]**tau_s*probsbinned['L=4'][1],      r'Normalised Avalanche size $s/L^D$',r'Normalised probability of avalanche size $s^{\tau_s}\tilde{P}(s;L)$','L=4 binned',True,False,False,marker='-',color='C0')
plotloglog(probsbinned['L=8'][0]/p8.L**D,  probsbinned['L=8'][0]**tau_s*probsbinned['L=8'][1],      r'Normalised Avalanche size $s/L^D$',r'Normalised probability of avalanche size $s^{\tau_s}\tilde{P}(s;L)$','L=8 binned',True,False,False,marker='-',color='C1')
plotloglog(probsbinned['L=16'][0]/p16.L**D, probsbinned['L=16'][0]**tau_s*probsbinned['L=16'][1],   r'Normalised Avalanche size $s/L^D$',r'Normalised probability of avalanche size $s^{\tau_s}\tilde{P}(s;L)$','L=16 binned',True,False,False,marker='-',color='C2')
plotloglog(probsbinned['L=32'][0]/p32.L**D, probsbinned['L=32'][0]**tau_s*probsbinned['L=32'][1],   r'Normalised Avalanche size $s/L^D$',r'Normalised probability of avalanche size $s^{\tau_s}\tilde{P}(s;L)$','L=32 binned',True,False,False,marker='-',color='C3')
plotloglog(probsbinned['L=64'][0]/p64.L**D, probsbinned['L=64'][0]**tau_s*probsbinned['L=64'][1],   r'Normalised Avalanche size $s/L^D$',r'Normalised probability of avalanche size $s^{\tau_s}\tilde{P}(s;L)$','L=64 binned',True,False,False,marker='-',color='C4')
plotloglog(probsbinned['L=128'][0]/p128.L**D,probsbinned['L=128'][0]**tau_s*probsbinned['L=128'][1],r'Normalised Avalanche size $s/L^D$',r'Normalised probability of avalanche size $s^{\tau_s}\tilde{P}(s;L)$','L=128 binned',True,False,False,marker='-',color='C5')
plotloglog(probsbinned['L=256'][0]/p256.L**D,probsbinned['L=256'][0]**tau_s*probsbinned['L=256'][1],r'Normalised Avalanche size $s/L^D$',r'Normalised probability of avalanche size $s^{\tau_s}\tilde{P}(s;L)$','L=256 binned',True,False,False,marker='-',color='C6')
plotloglog(probsbinned['L=512'][0]/p512.L**D,probsbinned['L=512'][0]**tau_s*probsbinned['L=512'][1],r'Normalised Avalanche size $s/L^D$',r'Normalised probability of avalanche size $s^{\tau_s}\tilde{P}(s;L)$','L=512 binned',True,False,False,marker='-',color='C7')
#plt.savefig('Task3b-Pscaled(s,L)_vs_sscaled.png', bbox_inches='tight',dpi=200)
dt = t()-t1
print('    Time Taken= %.5f'%dt)
print('Completed Task 3b...')

# In[]:
"""
Task 3c: Measure the kth moment of avalanche size <s^k> for k = 1, 2, 3, 4
"""
print('\nPerforming Task 3c...')
t1 = t()
Llist       = [p4.L, p8.L, p16.L, p32.L, p64.L, p128.L, p256.L,p512.L]
sklist = {'k=1':[],'k=2':[],'k=3':[],'k=4':[]}
tau_s = 1.55
D = 2.22

k = np.array([[1],[2],[3],[4]])
sk4, sk8, sk16, sk32, sk64 = p4.calc_kmoment(k),p8.calc_kmoment(k),p16.calc_kmoment(k),p32.calc_kmoment(k),p64.calc_kmoment(k)
sk128, sk256, sk512 = p128.calc_kmoment(k),p256.calc_kmoment(k),p512.calc_kmoment(k)
sklist['k=1']= [sk4[0],sk8[0],sk16[0],sk32[0],sk64[0],sk128[0],sk256[0],sk512[0]]
sklist['k=2']= [sk4[1],sk8[1],sk16[1],sk32[1],sk64[1],sk128[1],sk256[1],sk512[1]]
sklist['k=3']= [sk4[2],sk8[2],sk16[2],sk32[2],sk64[2],sk128[2],sk256[2],sk512[2]]
sklist['k=4']= [sk4[3],sk8[3],sk16[3],sk32[3],sk64[3],sk128[3],sk256[3],sk512[3]]
   
plt1 = plotloglog(Llist,sklist['k=1'],'System size L', r'Average moment $<s^k>$','k=1',True,True,marker='x',color='C0',i1=4)
plt2 = plotloglog(Llist,sklist['k=2'],'System size L', r'Average moment $<s^k>$','k=2',True,False,marker='x',color='C1',i1=4)
plt3 = plotloglog(Llist,sklist['k=3'],'System size L', r'Average moment $<s^k>$','k=3',True,False,marker='x',color='C2',i1=4)
plt4 = plotloglog(Llist,sklist['k=4'],'System size L', r'Average moment $<s^k>$','k=4',True,False,marker='x',color='C3',i1=4)
#plt.savefig('Task3c-sk_vs_k.png', bbox_inches='tight',dpi=200)

skgradients = [plt1[0],plt2[0],plt3[0],plt4[0]]
plt5=plotloglog([1,2,3,4],skgradients,'Moment k', r'Gradient of $log<s^k>$ vs. log(L)',log=False,newplot=True,fit=True,marker='x')
#plt.savefig('Task3c-momentGradient_vs_k.png', bbox_inches='tight',dpi=200)

# Corrections to scaling for moments analysis
skcorrections = {'k=1':[],'k=2':[],'k=3':[],'k=4':[]}
skcorrections['k=1'] = sklist['k=1']*Llist**(-D*(1+k[0]-tau_s))
skcorrections['k=2'] = sklist['k=2']*Llist**(-D*(1+k[1]-tau_s))
skcorrections['k=3'] = sklist['k=3']*Llist**(-D*(1+k[2]-tau_s))
skcorrections['k=4'] = sklist['k=4']*Llist**(-D*(1+k[3]-tau_s))

plotloglog(Llist,skcorrections['k=1'],'System size L', 'Corrections to scaling for moments','k=1',False,True,False)
plotloglog(Llist,skcorrections['k=2'],'System size L', 'Corrections to scaling for moments','k=2',False,False,False)
plotloglog(Llist,skcorrections['k=3'],'System size L', 'Corrections to scaling for moments','k=3',False,False,False)
plotloglog(Llist,skcorrections['k=4'],'System size L', 'Corrections to scaling for moments','k=4',False,False,False)
#plt.savefig('Task3c-momentCorrections_vs_L.png', bbox_inches='tight',dpi=200)

print('    Time Taken= %.5f'%dt)
print('Completed Task 3c...')


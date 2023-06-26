# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:27:56 2019

@author: I.K.Cheng 00941460

Title: Networks Project

BA Model Algorithm:
1. Set up initial network at time t0, a graph G0.
2. Increment time t -> t + 1
3. Add one new node
4. Add m edges to new node.
5. Attach the other end of the m edges as follows:
       Connect one end of the new edge to the new node
       Connect the other end of each new edge to an existing node chosen with 
       probability p. This will be specied in different ways in the different tasks.
6. Repeat from 2 until reach final number of N vertices in the network.
"""
"""
To do: Add option to only append to adj_ls if in test mode.
"""
#Import dependencies
import numpy as np # for math
from scipy import stats # for fitting
import matplotlib.pyplot as plt # for plots
from matplotlib import rcParams # change figure defaults
import networkx as nx # for visualising a graph
from scipy.stats import ks_2samp # for KS test
import random # for generating random numbers
import collections as cl # for counting degrees of vertices
import time # for timing performance
import datetime # for getting date and time
import pickle # for saving variables
from logbin622018 import logbin # for log-binning data
rcParams.update({'font.size': 12,'figure.autolayout':True}) # google matplotlib-rcparams for other defualt settings
main_seed=100
random.seed(main_seed)
#Start script timer
start_time = time.perf_counter()
print('Start time: ',datetime.datetime.now())
"""
Don't use self.G functions because they are very slow!! just add nodes to node_ls 
and edges to edge_ls. 
"""
####################################################################################
#Set-up parameters and problem
#Define parameters
save = False # True for saving data.

##############################################################################
# Task 1: Phase 1
class BA_model():
    """8
    A BA_ model with three different ways of edge attachment for the new vertex added each time step: 
        1) Peferential Attachment- attach new egde to an existing node 
           with probability proportional to its degree.
        2) Random Attachment-  Attach each of m edges to existing vertices in the network with 
           probability p_random propto 1 (i.e. unform pdf)
        3) Existing vertices model- Attach r edges from new vertex to existing vertex chosen with probability p_preferential.
                                    Attach m-r edges between two EXISTING vertices, each chosen with probability p_random.
    """
    def __init__(self, m=1, N0=1, attachment='p', r=1, Graph0='', test=False): 
        """ 
        Step1: Set up initial network at time t=0, a graph G(t=0).
        Input: m = number of new edges added per timestep 
               N0 = number of nodes in initial graph
               attachment = method to attach a new edge to existing verticies
                            1) p = Preferential (to degree)
                            2) r = Random
                            3) pr = Preferential and Random 
               r = number of edges attached between new vertex and existing vertex using preferential attachment.
                   for r=m/2, if m is odd, use int(m/2) which will always round down.    
                   Only used for attachment='pr'
               Graph0 = initial graph at t=0
               test   = True or False. True if want to visualise graph using networkx. 
                        Involves using G.add_node and G.add_edge which are slow methods.
        """
        random.seed(main_seed)
        self.G = nx.Graph() # generate an initial network
        self.t = 0          # time
        self.N = N0         # number of vertices at current time
        self.N0= N0         # number of vertices at t=0
        self.E = 0          # number of edges at current time
        self.E0= 0          # number of edges at t=0
        self.m = m          # number of edges added at each timestep
        self.attachment = attachment # method of attaching new edges to existing nodes.
        self.r = r          # number of edges attached with preferential attachment between new node and existing node.
        self.Graph0=Graph0  # initial graph at t=0
        self.test = test    # True or False (for visualising the graph)
        assert(m<=N0), "number of edges added at each timestep must be smaller than or equal to number of vertices available at current time"
        
        ################ Variables to describe the network ################
        
        self.adj_ls           = [[] for i in range(self.N)] # adjacency list= list of nearest neighbours of node i
                                                            # most effient way to store a network
        self.edge_ls          = []  # source and target nodes of each edge stored as a tuple
        self.node_ls          = []  # list of all the nodes in the network
        self.attached_node_ls = []  # every time new edge attached to a node, add that node to this list. 
                                    # The number of times a particular node appears 
                                    # in this list is proportional to its degree.
        
        ######### Set up a simple complete graph with N0 nodes #########
        if self.Graph0 == 'complete':
            for i in range(N0):
                self.G.add_node(i) # populate network with nodes labelled 'i'
                if attachment=='pr' and i==0: # if using existing vertices model
                    for j in range(i+1,int(N0-m/2)): # want to leave m/2 edges unconnected in the initial graph, so that a complete graph is made when new edges are added between existing vertices for the first timestep t=1.
                        self.G.add_edge(i, j%N0) # populate network with all possible edges
                        self.attached_node_ls.append(i)  # add source and target vertices to list when a new edge is attached
                        self.attached_node_ls.append(j%N0) 
                else:  
                    for j in range(i+1, N0):
                        self.G.add_edge(i, j%N0) # populate network with all possible edges
                        self.attached_node_ls.append(i)  # add source and target vertices to list when a new edge is attached
                        self.attached_node_ls.append(j%N0) 
                self.adj_ls[i]=[n for n in self.G[i]] # update adjacency matrix
            self.node_ls = list(self.G.nodes())
            self.edge_ls = list(self.G.edges())
            self.E0 = len(self.edge_ls)
            self.E  = len(self.edge_ls)


    def nextTimestep(self):
        """
        Step2-4: Increment time t -> t + 1 and add a new node with m edges.
        """
        self.t += 1 # move to next timestep
        self.N += 1 # add 1 new node
        self.newNode=self.N-1 # -1 because node index starts from 0
        self.E += self.m # add m new edges 
        self.adj_ls.append([]) # add new node to adj list. It has m nearest neighbours.

        """
        Step5: Attach the other end of the m edges using one of the following attachment methods.
        """
        self.chosenNode_ls=[] # keep a record of chosen nodes to avoid attaching to the same node twice.
        
        if self.attachment == 'p': # preferential attachment
            self.preferentialAttachment(self.m)
        
        elif self.attachment == 'r': # random attachment
            self.randomAttachment(self.m)
        
        elif self.attachment == 'pr': # existing node model: mixed preferential and random attachment
            self.mixedAttachment(self.m)
        
        self.node_ls.append(self.newNode) # update list of all nodes in network
        if self.test: self.G.add_node(self.newNode) # add node to graph for visualising
    
    def preferentialAttachment(self, m):
        """
        Attach each of m edges of new vertex to existing vertices in the network with 
        probability p_preferential propto degree of the nodes.
        Input: m = number of edges to attach in total
        """
        for i in range(m):
            self.chosenNode     = random.choice(self.attached_node_ls) # choose random node from attached_node_ls list. 
                                                                       # Probability is propto degree since higher degree means more occurence in this list.
            while self.chosenNode in self.chosenNode_ls: # don't want self-loops
                self.chosenNode = random.choice(self.attached_node_ls) # re-choose a node
            self.updateGraph()
        # add new node to attached_node_ls after all m edges are attached to avoid self-loop.
        self.attached_node_ls.extend([self.newNode]*m)  
        
    def randomAttachment(self, m):
        """
        Attach each of m edges to existing vertices in the network with 
        probability p_random propto 1 (i.e. unform pdf)
        Input: m = number of edges to attach in total
        """
        if self.attachment == 'pr':
            # want to attach two existing nodes that are not already connected.
            for i in range(m):
                self.chosenNodes = tuple(random.sample(self.node_ls, 2))                 
                count = 1
                while (self.chosenNodes in self.edge_ls) or (self.chosenNodes[::-1] in self.edge_ls): # prevent choosing same edge
                # e.g. (2,7) is same as (7,2) in undirected graph
                # time this loop: 
                    if count%1e10==0.0:
                        print(count)
                        raise ValueError('No new edge found between existing vertices. Exeeded %d trials.'%(count-1))
                        
                    self.chosenNodes = tuple(random.sample(self.node_ls, 2))
                    count+=1
                    #print('This loop took %.3e secs'%(time.perf_counter()-tstart))
                # update the network variables
                if self.test: self.G.add_edge(self.chosenNodes[0], self.chosenNodes[1]) # add new edge to nx graph for visualising 
                self.edge_ls.append((self.chosenNodes[0], self.chosenNodes[1]))
                self.attached_node_ls.extend(self.chosenNodes)
                self.adj_ls[self.chosenNodes[0]].append(self.chosenNodes[1])
                self.adj_ls[self.chosenNodes[1]].append(self.chosenNodes[0])
        else:
            for i in range(m):
                self.chosenNode = random.choice(self.node_ls)                
                while self.chosenNode in self.chosenNode_ls:
                    self.chosenNode = random.choice(self.node_ls)
                self.updateGraph()
                
            # add new node to attached_node_ls after all m edges are attached to avoid self-loop.
            self.attached_node_ls.extend([self.newNode]*m)  

    def mixedAttachment(self,m):
        """
        Attach m edges in total.
        Attach r edges from new vertex to existing vertex chosen with probability p_preferential.
        Attach m-r edges between two EXISTING vertices, each chosen with probability p_random.
        Input: m = number of edges to be connected in total.
               r = number of edges connecting new and existing vertices.
        """
        for i in range(m):
            if i<self.r: self.randomAttachment(1) # attach new edges between existing nodes first
            else:  self.preferentialAttachment(1) # then attach new edges to new node.
            #print(self.edge_ls[-1])
    
    def updateGraph(self):
        """ update the network variables for the case that an edge is attached between new node and existing node."""
        self.chosenNode_ls.append(self.chosenNode)
        if self.test: self.G.add_edge(self.newNode, self.chosenNode) # add new edge to nxgraph for visualising
        self.edge_ls.append((self.newNode,self.chosenNode))
        self.attached_node_ls.append(self.chosenNode)
        self.adj_ls[self.chosenNode].append(self.newNode)
        self.adj_ls[self.newNode].append(self.chosenNode)
        
    def repeatNextTimestep(self, Nsteps=10, approach='node'):
        """
        Step6: Repeat nextTimestep() method Nsteps times. 
        """
        for i in range (Nsteps):
            self.nextTimestep()
            
    def calcDegree(self):
        """
        Calculates the degree (k) of each node in the network.
        Method: Count the number of times each node appears in attached_node_ls 
        (i.e. the number of edges attached to that node = degree of that node). 
        Input: scale = number >=1 for log binning data. N.B. scale=1 outputs raw data.
        Output: a list of degrees k and corresponding probability P(k)
        """
        # Use collections.Counter to count frequency of each node. 
        # Use most_common() to change to list of tuples. e.g. [(1,2),(2,3),... (Ni,k)] where Ni = ith node, k= degree
        # Only interested in the degree of the ith node.
        node, self.degreeLs = zip(*cl.Counter(self.attached_node_ls).most_common())
        return self.degreeLs
    
    def calcNk(self):
        """
        Calculates the number of nodes with degree k i.e. N(k)
        """
        #self.k,self.Nk = zip(*cl.Counter(self.degreeLs).most_common())
        self.Nk = cl.Counter(self.degreeLs).most_common()
        return self.Nk
    
    def summariseGraph(self):
        """
        Print summary stats of network.
        """
        print('\ntime: t = %.1f'%self.t)
        print('Nodes: ',self.node_ls)
        print('Edges:', self.edge_ls)
        print("Number of nodes=%d"%len(self.node_ls))
        print("Number of edges=%d"%len(self.edge_ls))
        print('\n')

# In[]:
"""
General functions for fitting, plotting, calculating theoretical values, 
and other tools.
"""
def linearfit(x,y):
    (m,b)=np.polyfit(x ,y ,1)
    yfit = np.polyval([m,b],x)
    eqn = 'y = ' + str(round(m,4)) + 'x' ' + ' + str(round(b,4))
    return m, b, yfit, eqn

def plotloglog(x,y,xlabel='',ylabel='',legend='',title='',log=True, newplot=True,fit=True,marker='x-',color='',Alpha=1, i1=0,i2=None):
    """
    Input: i1 = starting index for regression of data array.
    """
    regression_stats=[]
    if newplot==True:
        plt.figure(title)
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

def calcPk(m,k,attachment='p'): 
    """
    Theoretical p(k) for k>=m. Input: m= number of edges/new node, k= degree
    """
    if attachment=='p': return 2*m*(m+1)/(k*(k+1)*(k+2))
    elif attachment=='r': return m**(k-m)/(m+1)**(k-m+1)
    elif attachment=='pr': return (27*m*(3*m+2)*(9*m+2)*(9*m+4)/4)/((k+4*m+4)*(k+4*m+3)*(k+4*m+2)*(k+4*m+1)*(k+4*m)) # from Wolfram alpha (sum from x=m/2 to infty)

def calck1(m,N, attachment='p'):
    """Theoretical maximum degree. Input: m= number of edges/new node, N=total number of nodes """
    if attachment=='p': return (-1. + np.sqrt(1 + 4*N*m*(m+1)))/2
    elif attachment=='r': return np.log(N)/(np.log(m+1)-np.log(m))+m
    elif attachment=='pr': return 0.5*(np.sqrt(5+np.sqrt(6561*m**4*N+8748*m**3*N+3564*m**2*N+432*m*N+16))-8*m-3)

def findIndex(array, value):
    """
    Find the index of a value in an array.
    """
    return np.searchsorted(array,value)

def lineColor(i):
    """ Set color of line. Input: i= integer"""
    return 'C%d'%i

def makeMultipleGraphs(Nparam, Nseeds, m_ls, N_ls, fix='N', attachment='p',Graph0='complete'):
    """Growing graphs Nseeds times per given parameter (e.g. per m value). i.e. total number of graphs = Nparam*Nseeds"""
    G_ls = [[0]*Nseeds for i in range(Nparam)] # a list of all the graphs. e.g. G_ls[i][j] = graph with ith m value and jth seed.    
    indiv_k  = [[0]*Nseeds for i in range(Nparam)] # store list of degrees for each seed e.g. indiv_degreeLs[i][j] is the degree list of ith m value, and jth seed
    total_k  = [[] for i in range(Nparam)] # store total degree distribution for all the seeds per value of m. 
    k1_ls= [[] for i in range(Nparam)] # a list of the largest degrees for each graph.

    for i in range(Nparam):
        if fix == 'N': print('i=%d, m = %d'%(i,m_ls[i]))
        if fix == 'm': print('i=%d, N = %d'%(i,N_ls[i]))
        for j in range(Nseeds):
            print('seed %d'%(j+1))        
            N0 = m_ls[i] + 1 # initial number of nodes N(0)=m+1 (N0 must be bigger than m)
            G_ls[i][j]=(BA_model(m_ls[i],N0,attachment=attachment,r=m_ls[i]/2,Graph0=Graph0)) # create initial graph (same seed each time in __init__ of BA_model)
            random.seed(seeds[j]) #seed the system with the jth seed to grow the network differently 
            G_ls[i][j].repeatNextTimestep(N_ls[i]-N0) # Add N-N0 nodes to the graph and m edges per node
            G_ls[i][j].calcDegree() # Calculate degree list for each graph
            total_k[i] += G_ls[i][j].degreeLs   # store list of degrees for each graph into a single list
            indiv_k[i][j] = G_ls[i][j].degreeLs # store list of degrees for each graph as individual lists
            k1_ls[i].append(max(G_ls[i][j].degreeLs)) # store largest degree in each run
    return G_ls, indiv_k, total_k, k1_ls

def calcPk_avg(Nparam,Nseeds,indiv_k,total_k):
    indiv_k_logbin  = [[0]*Nseeds for i in range(Nparam)]
    indiv_Pk_logbin = [[0]*Nseeds for i in range(Nparam)]
    zipped_kPk      = [[] for i in range(Nparam)]
    k_bins_raw      = [[] for i in range(Nparam)]
    Pk_raw          = [[] for i in range(Nparam)]
    k_bins          = [[] for i in range(Nparam)]
    Pk_mean         = [[] for i in range(Nparam)]
    Pk_err          = [[] for i in range(Nparam)]
    for i in range(Nparam):
        for j in range(Nseeds):
            k,Pk = logbin(indiv_k[i][j], logbin_scale,True) # log-bin individual degree lists from each seeded run
            indiv_k_logbin[i][j]=k
            indiv_Pk_logbin[i][j]=Pk
            zipped_kPk[i] += zip(indiv_k_logbin[i][j], indiv_Pk_logbin[i][j]) # zip indiv_k_logbin and indiv_Pk_logbin in form of (k,Pk)
        ### Concatenate the total list of (k,Pk) from each seed into a list like [k, [Pk1,Pk2,..]]###
        ### To access unique k for ith graph write: k, Pk = zip(*zipped_kPk[i])
        d = cl.defaultdict(list)
        for k, v in zipped_kPk[i]:# for each key and value in zipped_kPk
            d[k].append(v)
        zipped_kPk[i] = dict(sorted(d.items())) # convert zipped_kPk to dictionary
        
        # pad the Pk lists shorter than Nseeds with 0 at the end
        # Need to pad the Pk with only 1 data point with 0s at the end, otherwise the mean would be an overestimate. 
        for k in zipped_kPk[i].keys():
            zipped_kPk[i][k]=np.pad(zipped_kPk[i][k],(0,Nseeds-len(zipped_kPk[i][k])),'constant')
        
        # store degree bins k (bin centres), mean probability P(k), and errors P(k)_err for plotting. 
        k_bins[i]  = np.array(list(zipped_kPk[i].keys()))
        Pk_mean[i] = np.array(list({k:np.mean(v) for k,v in zipped_kPk[i].items()}.values()))
        Pk_err[i] = np.array(list({k:np.std(v)/np.sqrt(Nseeds) for k,v in zipped_kPk[i].items()}.values())) # np.std(v) is sample standard deviation. Divide by sqrt(number of runs) to get standard error of mean
        
        # store raw data
        k_bins_raw[i], Pk_raw[i] = logbin(total_k[i], 1.0,True)
    return k_bins_raw, Pk_raw, k_bins, Pk_mean, Pk_err

def dataCollapsePk(Nparam, k_bins, Pk_mean, Pk_err, attachment='p'):
    k_bins_collapse = [[] for i in range(Nparam)]
    Pk_mean_collapse = [[] for i in range(Nparam)]
    Pk_err_collapse = [[] for i in range(Nparam)]
    for i in range(Nparam):
        # collapse k_bins and Pk_mean data by scaling with k1 and P_inf(k)
        k_bins_collapse[i]  = k_bins[i]/calck1(m_ls[i],N_ls[i],attachment)
        Pk_mean_collapse[i] = Pk_mean[i]/calcPk(m_ls[i],k_bins[i],attachment)
        Pk_err_collapse[i]  = Pk_err[i]/calcPk(m_ls[i],k_bins[i],attachment)
    return k_bins_collapse, Pk_mean_collapse, Pk_err_collapse
    
def plotDegreeDist(Nparam,m_ls,N_ls,k_bins_raw,Pk_raw,k_bins,Pk_mean,Pk_err,fix='m',attachment='p'):
    plt.figure()
    if fix=='m': label,param='N',N_ls
    elif fix=='N': label,param='m',m_ls
    for i in range(Nparam):
        # raw data
        plotloglog(k_bins_raw[i], Pk_raw[i],'Degree k','Probability P(k)','%s=%d raw'%(label,param[i]),'',True,False,False,marker='.',color=lineColor(i),Alpha=0.3)
        # log-binned data
        if Pk_mean[i][0]<Pk_mean[i][1]: k_bins[i], Pk_mean[i], Pk_err[i] = k_bins[i][1:], Pk_mean[i][1:], Pk_err[i][1:]  # prevent lower p(k) at the beginning due to binning position
        plotloglog(k_bins[i],Pk_mean[i],r'Degree $k$',r'Probability $p(k)$','%s=%d binned'%(label,param[i]),'',True,False,False,marker='x-',color=lineColor(i))
        plt.fill_between(k_bins[i], Pk_mean[i]-Pk_err[i],Pk_mean[i]+Pk_err[i],color=lineColor(i),alpha=0.2) 
        # theory Pk- using the same set of k as the log-binned data
        Pk_theory = calcPk(m_ls[i],k_bins[i],attachment)
        plt.plot(k_bins[i],Pk_theory,label=None,linestyle='--',color=lineColor(i)) #label='%s=%d theory'%(label,param[i])
        plt.legend()
        
        """
        Stastical testing: Kolmogorov-Smirnov (KS) test
        
        Compare the numberical degree distribution to the theoretical power law. 
        Output: max difference D between the measured and theory distributions.
                p-value for the correlation between the two. (1=perfect correlation)
        
        !!! neglect k<m and k=k1(max expected degree) to minimise finite size effects
        """
        KS_test = ks_2samp(Pk_mean[i][findIndex(k_bins[i],m_ls[i]):-4],Pk_theory[:-4]) # only want k>=m where theory is valid, and ignore last 4 entries due to finite size effects (rapid decay)
        print('m = %s\tN = %.1e:\tD = %s, p = %s' % (m_ls[i],N_ls[i],'%.6f'%KS_test[0],'%.6f'%KS_test[1]))

def plotDegreeDistCollapse(Nparam,N_ls,k_bins_collapse,Pk_mean_collapse,Pk_err_collapse):
    """
    Plot data collapse of degree distribution. Only for fixed m, varying N. 
    """
    plt.figure()
    label,param='N',N_ls
    for i in range(Nparam):
        # log-binned data
        plotloglog(k_bins_collapse[i],Pk_mean_collapse[i],r'Scaled degree $k/k_1$',r'Scaled probability $p(k)/p_\infty(k)$',
                   '%s=%d binned'%(label,param[i]),'',True,False,False,marker='.',color=lineColor(i))
        plt.fill_between(k_bins_collapse[i], Pk_mean_collapse[i]-Pk_err_collapse[i],
                         Pk_mean_collapse[i]+Pk_err_collapse[i],color=lineColor(i),alpha=0.2) 

def plotLargestDegree(m_ls,N_ls,attachment,k1_mean,k1_err):
    # plot numerical largest degree
    plotloglog(N_ls,k1_mean,'Number of nodes N','Largest degree k1','Numerical','k1',True,True,False,marker='.-',color=lineColor(0))
    # plot error
    plt.fill_between(N_ls,k1_mean-k1_err, k1_mean+k1_err,color='orange',alpha=0.5)
    # plot theoretical largest degree
    k1_theory= calck1(m_ls[0],np.array(N_ls),attachment)
    plotloglog(N_ls,k1_theory,'Number of nodes N','Largest degree k1','Theoretical','',True,False,False,marker='--',color=lineColor(0))
    
    # Statistical test for numerical vs theory distributions: KS test
    KS_test_k1 = ks_2samp(k1_mean,k1_theory)
    print('m = %s, KS test:\tD = %s, p = %s' % (m_ls[0],'%.6f'%KS_test_k1[0],'%.6f'%KS_test_k1[1]))
        
def find_mean_distribution(data_ls):
    # Calculate error in data and find mean distribution
    err = np.array(data_ls).std(axis=1)
    data_mean_ls = np.mean(data_ls, axis=1)
    return data_mean_ls, err

def saveData(filename, data):
    outfile = open(filename,'wb')
    pickle.dump(data,outfile)
    outfile.close()
    
def loadData(filename):
    infile = open(filename,'rb')
    outfile = pickle.load(infile)
    infile.close()
    return outfile
# In[]:
"""
Task 2: Testing the network to ensure it is performing as expected
"""  
m = 8
r = int(m/2)
g1 = BA_model(m,N0=m+1,attachment='pr',r=r,Graph0='complete',test=True)

print('Testing graph:\n_______________')   
g1.summariseGraph()
plt.figure('Time = %.1f, N=%d, m=%d'%(g1.t,g1.N,g1.m))
pos = nx.circular_layout(g1.G)
nx.draw(g1.G, pos, with_labels=True)
Nsteps = 3 
for i in range(Nsteps): 
    g1.nextTimestep()
    print('time: %.1f'%g1.t)
    print('Node added: ',g1.node_ls[-1])
    print('Edges added:',g1.edge_ls[-m:])
    print('\n')
    plt.figure('Time = %.1f, N=%d, m=%d'%(g1.t,g1.N,g1.m))
    pos = nx.circular_layout(g1.G)
    nx.draw(g1.G, pos, with_labels=True)
# checked number of edges in graph matches len(g1.edge_ls)
# checked adj matrix of a node i is the same as g1.attached_node_ls.count(i)
# Check also the nodes chosen as nearest neighbours, the total number of nodes, the total number of edges, 
# and the network degree distribution. For preferential attachment, 
# the node list was also checked at each timestep and the number of times a node 
# appeared was compared with its degree to ensure they matched.
# In[]:
'''
Phase 1: Task 3 - Fixing N but varying m
Consider N = 10^4
and m = 3^n for n= {0,1,2,3,4,5,6}
'''
print('\nPerforming Task 3: Degree distribution. Fixed N, varying m.\n',
      '_________________________________________________________________')
print('Simulating graphs...')

Graph0       ='complete'
attachment   ='pr'
fix          ='N'
Nparam       = 1 # number of m values to test
N_ls         = [10000]*Nparam #Total number of nodes wanted in final graph
m_ls         = [2**i for i in range(5,Nparam+5)] # list of different m values 
Nseeds       = 10  # number of iterations (with different seed) for each m value (for averaging later)
seeds        = np.random.choice(range(1,int(1e5)),Nseeds,False) #randomly pick seeds
logbin_scale = 1.1

t0 = time.perf_counter()

G_ls, indiv_k, total_k, k1_ls = makeMultipleGraphs(Nparam,Nseeds,m_ls, N_ls, fix, attachment, Graph0)
k_bins_raw, Pk_raw, k_bins, Pk_mean, Pk_err= calcPk_avg(Nparam,Nseeds,indiv_k,total_k)
dt = time.perf_counter() - t0
print('Number of seeds= %d' %(Nseeds))
print('\t Time taken = %.3e sec' %(dt))

### Plot degree distribution P(k) vs degree k on log-log scale
#______________________________________________________________________________
t0 = time.perf_counter()
print('\nPerforming KS test: numerical p(k) vs theory p(k)')   
plotDegreeDist(Nparam, m_ls, N_ls, k_bins_raw, Pk_raw,
               k_bins, Pk_mean, Pk_err, fix, attachment)
plt.title('N=%d'%(N_ls[0]))
dt = time.perf_counter() - t0
print('\t Time taken = %.3e sec' %(dt))

### store variables in list for saving later
fixN_pr_m32 = [Nparam, m_ls, N_ls, k_bins_raw, Pk_raw, k_bins, Pk_mean, 
                   Pk_err, fix, attachment, Nseeds, seeds]

"""
Save data
"""
save = True # True for saving data.
print('\nSaving Data...')
t0 = time.perf_counter()
if save:
    #### Save data using pickle
    
    saveData('fixN_pr_m32'  ,fixN_pr_m32)
    #saveData('fixm_pr_N1e6'  ,fixm_p_N1e6)

print('\tTime Taken= %.3e'%(time.perf_counter()-t0))
print('Data Saved.')
# In[]:
'''
Phase 1: Task 4 - Fixing m but varying N
Consider m = 2
and N = 10^n for n =[2,3,4,5,6]
'''
print('\nPerforming Task 3: Degree distribution. Fixed m, varying N.\n',
      '_________________________________________________________________')
print('Simulating graphs...')
Graph0       ='complete'
attachment   ='pr'
fix          ='m'
Nparam       = 1 # number of N values
N_ls         = [10**(i+1) for i in range(4,Nparam+4)] #list of total nodes wanted
m_ls         = [2]*Nparam # fixed number of edges added for each node
Nseeds       = 10 # number of runs for each (m,N)
seeds        = np.random.choice(range(1,int(1e5)),Nseeds,False) #randomly pick seeds
logbin_scale = 1.1
t0 = time.perf_counter()

G_ls, indiv_k, total_k, k1_ls = makeMultipleGraphs(Nparam,Nseeds,m_ls,N_ls,fix,attachment,Graph0)
k_bins_raw, Pk_raw, k_bins, Pk_mean, Pk_err= calcPk_avg(Nparam,Nseeds,indiv_k, total_k)
k_bins_collapse, Pk_mean_collapse, Pk_err_collapse = dataCollapsePk(Nparam, k_bins, Pk_mean, Pk_err,attachment)
dt = time.perf_counter() - t0
print('Number of seeds= %d' %(Nseeds))
print('\t Time taken = %.3e sec' %(dt))    

# Plot degree distribution P(k) vs degree k on log-log scale
#______________________________________________________________________________
t0 = time.perf_counter()
print('\nPerforming KS test: numerical p(k) vs theory p(k)')   
# plot degree distribution
plotDegreeDist(Nparam,m_ls,N_ls,k_bins_raw,Pk_raw,
               k_bins,Pk_mean,Pk_err,fix,attachment)
plt.title('m=%d'%(m_ls[0]))

# plot data collapsed degree distribution
plotDegreeDistCollapse(Nparam,N_ls,k_bins_collapse,Pk_mean_collapse,Pk_err_collapse)
plt.title('m=%d'%(m_ls[0]))
dt = time.perf_counter() - t0
print('\t Time taken = %.3e sec' %(dt)) 

# Plot k1 vs N on log-log scale
#______________________________________________________________________________
print('\nPerforming KS test: numerical k1 vs theory k1')  
""" Plot largest degree vs N"""
t0 = time.perf_counter()
# calculate error and mean of k1_ls
k1_mean,k1_err=find_mean_distribution(k1_ls)
k1_err = k1_err/np.sqrt(Nseeds) # convert population error to mean error

# plot largest degree distribution
plotLargestDegree(m_ls,N_ls,attachment,k1_mean,k1_err)
plt.title('m=%d'%(m_ls[0]))

dt = time.perf_counter() - t0
print('\t Time taken = %.3e sec' %(dt))   
### store variables in list for saving later
fixm_pr_1e5 = [Nparam,m_ls,N_ls,k_bins_raw,Pk_raw,k_bins,Pk_mean,
                     Pk_err,fix,attachment,k_bins_collapse,Pk_mean_collapse,
                     Pk_err_collapse,k1_mean,k1_err,Nseeds, seeds]

"""
Save data
"""
save = True # True for saving data.
print('\nSaving Data...')
t0 = time.perf_counter()
if save:
    #### Save data using pickle
    
    #saveData('fixN_rand_m1to243'  ,fixN_r_m1to243)
    saveData('fixm_pr_N1e5'  ,fixm_pr_1e5)

print('\tTime Taken= %.3e'%(time.perf_counter()-t0))
print('Data Saved.')
# In[]:
"""
Load data
"""
################################ fix N ########################################
#fixN_p1 = loadData('fixN_pref_m243')
#fixN_p = loadData('fixN_pref_m1to243')# 1000 iter for m=1,3, 100 iter for the rest. 
#fixN_r = loadData('fixN_rand_m1to243')
#fixN_pr = loadData()

################################ fix m ########################################
#fixm_p = loadData('fixm_pref_N10to100000')
#fixm_p1 = loadData('fixm_pref_N1e6-10seeds')
#fixm_r = loadData('fixm_rand_N10to1e5')
#fixm_r1 = loadData('fixm_rand_N1e6-10seeds')

# In[]:
"""
Re-plot data
"""
plt.close()
################################ fix N ########################################
## Preferential
#plotDegreeDist(Nparam,m_ls,N_ls,k_bins_raw,Pk_raw,k_bins,Pk_mean,Pk_err,fix='m',attachment='p')
#               0      1    2    3          4      5      6       7      8       9 
N = 1e4
# adding new m values to the list of data
#fixN_r[0]+=1
#fixN_r[1].extend(fixN_r1[1])
#fixN_r[2].extend(fixN_r1[2])
#fixN_r[3].extend(fixN_r1[3])
#fixN_r[4].extend(fixN_r1[4])
#fixN_r[5].extend(fixN_r1[5])
#fixN_r[6].extend(fixN_r1[6])
#fixN_r[7].extend(fixN_r1[7])

## changing individual m value data 
#fixN_p[3][1]=k_bins_raw[0]
#fixN_p[4][1]=Pk_raw[0]
#fixN_p[5][1]=k_bins[0]
#fixN_p[6][1]=Pk_mean[0]
#fixN_p[7][1]=Pk_err[0]

#saveData('fixN_pref_m1to243'  ,fixN_p)

#plotDegreeDist(fixN_p[0],fixN_p[1],fixN_p[2],fixN_p[3],fixN_p[4],
#               fixN_p[5],fixN_p[6],fixN_p[7],fixN_p[8],fixN_p[9])
#
#plt.title('N=%d'%(N))

## Random
#plotDegreeDist(fixN_r[0],fixN_r[1],fixN_r[2],fixN_r[3],fixN_r[4],
#               fixN_r[5],fixN_r[6],fixN_r[7],fixN_r[8],fixN_r[9])
##
#plt.title('N=%d'%(N))


## Existing vertices model

################################ fix m ########################################
## Preferential
m=3

#fixm_p_N100000 = [Nparam,m_ls,N_ls,k_bins_raw,Pk_raw,k_bins,Pk_mean,Pk_err,fix,attachment,k_bins_collapse,Pk_mean_collapse,Pk_err_collapse,k1_mean,k1_err,Nseeds, seeds]
#                  0      1    2    3          4      5      6       7      8   9          10              11               12              13      14     15      16


#fixm_p[0]+=1
#fixm_p[1].extend(fixm_p1[1])
#fixm_p[2].extend(fixm_p1[2])
#fixm_p[3].extend(fixm_p1[3])
#fixm_p[4].extend(fixm_p1[4])
#fixm_p[5].extend(fixm_p1[5])
#fixm_p[6].extend(fixm_p1[6])
#fixm_p[7].extend(fixm_p1[7])
#fixm_p[10].extend(fixm_p1[10])
#fixm_p[11].extend(fixm_p1[11])
#fixm_p[12].extend(fixm_p1[12])
#fixm_p[13] = list(fixm_p[13])
#fixm_p[13].extend(fixm_p1[13])
#fixm_p[14] = list(fixm_p[14])
#fixm_p[14].extend(fixm_p1[14])
#fixm_p[13]=np.array(fixm_p[13])
#fixm_p[14]=np.array(fixm_p[14])
##
#plotDegreeDist(fixm_p[0],fixm_p[1],fixm_p[2],fixm_p[3],fixm_p[4],
#               fixm_p[5],fixm_p[6],fixm_p[7],fixm_p[8],fixm_p[9])
#plt.title('m=%d'%(m))
###
#plotDegreeDistCollapse(fixm_p[0],fixm_p[2],fixm_p[10],fixm_p[11],fixm_p[12])
#plt.title('m=%d'%(m))
#
#plotLargestDegree(fixm_p[1],fixm_p[2],fixm_p[9],fixm_p[13],fixm_p[14])
#plt.title('m=%d'%(m))

## Random
#saveData('fixm_rand_N10to1e6'  ,fixm_r)

#fixm_r[0]+=1
#fixm_r[1].extend(fixm_r1[1])
#fixm_r[2].extend(fixm_r1[2])
#fixm_r[3].extend(fixm_r1[3])
#fixm_r[4].extend(fixm_r1[4])
#fixm_r[5].extend(fixm_r1[5])
#fixm_r[6].extend(fixm_r1[6])
#fixm_r[7].extend(fixm_r1[7])
#fixm_r[10].extend(fixm_r1[10])
#fixm_r[11].extend(fixm_r1[11])
#fixm_r[12].extend(fixm_r1[12])
#fixm_r[13] = list(fixm_r[13])
#fixm_r[13].extend(fixm_r1[13])
#fixm_r[14] = list(fixm_r[14])
#fixm_r[14].extend(fixm_r1[14])
#fixm_r[13]=np.array(fixm_r[13])
#fixm_r[14]=np.array(fixm_r[14])

#plotDegreeDist(fixm_r[0],fixm_r[1],fixm_r[2],fixm_r[3],fixm_r[4],
#               fixm_r[5],fixm_r[6],fixm_r[7],fixm_r[8],fixm_r[9])
#plt.title('m=%d'%(m))
##↕
#plotDegreeDistCollapse(fixm_r[0],fixm_r[2],fixm_r[10],fixm_r[11],fixm_r[12])
#plt.title('m=%d'%(m))
#
#plotLargestDegree(fixm_r[1],fixm_r[2],fixm_r[9],fixm_r[13],fixm_r[14])
#plt.title('m=%d'%(m))

## Existing verticies model
# In[]:
"""
Completed!
"""
print('Total time taken: %.3e sec'%(time.perf_counter()-start_time))
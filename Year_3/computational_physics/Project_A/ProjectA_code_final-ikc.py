# author: ikc15
# date: 13/11/2017
# In[]:
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
# In[]:
"""
####################### Q1. Floating point variables ##########################
"""
def machineAccuracy():
    """
    Finds the smallest meaningful number that can be subtracted from 1.
    Output: smallest number 2^-(i-1) 
    """
    
    for i in range (0,100,1):                         # loop through exponent from 0 to -99 in steps of -1
        a = float(1) - 2**(-i)                        # subtract 2^i from one (N.B. plus sign gives different result)
        if a==1:                                      # then i-1 was the smallest exponent that produced meaningful subtraction
            print('Machine accuracy: 2^',-(i-1))
            break
    
    for i in range (0,100, 1):                        # loop through exponent from 0 to -99 in steps of -1
        a = np.single(1) - np.single(2**(-i))         # subtract 2^i from one
        if a==1: 
            print ('single precision: 2^',-(i-1))     # single: 1+8+23 = 32bit (+1 bit implicit)-> expect theoretical accuracy to be 2^-23 (=1e-7) 
            break
    
    for i in range (0,100, 1):                        # loop through exponent from 0 to -99 in steps of -1
        a = np.double(1) - np.double(2**(-i))         # subtract 2^i from one
        if a==1: 
            print ('double precision: 2^',-(i-1))     # double -> 1+11+52=64bit (+1 bit implicit)-> expect theoretical accuracy to be 2^-52 (=1e-16)
            break
    
    for i in range (0,100, 1):                        # loop through exponent from 0 to -99 in steps of -1
        a = np.longdouble(1) - np.longdouble(2**(-i)) # subtract 2^i from one
        if a==1: 
            print ('extended precision: 2^',-(i-1))   # extended -> 1+15+64=80bit -> expect theoretical accuracy to be 2^-64 (=1e-20) 
            break
    return '---End of Question 1---'
print('---Question 1. a), b)---','\n')
print(machineAccuracy(),'\n')
# In[]: #need to modify b) to return L,U in a single NxN matrix without the diagonal of L.
"""
######################## Q2. Matrix methods ###################################
"""
## a) LU decompoisition of NxN matrix using Crout's method (A = LU)
class Matrix:
    """
    A Matrix class containing Matrix objects which can:
        -> print itself and its shape
    """
    def __init__(self, m=np.zeros(1)):
        self.matrix = m                             # prints itself
        self.shape = len(m)                         # shape of matrix (len(m)xlen(m))
        
    def L(self):
        """ Output: L matrix """ 
        return self.LUdecomp()[0]
    
    def U(self):
        """ Output: U matrix """
        return self.LUdecomp()[1]
    
    def LUcomb(self):
        """ Output: L and U combined into an (N × N) matrix containing all the
                    elements of U, and the non-diagonal elements of L. """
        return self.LUdecomp()[2]
    def pivot(self, mat, row):
        """ 
        Pivot matrix A to avoid division by zero in LUdecomp.
        Input: mat (matrix to be pivotted), row (pivot this row)
        Output: pivotted matrix
        """
        for ii in range(row+1,len(mat)):            # search rows below original row until non-zero value found in that column
            if abs(mat[ii,row]) > 1e-15:            # found non-zero entry 
                temp = mat[row].copy()              # copy original row 
                mat[row] = mat[ii]                  # replace original row with iith (non-zero) row
                mat[ii] = temp                      # replace iith row with original row                    
        if abs(mat[row,row]) < 1e-15:               # check diagonal again after pivot
            raise ValueError('Division by zero - singular matrix')
        return mat 
    
    def LUdecomp(self):
        """ 
        Crout's method: Decompose a NxN matrix A into lower- (L) and upper- (U) triangular  matrices. 
        Output: L, U, LUcomb
        """
        N = self.shape                              # shape of matrix
        L = np.zeros((N,N))                         # define shape of L and U matrices
        U = np.zeros((N,N))
        LUcomb = np.zeros((N,N))
        for i in range(N):                      
            L[i,i] = 1                              # Step 1: set diagonal of L to 1.
                                                    # Step 2: use Crout's Algorithm to get entries for L and U:     
        for i in range(N):                          # for each row i
            for j in range(N):                      # cycle through columns j
                pivot = False
                LUl = 0 
                LUu = 0                             # intermediate variables for calculating sums
                if i<=j:                            # in upper triangle
                    while not pivot:                # while loop -> to re-evaluate U[i,j] after pivoting 
                        for k in range(i):
                            LUl += L[i,k]*U[k,j] 
                        U[i,j] = self.matrix[i,j] - LUl                 # calculate U_ij
                        LUcomb[i,j] = U[i,j]
                        if (i==j) and (abs(U[j,j])<1e-15):              # if the diagonal entry of U = 0 
                            print('pivot')
                            self.matrix = self.pivot(self.matrix, j)    # pivot row in matrix A.
                        else: 
                            pivot = True
                if i>j:                                                 # in lower triangle
                    for k in range(j):
                        LUu += L[i,k]*U[k,j]
                    L[i,j] = 1/U[j,j]* \
                                  (self.matrix[i,j] - LUu)              # calculate L_ij 
                    LUcomb[i,j] = L[i,j]
        return L, U, LUcomb
               
## b) decompose A, and compute det(A)
    def det(self):
        """ 
        Calculates determinant of matrix A using matrix U.
        Output: det (determinant of a matrix)
        """                                            
        self.U()                                    # Step 1: perform LU decomposition on matrix A to get U
        det = 1
        for i in range(self.shape):                      
            det *= self.U()[i,i]                    # Step 2: determinant = product of diagonal entries in U 
        return det                                        
        
## c) solve LUx = b for x using 
    def solveLUb(self,L,U,b):
        """
        Solves matrix equation of form LUx = b, by splitting up into 
        Ly = b and Ux = y. Uses forward and back substitution to get x.
        Input: L, U, b
        Output: x (solution to matrix eqn)
        """
        N = b.shape[0]                              # shape of b     
        y = np.zeros((N,1))                         # define vector x, y
        x = np.zeros((N,1))                            
    
        y[0] = b[0]/L[0,0]                          # Step 1: solve for 1st entry in y
        
        for i in range(1, N):                       # Step 2- compute y from top to bottom
            LY = 0.
            for k in range(i):
                LY += L[i,k]*y[k]
            y[i] = 1/ L[i,i] * (b[i] - LY)
            
        #print(np.matmul(self.L,y)-b < 1e-10)       # validated y by checking Ly = b
        
        x[N-1] = y[N-1]/U[N-1,N-1]                  # Step 3: solve for last entry in vector x

        for i in range(N-2,-1,-1):                  # Step 4: compute x from the bottom up
            UX = 0.
            for k in range(i+1, N):
                UX += U[i,k]*x[k]
            x[i] = 1/U[i,i] *(y[i] - UX)
        return x
    
    def getCol(self, array, i):
        """ 
        Returns ith column of an array. 
        Input: array, i
        Output: ith column
        """
        return np.array([array[row,i] for row in range(len(array))])

    def inv(self):
        """ Output: inv (the inverse of a matrix). """
        N = self.shape                              # shape of matrix
        inv =  np.zeros((N,N))                      # define inverse matrix shape
        self.LUdecomp()                             # find L, U 
        
        identity = np.eye(N)                        # define identity matrix shape
        
        for j in range(N):                          # compute jth column of inverse
            b = self.getCol(identity, j)            # set b = jth column in identity
            AInvj = self.solveLUb(
                    self.L(),self.U(),b)            # solve for x using method solveLUb()
            for i in range(N):
                inv[i,j] = AInvj[i]                 # populate inverse matrix                          
        return inv
    
## d) solve Ax = b
A = Matrix(m=np.array([[2., 1., 0., 0., 0.],\
                      [3., 8., 4., 0., 0.],\
                      [0., 9., 20., 10., 0.],\
                      [0., 0., 22., 51., -25.],\
                      [0., 0., 0., -55., 60]]))
                                   
b = np.array([[2.],[5.],[-4.],[8.],[9.]]) 

#print(np.linalg.det(A.matrix))                     # validation of results with Python built-in fn
#print(np.linalg.solve(A.matrix,b))   
#print(np.linalg.inv(A.matrix))     

print('Q2.a),b),c)', '\n')
print('LU_combined = ', '\n', A.LUcomb(), '\n')
print('det(A)= ', '\n', A.det(), '\n')
print('Q2.d) Solve matrix equation', '\n')
print('x = ', '\n', A.solveLUb(A.L(), A.U(), b),'\n')  
print('Q2.e) Find inverse of matrix A', '\n')
print('A^-1 =', '\n', A.inv(),'\n')      
print('---End of Question 2---','\n')
# In[]:
"""
######################## Q3. Interpolation ###################################
"""
## a) Perform linear interpolation on a tabulated set of x–y data

#data set
xi = [-2.1, -1.45, -1.3, -0.2, 0.1, 0.15, 0.8, 1.1, 1.5, 2.8, 3.8]    
yi = [0.012155, 0.122151, 0.184520, 0.960789, 0.990050, 0.977751,
      0.527292, 0.298197, 0.105399, 3.936690e-4, 5.355348e-7]

fig = plt.figure('Interpolation')                   # create figure to plot data set and interpolation curves
ax = fig.add_subplot(111)
ax.plot(xi, yi,'x',color= 'red',label='Raw data')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')

def linInterp(xi, yi, x):
    """
    Performs linear interpolation on a tabulated dataset.
    Input: xi & yi (data set), x (value to be interpolated at) 
    Output: f (interpolated value at x)
    """
    for i in range(len(xi)):                        # Step 1: find nearest 2 data pts from x (assumes xi in ascending order)
        if x == xi[i]:                              #         if x is one of the data pts 
            return yi[i]                            #         return corresponding yi   
        elif xi[i] < x:                             #         otherse loop through data until we find a value > x
            continue
        elif xi[i] > x:                             #         when value > x, the nearest data pts from x are xi[i] and xi[i-1]
            f = 1/(xi[i] - xi[i-1])*\
                ((xi[i] - x)*yi[i-1]+\
                 (x-xi[i-1])*yi[i])                 # Step 4: perform linear interpolation to find f(x)
            return f

## b) Perform cublic spline interpolation on a tabulated set of x–y data
def get_fpp(xi, yi):
    """
    Calculates f'' in matrix eqn: triDiagMat*fpp = F
    Input: xi, yi (data set)
    Output: fpp (=f'')
    """
    
    #global fpp, F, triDiagMat                       # Aim: solve matrix eqn: triDiagMat*fpp = F to get fpp                                                  
    n = len(xi)
    triDiagMat = np.zeros((n-2,n-2))                # Step 1: define shape of tridiagonal matrix 
    F = np.zeros(n-2)                               # Step 2: define shape of F. 
    
    for i in range(1,n-2):                          # Step 3: for each i:
        z = (xi[i+1] - xi[i])/6                     #         populate triDiagMat in an upside down L fashion 
        b = (xi[i+1] - xi[i-1])/3
        triDiagMat[i-1,i-1] = b                     #         main diag: bi
        triDiagMat[i, i-1] = z                      #         lower diag: ai (N.B c_i = a_i+1)
        triDiagMat[i-1, i] = z                      #         upper diag: ci
        F[i-1] = (yi[i+1] - yi[i])/ \
                 (xi[i+1] - xi[i])- \
                 (yi[i] - yi[i-1])/ \
                 (xi[i] - xi[i-1])                  #          populate F
    F[n-3] = (yi[n-1] - yi[n-2])/ \
                 (xi[n-1] - xi[n-2])- \
                 (yi[n-2] - yi[n-3])/ \
                 (xi[n-2] - xi[n-3])                # Step 4: add the last element into F
    triDiagMat[n-3, n-3] = \
                      (xi[n-1] - xi[n-3])/3         # Step 5: add the last element of b into main-diagonal
    triDiagMat = Matrix(m=triDiagMat)
    triDiagMat.LUdecomp()                           # Step 6: decompose triDiagMat into L and U
    
    fpp = triDiagMat.solveLUb(
                  triDiagMat.L(),triDiagMat.U(),F)  # Step 7: solve for fpp using Matrix method solveLUb()

    fpp = np.concatenate(([[0]],\
                          fpp,\
                          [[0]]),\
                          axis = 0)                 # Step 8: append zero to top and bottom of f'' array -> BC for 1st and last point.
    return fpp

def cubicInterp(xi, yi, x,fpp):
    """
    Performs cubic spline interpolation on a tabulated dataset.
    Input: xi & yi dataset, value of x to be interpolated at 
    Output: interpolated value f at x
    """
                                                    
    for i in range(len(xi)):                        # Step 1: find nearest 2 data pts from x (assumes xi in ascending order)
        if x == xi[i]:                              #         if x is one of the data pts   
            return yi[i]                            #         return corresponding yi     
        elif xi[i] < x:                             #         otherse loop through data points until we find one > x
            continue
        elif xi[i] > x:
            x1, x2, = xi[i-1], xi[i]                #         nearest data pts from x are xi[i] and xi[i-1]
            y1, y2 = yi[i-1], yi[i]                 # Step 2: get values of yi, fpp at these indices
            fpp1, fpp2 = fpp[i-1], fpp[i]           
            A = (x2 - x) / (x2 - x1)                # Step 3: define coefficients A, B, C, D  
            B = 1 - A                               
            C = 1/6*(A**3 - A)*(x2 - x1)**2
            D = 1/6*(B**3 - B)*(x2 - x1)**2         
            f = A*y1 + B*y2 + C*fpp1 + D*fpp2       # Step 4: perform cubic spline interpolation to find f(x)
            return f

                                                    # Aim: plot straight lines between data points for whole dataset 
n = 1000                                            # Step 1: set no. of interpolation points
linarray = []                                       # Step 2: define linarray to store the interpolation data
X = np.linspace(min(xi), max(xi), n)                # step 3: generate n uniform samples of x
for i in range(len(X)):
    linarray.append(linInterp(xi,yi,X[i]))          # Step 4: populate linarray with linear interpolation values
ax.plot(X, linarray, '-',                           # Step 5: plot 
        color='green',label='Linear',alpha=0.5)    
ax.legend()

fpp = get_fpp(xi,yi)                                # Aim: plot straight lines between data points for whole dataset 
cubarray = []                                       # define array shape to store the interpolation data
for i in range(len(X)):
    cubarray.append(cubicInterp(xi,yi,X[i],fpp))
ax.plot(X, cubarray, '-',                            
        color='blue',label='Cubic spline',alpha=0.5)    
ax.legend()
      
## c) plot 
print('Q3. Interpolation', '\n')
print('Linear: f(%s) = %.3f'%(2,linInterp(xi, yi, 2))) 
print('Cubic spline: f(%s) = %.3f'%(3.5,cubicInterp(xi,yi,3.5,fpp)))
plt.show()
#fig.savefig('interpolation.jpg', dpi=150, bbox_inches='tight')
print('---End of Question 3---','\n')
# In[]:
"""
######################## Q4. Fourier Transforms ################################
"""
#a) Use np.fft to convolve the signal function with the response function.
N = 2**8                                             # no. of samples (even no. to increase efficiency)
T = 8                                                # period (minimum of 8 to include complete tophat fn)
dt= T/N                                              # sample spacing in time domain
t = np.zeros(N)
h = np.zeros(N)
g = np.zeros(N)

def signal(t):
    """ 
    signal: top-hat fn h(t)-> h(t) = 5 for 2<=t<=4 
    input: t
    output: h
    """
    if t>=2 and t<=4:
        return 5
    else:
        return 0
   
def response(t):
    """ 
    gaussian (normalised): g(t)
    input: t
    output: g
    """
    return 1/np.sqrt(2*np.pi)*np.exp(-(t*t/2))

def convolve(h, g):
    """ 
    Convolves signal (h) with response (g) using np.fft and convolution theorem 
    input: h, g
    output: hConvgR (real part of convolution)
    """
    H = np.fft.fft(h)                                # compute fft of h(t) and g(t)
    G = np.fft.fft(g)
    HG = H*G                                         # compute H*G
    hConvgR = dt*np.real(\
            np.fft.fftshift(np.fft.ifft(HG)))        # Convolution theorem: hConvg = ifft(H*G) -> dt is normalising factor
    return hConvgR                                   # np.fft.fftshift shifts the output to centre around the signal.                          

for n in range(N):                                   # populate time, signal and response array    
    t[n] = n*dt - T/2
    h[n] = signal(t[n])
    g[n] = response(t[n])
h_pad = np.pad(h,(int(N/2),),'constant')             # zero pad signal on both sides
g_pad = np.pad(g,(int(N/2),),'constant')             # zero pad response on both sides
hConvg = convolve(h_pad,g_pad)                       # convolute h and g

print('Q4. Fourier transform', '\n')                 # plot signal, response, and convolution

time_pad = [(i*dt + 2*min(t)) for i in range(len(hConvg))] # 2*min(t) accounts for padding
fig1 = plt.figure()
plt.plot(time_pad, h_pad, '-', color='blue',label= 'h(t): top-hat')
plt.plot(time_pad, g_pad, '-', color='green',label= 'g(t): gaussian')
plt.plot(time_pad, hConvg, '--', color = 'red',label= '(h*g)(t): convolution')
plt.xlabel('time/ s')
plt.grid(True)
plt.legend()
plt.show()
#fig1.savefig('fft.jpg', dpi=150, bbox_inches='tight')
print('---End of Question 4---','\n')
# In[]:
"""
######################## Q5. Random Numbers ###################################
"""
##a) compute 10e5 uniformly-distributed random numbers over interval [0,1]
N = 100000                                           # number of samples
seed = 2                                             # seed 

def uniformrandom(x1,x2, seed, N):
    """ 
    Returns N uniformly distributed random numbers between x1 and x2 given seed and number of samples.
    Input: x1, x2, seed, N
    Output: random number 
    """
    np.random.seed(seed)                             # seed to initialise pseudo-random number generator
    return np.random.uniform(x1,x2,N)

##b) compute random numbers for interval x[0,pi] with pdf(x) = 0.5*sin(x) using transformation method.

def transform(x):
    """
    Transformation method: transform of uniform deviate to a new one based on PDF.
    E.g. pdf: P(y)= 0.5*np.sin(y) on interval [0,pi]-> integrate P(y) from 0 to y 
              -> equate to x -> rearrange for y(x)
    Input: x (a uniform deviate) 
    Output: y (random number following new pdf)
    """
    y = np.arccos(-2*x+1)                            # x mapped onto y with pdf 0.5*sin(y)
    return y                                

##c) compute random numbers for interval [0,pi] with pdf(x) = 2/pi*sin^2(x) using rejection method.
def reject(seed,N): 
    """
    Rejection method: compares a random number to comparison function to get new deviate based on PDF. 
    #E.g. pdf: P(y)= 2/pi*sin^2(y) on interval [0,pi]
    Input: seed, N (no. of samples)
    """
        
    np.random.seed(seed)                             # seed to initialise pseudo-random number generator
    yiList = []                                      # Step 1: define list to store accepted values of yi
    counter = 0
    
    while counter < 1e10:
        yi = transform(np.random.uniform(0,1))       # Step 2: pick random no. yi using the pdf 0.5*sin(x) from previous part
        
        fyi = np.sin(yi)                             # Step 3: define comparison function f(y) >= P(y) for all y
        
        zi = np.random.uniform(0,fyi)                # Step 4: pick another random number zi between [0,f(yi)]
        
        Pyi = 2/np.pi*(np.sin(yi))**2                # Step 5: Compute probabiility at yi in desired pdf
       
        if Pyi < zi:                                 # Step 6: if probability at yi in desired pdf < zi
            counter += 1                             #               reject yi-> increase counter by 1 -> repeat 
        elif len(yiList) < N:                        #         else: accept yi if list not full
            yiList.append(yi)                        
            if len(yiList) == N:                     #         If yiList is full (with N numbers):
                return yiList                        #               return list of accepted yi
            else:                                    #         else: 
                counter += 1                         #               increase counter by 1 and repeat from Step 2
    raise ValueError('counter exceeded 1e10.')

print('Q5. Random numbers', '\n')
start_time1 = time.clock()                           # record start time- time.clock() has higher precision than time.time()
randarray = transform(uniformrandom(0,1,seed,N))     # populate randarray using transformation method
transform_time = time.clock() - start_time1          # record finish time

start_time2 = time.clock()                           # record start time 
randarray1 = reject(seed, N)                         # populate randarray1 using rejection method
reject_time = time.clock() - start_time2             # record finish time

fig2, ax2 = plt.subplots(3,)                         # plot histograms      
fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.5)
ax2[0].title.set_text('Uniform deviate generator: %.0E samples, range= [0,1]'%N)
ax2[0].plot([x for x in np.linspace(0,1,10)], [1]*10,color='red', label='pdf=1', alpha=0.5)
ax2[0].hist(uniformrandom(0,1,seed,N), bins=100, edgecolor='black', linewidth=0.2,color='blue', normed=True)
ax2[0].legend()
ax2[0].set_ylabel('Relative Freq')
ax2[1].title.set_text('Transformation method: %.0E samples, range= [0,pi]'%N)
ax2[1].plot([x for x in np.linspace(0,np.pi,100)], 0.5*np.sin([x for x in np.linspace(0,np.pi,100)]),color='red',label='pdf=0.5sin(x)',alpha = 0.5)
ax2[1].hist(randarray, bins=100, edgecolor='black',color='blue', linewidth=0.2, normed=True)
ax2[1].legend()
ax2[1].set_ylabel('Relative Freq')   
ax2[2].title.set_text('Rejection method: %.0E samples, range= [0,pi]'%N)
ax2[2].plot([x for x in np.linspace(0,np.pi,100)], 2/np.pi*(np.sin([x for x in np.linspace(0,np.pi,100)]))**2,color='red', label='pdf=2/pi*sin^2(x)',alpha=0.5)
ax2[2].hist(randarray1, bins=100, edgecolor='black',color='blue', linewidth=0.2, normed=True)
ax2[2].legend()
plt.xlabel('x')
ax2[2].set_ylabel('Relative Freq')                    # no. samples /bin = relative freq x N / no. bins
plt.show()
#fig2.savefig('random.jpg', dpi=250, bbox_inches='tight')
print('transformation method time = ', transform_time)
print('rejection method time = ', reject_time)
print ('ratio = ', (reject_time) / (transform_time))  
print('---End of Question 5---','\n')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("--- %s seconds ---" % (time.time() - start_time))
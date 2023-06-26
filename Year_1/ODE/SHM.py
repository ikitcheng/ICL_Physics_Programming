import scipy as sp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

m= 2 #2kg mass
k= 5 #spring constant 
g= 9.8 #local gravity
init= [1.0,0.] # initial conditions for y and y' at t=0:

def f(y,t):
    y0=y[0] #this is y
    y1=y[1] #this is y dot, these will be defined by initial conditions 
    y2=(-k/m)*y0
    return [y1, y2]
    
t= sp.linspace(0, 10.0,100)

sol=odeint(f, init, t)

#displacement time graph 
plt.xlabel("time (s)")

plt.plot(t, sol[:, 0], 'r+-')
#velocity time graph 
plt.xlabel("time (s)")

plt.plot(t, sol[:, 1], 'bo-')
#legend
plt.legend(('displacement (m)','velocity (m/s)'),loc='upper right') 
plt.show() 

print sol




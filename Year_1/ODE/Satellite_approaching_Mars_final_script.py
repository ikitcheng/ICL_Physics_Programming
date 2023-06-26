'''Mars satellite simulation'''
import pylab as py
import scipy as sp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

G = 6.67*10**-11
Mmars = 6.4*10**23
Rmars = 3.4*10**6
Msat = 260
Vmars= 24100

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Defining the system of first order ODEs to be integrated:
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''
Scenario 1: Mars at rest
'''''''''''''''''''''''''''
# In Mars' frame of reference 
def sat_motion(initial,t):
    x = initial[0]
    vx = initial[1]
    y = initial[2]
    vy = initial[3]
    r= np.sqrt(x**2 + y**2)
    ax = (-G*Mmars)*x/r**3
    ay = (-G*Mmars)*y/r**3
    if r<=Rmars:
        return [0,0,0,0]   
    else:
        return [vx, ax, vy, ay] 
        #returns a list of the ODEs to integrate [dx/dt, dvx/dt, dy/dt, dvy/dt]

'''''''''''''''''''''''''''
Scenario 2: Mars in motion
'''''''''''''''''''''''''''
# Assume Mars is moving at 24.1km/s in the +ve x direction

# In Mars' frame of reference 
def sat_motion_1(initial,t):
    x = initial[0]
    vx = initial[1]
    y = initial[2]
    vy = initial[3]
    r = np.sqrt(x**2 + y**2)
    ax = (-G*Mmars)*x/r**3
    ay = (-G*Mmars)*y/r**3
    if r<=Rmars:
        return [0,0,0,0]   
    else:
    # return a list of the ODEs we want to integrate
        return [vx-Vmars, ax, vy, ay] 

# In Sun's frame of reference
def sat_motion_2(initial,t):
    x = initial[0]
    vx = initial[1]
    y = initial[2]
    vy = initial[3]
    Xmars = initial[4]
    r = np.sqrt((x-Xmars)**2 + y**2)
    ax = (-G*Mmars)*(x-Xmars)/r**3
    ay = (-G*Mmars)*y/r**3
    #return a list of the ODEs we want to integrate
    return [vx, ax, vy, ay, Vmars] 

# create a set of values for t over which the ODEs are integrated
t= sp.linspace(0,1e7,1e5)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Task 1: Satellite trajectories: x against y  
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def plot_xy_graph(fig,n): 
    plt.figure(fig)
    if n == 1:
        plt.title("Satellite trajectory relative to Mars ")
    if n == 2:
        plt.title("Satellite trajectory relative to Sun ")
    plt.xlabel("x-displacement (m)")
    plt.ylabel("y-displacement (m)")
    plt.plot(x, y, label='Satellite trajectory')
    mars = plt.Circle((0, 0), radius=Rmars, fc='red')
    plt.gca().add_patch(mars)
    plt.plot(0,0,'g+')
    
    if n == 1:
        plt.axis('scaled')
        plt.axis([-0.7e8, 2e8, -1.85e8, 2e8])
    if n == 2:
        plt.plot(Xmars, Ymars, 'r--', label= 'Mars trajectory')
        plt.axis([0,2e10,-0.5e8,0.5e8])
        plt.legend()
    plt.show()

'''''''''''''''''''''''''''
Scenario 1: Mars at rest
'''''''''''''''''''''''''''
# In Mars' frame of reference
for j in range(200, 1500, 300):
#initial conditions for [x, vx, y, vy] at time t=0:
    initial= [-20*Rmars,0,-20*Rmars,j] 
    sat_soln = odeint(sat_motion, initial, t)
    x=sat_soln[:,0]
    y=sat_soln[:,2]
    plot_xy_graph('Stationary Mars- Displacement graph',1)

'''# Under what circumstances does the satellite collide with Mars?
print ('Initial vy which results in collision:')
collision=[]
for j in range(0,250):
    initial= [-20*Rmars,0,-20*Rmars,j] 
    sat_soln = odeint(sat_motion, initial, t)
    x=sat_soln[:,0]
    y=sat_soln[:,2]
    r = np.sqrt(x**2 + y**2)
    for i in r:
        if i <= Rmars:
            collision.append(j)
            break    
print collision[-1]'''

'''''''''''''''''''''''''''
Scenario 2: Mars in motion
'''''''''''''''''''''''''''
# In Mars' frame of reference 
for j in range(500, 2000,300):
#initial conditions for [x, vx, y, vy] at time t=0:
    initial_1= [-20*Rmars,24100,-20*Rmars,j]
    sat_soln_1 = odeint(sat_motion_1, initial_1, t)
    x = sat_soln_1[:,0]
    y = sat_soln_1[:,2]
    plot_xy_graph('Moving Mars- Displacement graph relative to Mars',1)
    

# In Sun's frame of reference
# Circular orbit initial conditions 
initial_2 = [-Rmars-24e6 ,24100,0,np.sqrt(G*Mmars/(Rmars+24e6)),0]
sat_soln_2 = odeint(sat_motion_2, initial_2, t)
x = sat_soln_2[:,0]
y = sat_soln_2[:,2]
Xmars = sat_soln_2[:,4]
Ymars = 0 * t
plot_xy_graph('Moving Mars-Displacement graph relative to Sun',2)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Task 2: Closest approach and Angular deviation
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''
lists for Task 2
'''''''''''''''''
#list of values for initial velocity, angle of deflection and closest appraoch 
V0=[]
Deflection=[]  
rmin=[]

def plot_closest_approach(fig,n):
    plt.figure(fig)
    plt.title("Closest approach vs Initial velocity graph")
    plt.xlabel("Initial velocity (m/s)")
    plt.ylabel("displacement (m)")
    if n == 1:
        plt.plot(V0, rmin, 'r-', label='Stationary Mars- Closest approach')
        plt.plot([Vesc,Vesc],[5e7,np.absolute(initial[0])],'g--')
    if n == 2: 
        plt.plot(V0, rmin, '#7f0000', label='Moving Mars- Closest approach')
        plt.plot([Vesc,Vesc],[5e7,np.absolute(initial[0])],'g--',label='Escape velocity')
        plt.plot([0,30000],[np.absolute(initial[0]),np.absolute(initial[0])]
        ,'b--', label= 'Closest approach (no gravity)')
    plt.legend()
    leg = plt.legend(loc=9, ncol=4, prop={'size':11},)
    leg.get_frame().set_alpha(0.4)
    plt.grid(True)
    plt.show()
    
                    
def plot_angular_deviation(fig,n):
    plt.figure(fig)
    plt.title("Angular Deviation vs Initial Velocity")
    plt.xlabel("Initial velocity (m/s)")
    plt.ylabel("Angle of Deviation (degrees)")
    if n == 1:
        plt.plot(V0,Deflection,'r-',label='Stationary Mars- Angle of deviation')
        plt.plot([Vesc,Vesc],[0,120],'g--')
    if n == 2: 
        plt.plot(V0,Deflection,'#7f0000',label='Moving Mars- Angle of deviation')
        plt.plot([Vesc,Vesc],[0,120],'g--', label='Escape velocity')
    plt.legend()
    leg = plt.legend(loc=9, ncol=3, prop={'size':11},)
    leg.get_frame().set_alpha(0.4)
    plt.grid(True)
    plt.show()  
    
    
'''''''''''''''''''''''''''
Scenario 1: Mars at rest
'''''''''''''''''''''''''''
for i in range(900,10100,100):
    initial= [-20*Rmars,0,-20*Rmars,i]
    sat_soln = odeint(sat_motion, initial, t)
    x=sat_soln[:,0]
    vx=sat_soln[:,1]
    y=sat_soln[:,2]
    vy=sat_soln[:,3]
    Vesc= np.sqrt(2*G*Mmars/np.sqrt(initial[0]**2 + initial[2]**2))
    
    if i > Vesc:
        V0.append(i)

# Closest apprach       
        #array of all the displacements of a trajectory
        r= np.array([np.sqrt(x**2 + y**2)])          
        #add the shortest displacements into closest approach list 
        rmin.append(r.min())        
        if i == 10000:
            plot_closest_approach('Stationary Mars vs Moving Mars- Closest approach',1)          
# Angular deviation 
        phi=np.degrees(sp.arctan(vx[-1]/vy[-1]))        
        if phi < 0:
            phi= 180-np.absolute(phi)        
        Deflection.append(phi) 
        
        if i == 10000:
            plot_angular_deviation('Stationary Mars vs Moving Mars- Angular deviation',1)
          
V0=[]
Deflection=[]
rmin=[]

'''''''''''''''''''''''''''
Scenario 2: Mars in motion
'''''''''''''''''''''''''''
for i in range(1000,10100,100):
    initial= [-20*Rmars,24100 ,-20*Rmars,i]
    sat_soln= odeint(sat_motion_1, initial, t)
    x= sat_soln[:,0]
    vx= sat_soln[:,1]-Vmars
    y= sat_soln[:,2]
    vy= sat_soln[:,3]
    Vesc= np.sqrt((2*G*Mmars/np.sqrt(initial[0]**2 + initial[2]**2))+Vmars**2)
    Vsat= np.sqrt(i**2+Vmars**2)
    if Vsat > Vesc:
        V0.append(Vsat)

# Closest apprach
        #array of all the displacements of a trajectory 
        r= np.array([np.sqrt(x**2 + y**2)]) 
        rmin.append(r.min())
        if i == 10000:
            plot_closest_approach('Stationary Mars vs Moving Mars- Closest approach',2)
# Angular deviation 
        phi=np.degrees(sp.arctan(vx[-1]/vy[-1]))        
        if phi < 0:
            phi= 180-np.absolute(phi)        
        Deflection.append(phi) 
        if i == 10000:
            plot_angular_deviation('Stationary Mars vs Moving Mars- Angular deviation',2)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Task 3: Energy of the System:
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def plot_system_energy(fig,n):
    r = np.sqrt((x)**2 + y**2)
    KEsat = 0.5*Msat*(vx**2+vy**2)
    PEsat = -G*Mmars*Msat/np.sqrt(x**2 + y**2)
    TEsat = KEsat + PEsat
               
#plot trajectory      
    plt.figure(fig) 
    plt.subplot(1,2,1)    
    plot_xy_graph(fig,1)  
    if n == 1:
        plt.axis('scaled')   
    plt.legend() 
    leg = plt.legend(loc=9, ncol=4, prop={'size':11},)
    leg.get_frame().set_alpha(0.4)  

#plot energy graph 
    plt.subplot(1,2,2)        
    plt.title(fig)         
    plt.xlabel("time (s)")
    plt.ylabel("Energy (J)")
    
#Stationary Mars: plot KE, PE, TE as a function of time:         
    if n == 1:
        plt.plot(t,KEsat, 'b-', label='KEsat')            
        plt.plot(t,PEsat, 'r-', label='PEsat') 
        plt.plot(t, TEsat, 'g--', label= 'TEsat')
        plt.legend()
        leg = plt.legend(loc=9, ncol=3, prop={'size':11},)
        leg.get_frame().set_alpha(0.4)
    
# Moving Mars: plot KE, PE, TE as a function of the trajectory 
    if n == 2: 
        plt.plot(r, TEsat, 'g--', linewidth= 3, label= 'TEsat' )
        plt.plot(r,PEsat, 'r-', label='PEsat')
        plt.plot(r,KEsat, 'b-', label='KEsat')
        plt.legend()
    plt.grid(True) 
    plt.show()

'''''''''''''''''''''''''''
Scenario 1: Mars at rest
'''''''''''''''''''''''''''   
for j in range(500,1210,10):
    initial= [-20*Rmars,0,0,j] # Orbital height of MRO
    sat_soln = odeint(sat_motion, initial, t)
    x=sat_soln[:,0]
    vx=sat_soln[:,1]
    y=sat_soln[:,2]
    vy=sat_soln[:,3]    

# Case 1: Circular orbit 
    if j == 790:
        plot_system_energy('Circular orbit Energy', 1)

# Case 2: Elongated orbit 
    if j == 1070:
        plot_system_energy('Elliptical orbit Energy', 1)

# Case 3: escape orbit
    if j == 1200:
        plot_system_energy('Escape orbit Energy', 1)

'''''''''''''''''''''''''''
Scenario 2: Mars in motion
'''''''''''''''''''''''''''
#satellite pass behind mars 
for i in range(2):
    if i == 0:
        initial= [-50*Rmars,24100 ,-50*Rmars,600]
    else: 
        initial= [-50*Rmars, 24700 ,-50*Rmars, 0]
    sat_soln= odeint(sat_motion_1, initial, t)
    x= sat_soln[:,0]
    vx= sat_soln[:,1]
    y= sat_soln[:,2]
    vy= sat_soln[:,3]
    if i == 0:
        plot_system_energy('Passing behind Mars Energy',2)
    else:
        plot_system_energy('Passing in front of Mars Energy',2)
from __future__ import absolute_import, division, print_function
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 12:13:26 2016

@author: ikc15
"""

# -*- coding: utf-8 -*-
"""
Animation of 2-Dimensional Gas Particles In a Circular Container 
Created on Wed Nov 02 18:27:19 2016

@author: ikc15
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.stats as stats
import math

print('!!!!!!!!!!!! Just press Enter for default settings!!!!!!!!!!!!')
Animate = input('Animation (True/False)? ') or 'True'
kB = 1.3806E-23 #Boltzmann Constant
u = 1.66E-27

       
class Ball:
    
    """
    A Ball class which assigns mass,radius, position and velocity to each ball object. 
    """
    number_of_balls = 0
    def __init__(self, m=1, R = 1, r=[0.0,0.0], v=[0.0,0.0]):

        self.__pos = np.array(r)
        self.vel = np.array(v)
        self.mass = m
        self.size = R
        self.impulse = [0]
         
        Ball.number_of_balls += 1
    
    def pos(self):
        """returns the position of the ball."""
        return self.__pos
    
    def setPos(self, r):
        """sets the position of the ball."""
        self.__pos = np.array(r)
        
    def __str__(self):
        """returns the informal representation of the ball"""
        return str((self.mass, self.size, str(self.__pos), str(self.vel)) )
        
    def ke(self):
        """returns the kinetic energy of the ball"""
        return 0.5*self.mass*np.dot(self.vel,self.vel)
        
    def angularMomentum(self):
        """returns the angular momentum of the ball"""
        return self.mass*np.cross(self.pos(),self.vel)
        
    def move(self, dt):
        """moves the ball foward by dt seconds and updates its position."""
        r = self.pos() + (self.vel * dt)
        
        # updates position
        self.__pos = r
        return self.__pos
        
    def time_to_collision(self,other):
        """finds the time to the next collision for this ball with another ball or container."""
        time_to_collision = []        
        
        #relative velocity vector
        dv = self.vel  - other.vel
        
        #relative position vector
        dr = self.pos() - other.pos()
        
        #solving the quadratic equation (r1+v1δt−r2−v2δt)**2 = (R1±R2)**2 for δt:
        a = np.dot(dv,dv)
        b = 2*np.dot(dr,dv)
        c = np.dot(dr,dr)-(self.size + other.size)*(self.size + other.size)
        discriminant = b*b - 4*a*c
        if discriminant > 0:
            
            dt1 = (-b + np.sqrt(b*b - 4*a*c))/(2*a)
            dt2 = (-b - np.sqrt(b*b - 4*a*c))/(2*a)
            
            if dt1 < 0 and dt2 < 0:
                time_to_collision.append(100)
            elif dt1 > 0 and dt2 < 0:
                time_to_collision.append(dt1)
            elif dt1 < 0 and dt2 > 0:
                time_to_collision.append(dt2)
            elif dt1 > 0 and dt2 > 0:
                time_to_collision.append(min (dt1,dt2)) 
                
        else:
            
            time_to_collision.append(100)
        
        # returns minimum time to next collision for this ball with the other ball
        min_col_time = min(time_to_collision)
        return min_col_time        
    
    def collide_container(self):
        """updates the velocitiy of the ball after collision with container"""
        r1 = self.pos() 
        v1 = self.vel
        
        #unit position vector 
        r1hat = r1/np.sqrt(np.dot(r1, r1))

        KE_before = self.ke() 
        
        #the new velocity vector of the ball after collision 
        v1_after = v1 -2*np.dot(v1, r1)/np.dot(r1, r1) *r1

        #updates velocity 
        self.vel = v1_after
        
        KE_after = self.ke()
        
        #check for conservation of kinetic energy after collision
        assert KE_before - KE_after < 0.001, "KE not conserved!" 
        
        # calculates the impulse exerted on the container and add to list defined under  __init__.
        self.impulse.append(2 * self.mass * np.dot(self.vel,r1hat))
        
    
    def collide_ball(self, other):
        """updates the velocitiy of the ball after collision with another ball"""
        r1 = self.pos()
        r2 = other.pos()
        v1 = self.vel
        v2 = other.vel        
        
        #mass of particles in collision 
        mT = (self.mass + other.mass)
        mRatio1 = 2*other.mass/mT
        mRatio2 = 2*self.mass/mT
        
        #relatvie position vector        
        dr1 = r1 - r2
        dr2 = -dr1
        
        #relative velocity vectors
        dv1 = v1 - v2 
        dv2 = -dv1
        
        KE_before = self.ke() + other.ke()  
        
        #the new velocity vectors of each ball after collision
        v1_after = v1 - (mRatio1*np.dot(dv1, dr1)/np.dot(dr1, dr1)*dr1)
        v2_after = v2 - (mRatio2*np.dot(dv2, dr2)/np.dot(dr2, dr2)*dr2)
        
        #updates velocities 
        self.vel = v1_after
        other.vel = v2_after
        
        KE_after = self.ke() + other.ke()  

        #check for conservation of kinetic energy after collision
        assert KE_before - KE_after < 0.001, "KE not conserved!" 

def randomList(n):
    """
    Contains a list of N balls distributed randomly.    
    Random veolocities from a gaussian distribution (with mean = 0 and 
    std deviation = 1000) assigned to each ball.
    """

    # balls instantiated with mass of He atoms
    ballList = [Ball(4*u,1, [0,0], 
                [np.random.normal(0,800,1)[0], np.random.normal(0,800,1)[0]]) 
                for i in range(int(n))]
    
    #container created    
    container = Ball(m=10, R = -float(input('Enter radius of container: ') or 50),r=[0,0], v=[0,0]) 
    
    #generate random position for balls
    for ball in ballList:
        check = 0
        counter = 0
        r = (abs(container.size) - 1)*np.random.random()
        theta = 2*np.pi*np.random.random()
        ball.setPos([r*np.cos(theta),r*np.sin(theta)])

        while check == 0:
            distance =[]
            for other in ballList:
                if other is not ball:
                    #print 'helooooooooooooooo other',other.pos()
                    dr = ball.pos() - other.pos()
                    #print 'this is seperation vector', dr
                    distance_apart = np.sqrt(np.dot(dr,dr)) - (ball.size + other.size)
                    distance.append(distance_apart)
            if all(i > 0 for i in distance):
                check = 1
            else:
                counter += 1
                #print counter
                assert counter < 1000, "Too many balls!!" 
                theta = 2*np.pi*np.random.random()
                r = (abs(container.size) - 1)*np.random.random()
                ball.setPos([r*np.cos(theta),r*np.sin(theta)])

    # adds container to the end of ballList
    ballList.append(container)
    return ballList


class Gas:
    """
    A Gas class which simulates the motion of the balls created in Ball class.
    Gas objects collect data on Pressure, Temperature, Kinetic Energy and 
    Angular Momentum.   
    """
    #lists for the positions and pathces of the balls
    positions = []
    patches = []

    #ideal time-step per frame
    ideal_dt = 0.0001 
    
    # actual time-step per frame
    dt = 0  
    
    # counter used to check whether frames elasped = frames to collision. 
    frames_to_collision = 0
    counter = 0
    
    # balls invovled in the next collision 
    ball1 = None
    ball2 = None
    
    time_elapsed = 0
    frames = 0
    frames_wanted = 0
    
    #useful data
    Pressure = []
    Temp = []
    P_T = []
    KE = []
    angularMomentum = []
    time = []
    vel = []

    #Generate a list of balls for the Gas
    numberOfBalls = int(input('Enter number of balls: ') or 100)
    balls = randomList(numberOfBalls) 
    container = balls[len(balls)-1]
    
    def __init__(self):
        """Initialise the Gas class for N balls defined in ballList."""
        
        # defines patches that represent the balls in ballList
        i = 0
        while i < (len(Gas.balls)-1):
            #generate random colour
            de, re, we, ge = ("%02x"%np.random.randint(0,255)),("%02x"%np.random.randint(0,255)),("%02x"%np.random.randint(0,255)), "#"
            color = ge + de + re + we
            
            self.__i = Gas.balls[i].pos()
            Gas.positions.append(self.__i)
            
            self.__i = plt.Circle((Gas.balls[i].pos()[0],Gas.balls[i].pos()[1]), Gas.balls[i].size, fc= color)
            Gas.patches.append(self.__i)
            i += 1

        self.__text0 = None
        self.__text1 = None
        
    def plots(self,figure, title, xlabel,ylabel,xdata, ydata):
        """plots data obtained by function simulatePhysics()."""
        plt.figure(figure)    
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(xdata, ydata)
        plt.show()
    
    def next_collision(self):
        next_collision_time = 1000
        
        for i in range(len(Gas.balls)-1):
            ball = Gas.balls[i]

            for j in range(i+1, len(Gas.balls)):
                other = Gas.balls[j]

                t = ball.time_to_collision(other)
        
                if t < next_collision_time:
                    # set next collision time to t
                    next_collision_time = t
                    # records the balls involved
                    Gas.ball1 = ball
                    Gas.ball2 = other   
                    # calculates the number of frames to next collision. 
                    Gas.frames_to_collision = math.ceil(next_collision_time / Gas.ideal_dt)
                    # sets value for dt, the time-step per frame
                    Gas.dt = next_collision_time / Gas.frames_to_collision
                    Gas.counter = 0
       
        
    def init_figure(self):
        """initialises the animation."""
        r = Gas.container.size
        # adds contianer onto the animation screen         
        BigCirc = plt.Circle((0,0), abs(Gas.container.size), ec = 'b', fill = False, ls = 'solid')
        ax.add_artist(BigCirc)
        
        # adds balls onto the animation screen 
        i = 0
        while i < (len(Gas.balls)-1):
            ax.add_patch(Gas.patches[i])
            i += 1
        
        # adds frame number and time elapsed on the animation screen
        self.__text0 = ax.text(r-r/10,abs(r)+r/5,"f={:4d}".format(0,fontsize=12))
        self.__text1 = ax.text(r-r/10,abs(r)+r/10, "t={:4g}".format(0,fontsize=12))
        
        # Determines the minimum time to next collision
        self.next_collision()
   
        return (self.__text0, self.__text1)
    
    def next_frame(self, i):
        """ Sets up the next frame of the animation"""
        
        # 'i' is the frame number.
        Gas.counter += 1
        
        # Check for collision. If frames elasped = frames to next collision, collide balls.
        if Gas.counter == Gas.frames_to_collision:
            if Gas.ball2.size < 0:
                Gas.ball1.collide_container()
            else:
                Gas.ball1.collide_ball(Gas.ball2)
                
            # Re-determine the minimum time to next collision and balls involved
            self.next_collision()
    
        else:
            Gas.time_elapsed += Gas.dt
            p = 0
            while p < (len(Gas.balls)-1):
                Gas.positions[p] = Gas.balls[p].move(Gas.dt)
                Gas.patches[p].center = (Gas.positions[p][0],Gas.positions[p][1])
                p += 1
            self.__text0.set_text("f={:4d}".format(i))
            self.__text1.set_text("t={:4g}".format(Gas.time_elapsed))
        
        Gas.frames = i
        if Gas.frames == Gas.frames_wanted and Animate is True:
            self.collectData()     
            anim._stop()
            
        else:
            self.collectData()        
        
        return (self.__text0, self.__text1,Gas.patches)
    
    def collectData(self):
        #Collecting Data
            totalImpulse = 0
            KE = 0
            angularMomentum = 0
            for ball in Gas.balls:
                if ball is not Gas.balls[len(Gas.balls)-1]:
                    KE += ball.ke()
                    totalImpulse += sum(ball.impulse)
                    Gas.vel.append(np.sqrt(np.dot(ball.vel,ball.vel)))
                    angularMomentum += ball.angularMomentum()
                     
            print ('Time = ', ('%.3f' % Gas.time_elapsed), 's')
            print ('Frame = ', Gas.frames)
            print ('Kinetic Energy = ', KE, 'J')
            print ('Augular Momentum = ', angularMomentum, 'kg m^2/s')
                
            T = 2.*KE / (2.* kB * len(Gas.balls))
            print ('Temperature = ', T, 'K')
                
            Pressure = totalImpulse / ((Gas.time_elapsed+1e-100) * 2*np.pi*Gas.container.size)
            print ('Avg Pressure = ', Pressure, 'Pa m')
            
            print ('P/T ratio = ', Pressure/T, 'Pa.m/K')
            print (' ')
                
            Gas.Pressure.append(Pressure)
            Gas.Temp.append(T)
            Gas.time.append(Gas.time_elapsed)
            Gas.KE.append(KE)
            Gas.angularMomentum.append(angularMomentum)
            Gas.P_T.append(Pressure/T)
            
            if Gas.frames == Gas.frames_wanted:
                
                print ('Most Probable Speed= ', np.sqrt(kB*T/(4*u)), 'm/s')
                
                self.plots('KE vs time','The kinetic energy of the system over time',"time/s","KE/J",Gas.time, Gas.KE)
                self.plots('Pressure vs time', 'The Pressure of the system over time',"time/s", "2-D Pressure/N/m", Gas.time, Gas.Pressure)
                self.plots('Temperature vs time', 'The Temperature of the system over time',"time/s", "Temperature/K", Gas.time, Gas.Temp)
                self.plots('P/T vs time', 'The ratio of Pressure and Temperature over time',"time/s", "Pressure/Temperature/Pa.m/K", Gas.time, Gas.Temp)
                self.plots('Augular Momentum vs time', 'The augular momentum of the system over time',"time/s", "Angular momentum/kg.m^2/s", Gas.time, Gas.angularMomentum)

                #Maxwell distribution and histogram 
                plt.figure("Histogram for velocity distribution of particles ")
                plt.title('Maxwell-Boltzmann distribution of particle velocities')
                plt.xlabel("velocity/m/s")
                plt.ylabel("p(v)")
                maxwell = stats.maxwell
                data = Gas.vel
                params = maxwell.fit(data, floc=0)
                plt.hist(data, bins='auto', normed=True, label='Histogram of velocities')
                v = np.linspace(0, max(Gas.vel), len(Gas.time))
                plt.plot(v, maxwell.pdf(v, *params), lw=3, label='Python MB-pdf')
                mean, var = maxwell.stats(moments='mv')
                plt.plot(v, (4*u/(kB*T))*v*np.exp(-0.5*4*u*v*v/(kB*T)), lw=3, label='Theoretical 2-D MB-distribution p(v) = (m/kBT)*v*exp(-mv^2/2kBT)')
                plt.legend(loc='best', frameon=False)
                plt.show()
        
        
    def simulatePhysics(self,frames_wanted=10000):
        """
        Collects all the data required to plots graph and prints time, kinetic 
        energy(KE), angular momentum, temperature(T), pressure(P) and P/T ratio 
        in real-time. 
        """
        Gas.frames_wanted = frames_wanted
        self.init_figure()
        Gas.frames = 0
        while Gas.frames < frames_wanted:
            Gas.frames += 1
            self.next_frame(Gas.frames)

            
if __name__ == '__main__':

    movie = Gas()
    if Animate == 'True':
        frames_wanted = int(input('Enter number of frames to simulate: ') or 1000)
        Gas.frames_wanted = frames_wanted
        fig1 = plt.figure('Animation')
        ax = plt.axes(xlim=(Gas.container.size, abs(Gas.container.size)), ylim=(Gas.container.size, abs(Gas.container.size)))
        ax.axes.set_aspect('equal')  
        anim = animation.FuncAnimation( fig1, 
                                        movie.next_frame, 
                                        init_func = movie.init_figure, 
                                        interval = 0.000001, # no. of ms pause/ frame,
                                        blit = False)
        plt.show()
        
    else: 
        ax = plt.axes(xlim=(Gas.container.size, abs(Gas.container.size)), ylim=(Gas.container.size, abs(Gas.container.size)))
        ax.axes.set_aspect('equal')          
        movie.simulatePhysics(int(input('Enter number of frames to simulate: ')))
        
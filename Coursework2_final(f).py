
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 23:30:44 2021

@author: xianli
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy import optimize

pi = np.pi
cos = np.cos
sin = np.sin

# parameters which are constant throughout

R = 10  #head is a sphere with r =10cm
L = 4   #length of hair
smin = 0    #Arclength to determine computational span value
smax = L    ##Arclength to determine computational span value
Num_iters = 600 #total number of steps/resolution
s = np.linspace(smin, smax, Num_iters)  #Setup the array of points on grid to integrate IVP.

def hair2D(L, R, fg, fx, theta0, guess1, guess2):
    """
    The function takes in length of hair, radius of head, gravity, wind force and array of hair latitude.
    Uses an array of initial guess z0 range from linspace(-5,5,500) to integrate the IVP.The solution 
    obtained then uses a scipy.optimize.root(return roots of non-linear eqn) to find the correct z0 that 
    correspond to the set boundary conditions. IVP is integrated again using the new initial values,
    [theta0, root] where root is the solution optimized from fsolve to calculate the final solution for 
    theta of the BVP.
    
    Parameters
    ---------------
    L: float.
        The value of the length of hair in cm
 
        wanted
    fg: float.
        The value of the force acting upon the hair due to gravity. In units of
        cm ** -3.
    fx: float.
        The value of the forced relative to the bending stiffness of the hair acting in the
        positive x-direction. 
    theta0: 1 x n array of floats.
        The initial theta0, lattitude angle of each hair determined by n number of hairs.
    guess1: float.
        Guess value for z0 found by referring the test_hair2D for right hair
    guess2: float.
        Guess value for z0 found by referring the test_hair2D for left hair

    Returns
    --------------
    x: list of arrays,
        The x-coordinate of the hair strands in a list and value of each strand is stored in an array
 
    z: list of arrays,
        The z-coordinate of the hair strands in a list and value of each strand is stored in an array
    """
    #   Assert block to verify input for length of hair and radius of head
    assert(np.isscalar(R) and R>0 and np.isreal(R) and np.isfinite(R))
    assert(np.isscalar(L) and L>0 and np.isreal(L) and np.isfinite(L))
    #  Assert block to verify input for fx and fg
    assert(np.isreal(R) and np.isfinite(R))
    assert(np.isreal(R) and np.isfinite(R))
    assert type(theta0) == np.ndarray
    assert theta0.size >= 3
    #   Assert block to verify input for guess1 and guess2
    assert(np.isscalar(guess1) and np.isreal(guess1) and np.isfinite(guess1))
    assert(np.isscalar(guess2) and np.isreal(guess2) and np.isfinite(guess2))
    
    sspan = (s[0], s[-1]) #create xspan for ivp solving
    x=[] #empty list of x-coordinates
    z=[] #empty list of y-coordinates

    for i in range(20): #interate through 20 hair to get x.z coordinate for each hair
        
        if i < 10: #represent right side of the hair
            guess = guess1 #guess value for fsolve
        else: #represent left side of the hair
            guess = guess2 #guess value for fsolve
            
        t0= theta0[i] #setting latitude where the hair grows from the head 
        
        phi = 0  #2D solution only has 1 plane hence phi = 0
        p0 = phi
        
        def ft(s,q):
            """
            Define the IVP q'(s). This is the IVP which will be integrated to find
            the solution theta(s).
    
            Parameters
            ----------
            q: 1 x 2 array of floats.
                The vector of initial values used to integrate the IVP.
            s: 1 x n_points array of floats.
                An array containing the grid positions used to integrate the IVP.
            Returns
            -------
            dtds: floats,
                dtheta/ds
            dzds: floats,
                dtheta/ds
            """
            t = q[0]
            z = q[1]
            dtds = z
            dzds = s*fg*cos(t)+s*fx*sin(t)*cos(p0)
            return [dtds, dzds]
    
        
        @np.vectorize
        def dz(z0):
            """
            Define the function dz/ds which takes in an array of intial guess of z0
            and creates a residual function. Using rooting finding methods on this
            function will return an appropriate value of z to use in the intial
            guess when integrating the ft(s) to find theta(s).

            Parameters
            ----------
            z0: array.
            The values of the initial guess for calculating theta prime 
            Returns
            -------
            z: float.
            The value of theta prime calculated for the current value of z0. Use a
            root finding method to find the value of z such that dz(z) = 0 to find
            the correct intial condition for theta prime to solve the BVP.
            """
            y0 = np.array([t0,z0])
            
            sol = solve_ivp(ft, sspan, y0, t_eval=s)
            t, z = sol.y
            
            return z[-1]

        #an array of z guesses
        z0=np.linspace(-5,5,500)

         # guess based on that inputs guess1 and guess2 where both guesses were found
         # by referring to the plots in test_hair2D plots
        sol0 = optimize.root(dz, guess) #optimize for find root
        root1 = sol0.x
        
        B0 = np.array([t0, root1],dtype=object) #initial conditions
        #solve ivp
        sol = solve_ivp(ft, sspan, B0, t_eval = s)
        the= sol.y[0]
        
#        plt.plot(s, t, 'k')
#        plt.show()
#        plt.plot(s, z, 'b')
#        plt.show()

        ##fit to find theta 
        thet= np.polyfit(s, the, 4)
#        print(thet)

        #solve for x, and z
        x0 = [R*cos(t0)*cos(p0)]
        z0 = [R*sin(t0)]

        def X(s,x):
            theta = thet[0]*s**4+thet[1]*s**3+thet[2]*s**2+thet[3]*s+thet[4]
#            phi = ph[0]*s**4+ph[1]*s**3+ph[2]*s**2+ph[3]*s+ph[4]
            phi = 0
            dxds = cos(theta)*cos(phi)
            return dxds
        
        #solve ode for x coordinate
        sol1 = solve_ivp(X, sspan, x0, t_eval = s)
#        print(sol.y)
        x.append(sol1.y[0])
        
        def Z(s,z):
            theta = thet[0]*s**4+thet[1]*s**3+thet[2]*s**2+thet[3]*s+thet[4]

            dzds = sin(theta)
            return dzds
        
        #solve ode  for z coordinate
        sol3 = solve_ivp(Z, sspan, z0, t_eval = s)
        # print(sol.y)
        z.append(sol3.y[0])

    return x, z

def test_hair2D(L, R, fg, fx, theta0):
    """
    The function works exactly like hair2D but produce plots of the dz/ds for different values of z0
    to help in estimating the initial guess for root optimizing function. Also produce the plots to verify 
    solution satisfy boundary conditions.
    
    Parameters
    ---------------
    L: float.
        The value of the length of hair in cm
 
        wanted
    fg: float.
        The value of the force acting upon the hair due to gravity. In units of
        cm ** -3.
    fx: float.
        The value of the forced relative to the bending stiffness of the hair acting in the
        positive x-direction. 
    theta0: 1 x n array of floats.
        The initial theta0, lattitude angle of each hair determined by n number of hairs.

    Returns
    --------------
    20 plots of dz/ds and the final dz/ds solution that meets the boundary condition
    """
    #   Assert block to verify input for length of hair and radius of head
    assert(np.isscalar(R) and R>0 and np.isreal(R) and np.isfinite(R))
    assert(np.isscalar(L) and L>0 and np.isreal(L) and np.isfinite(L))
    #  Assert block to verify input for fx and fg
    assert(np.isreal(R) and np.isfinite(R))
    assert(np.isreal(R) and np.isfinite(R))
    assert type(theta0) == np.ndarray
    assert theta0.size >= 3
    
    sspan = (s[0], s[-1]) #create xspan for ivp solving
    
    # x=[] #empty list of x-coordinates
    # z=[] #empty list of y-coordinates

    for i in range(20): #interate through 20 hair to get x.z coordinate for each hair
        print('theta',i+1)

        t0= theta0[i] #setting latitude where the hair grows from the head 
        p0= 0 #2D solution only has 1 plane hence phi = 0
        
        def ft(s,q):
            """
            Define the IVP q'(s). This is the IVP which will be integrated to find
            the solution theta(s).
    
            Parameters
            ----------
            q: 1 x 2 array of floats.
                The vector of initial values used to integrate the IVP.
            s: 1 x n_points array of floats.
                An array containing the grid positions used to integrate the IVP.
            Returns
            -------
            dtds: floats,
                dtheta/ds
            dzds: floats,
                dtheta/ds
            """
            t = q[0]
            z = q[1]
            dtds = z
            dzds = s*fg*cos(t)+s*fx*sin(t)*cos(p0)
            return [dtds, dzds]
        
        @np.vectorize
        def dz(z0):
            """
            Define the function dz/ds which takes in an array of intial guess of z0
            and creates a residual function. Using rooting finding methods on this
            function will return an appropriate value of z to use in the intial
            guess when integrating the ft(s) to find theta(s).

            Parameters
            ----------
            z0: array.
            The values of the initial guess for calculating theta prime 
            Returns
            -------
            z: float.
            The value of theta prime calculated for the current value of z0. Use a
            root finding method to find the value of z such that dz(z) = 0 to find
            the correct intial condition for theta prime to solve the BVP.
            """
            y0 = np.array([t0,z0])
            
            sol = solve_ivp(ft, sspan, y0, t_eval=s)
            t, z = sol.y
            
            return z[-1]
        #an array of z guesses
        z0=np.linspace(-5,5,500)
        
        plt.plot(z0, dz(z0), 'b')
        plt.axhline(c = 'k')
        plt.grid()
        plt.title('dtheta/ds at s = 4 values')
        plt.xlim(-4,4)
        plt.ylim(-0.1,0.1)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$dz/ds$')
        plt.show()


def head2DXZ(theta0):
    """
        Compute the hair positions on the head from range of values from theta0
        
        Parmeters
        ---------
        theta0: 1 x n array of floats.
        The initial theta0, lattitude angle of each hair determined by n number of hairs.
        
        Returns
        -------
        x: list,
        The x-coordinate of the hair position on head
 
        z: list,
        The z-coordinate of the hair position on head
    """
    assert type(theta0) == np.ndarray
    assert theta0.size >= 3
    
    X = []
    Z = []
    X.append(R*cos(theta0))
    Z.append(R*sin(theta0))
    return X, Z

#definining the derivative of the governing function
#definie shooting ivp

def hair3D(L, R, fg, fx, theta0, phi0, guess1, guess2):
    """
    The function takes in length of hair, radius of head, gravity, wind force and array of hair latitude
    and hair longitude. Uses an array of initial guess z0 range from linspace(-5,5,500) to integrate the 
    IVP.The solution obtained then uses a scipy.optimize.root(return roots of non-linear eqn) to find the 
    correct z0 and h0 that correspond to the set boundary conditions. IVP is integrated again using the new initial 
    values, [theta0, root] where root is the solution optimized from fsolve to calculate the final solution 
    for theta of the BVP.
    
    Parameters
    ---------------
    L: float.
        The value of the length of hair in cm
 
        wanted
    fg: float.
        The value of the force acting upon the hair due to gravity. In units of
        cm ** -3.
    fx: float.
        The value of the forced relative to the bending stiffness of the hair acting in the
        positive x-direction. 
    theta0: 1 x n array of floats.
        The initial theta0, lattitude angle of each hair determined by n number of hairs.
    phi0: 1 x n array of floats.
        The initial phi0, longitude angle of each hair determined by n number of hairs.
    guess1: float.
        Guess value for z0 found by referring the test_hair3D
    guess2: float.
        Guess value for h0 found by referring the test_hair3D

    Returns
    --------------
    x: list of arrays,
        The x-coordinate of the hair strands in a list and value of each strand is stored in an array
 
    z: list of arrays,
        The z-coordinate of the hair strands in a list and value of each strand is stored in an array
    """
    #   Assert block to verify input for length of hair and radius of head
    assert(np.isscalar(R) and R>0 and np.isreal(R) and np.isfinite(R))
    assert(np.isscalar(L) and L>0 and np.isreal(L) and np.isfinite(L))
    #  Assert block to verify input for fx and fg
    assert(np.isreal(R) and np.isfinite(R))
    assert(np.isreal(R) and np.isfinite(R))
    assert type(theta0) == np.ndarray
    assert theta0.size >= 3
#    Assert block to verify input for guess1 and guess2
    assert(np.isscalar(guess1) and np.isreal(guess1) and np.isfinite(guess1))
    assert(np.isscalar(guess2) and np.isreal(guess2) and np.isfinite(guess2))
    sspan = (s[0], s[-1])
    A=[]
    B=[]
    C=[]
    #iterate through all 10x10 phi0 and theta0
    for j in range(10):
        
        p0 = phi0[j]
        if j < 5:
            guess2 = 0
        elif j ==5:
            guess2 = 0.1
        elif j ==6:
            guess2 = 0.1
        else:
            guess2 = 0.6
        
        
        for i in range(10):
            
#         
            if j < 7:
                guess1 = -0.6
            elif j ==5:
                guess1 = -0.7
            else:
                guess1 = -0.6            
            # guess1 = -0.7
                
            t0= theta0[i]

            def ft(s,q):
                 t = q[0]
                 z = q[1]
                 p = q[2]
                 h = q[3]
                
                 dtds = z
                 dzds = s*fg*cos(t)+s*fx*cos(p)*sin(t)
                 dpds = h 
                 dhds = -s*fx*sin(t)*sin(p)
                 return [dtds, dzds, dpds, dhds]

            
            def dp(z0):
          #    y0 = np.array([t0, q0, p0, q1])
              y0 = np.array([t0, z0[0], p0, z0[1]])
          #    h = odeint(ft, y0, s)
              sol = solve_ivp(ft, sspan, y0, t_eval=s)
              z = np.zeros(2)
              # t = sol.y[0]
              z[0] = sol.y[1,-1]
              # p = sol.y[2]
              z[1] = sol.y[3,-1]
              return z

      
            root1 = optimize.root(dp, [guess1,guess2])
      
            root = np.zeros(2)
            root[0]= root1.x[0]
      
            root[1]= root1.x[1]


#           
            B1 = np.array([t0, root[0], p0, root[1]])
            sol = solve_ivp(ft, sspan, B1, t_eval = s)
            tf = sol.y[0]
            # zf = sol.y[1]
            pf = sol.y[2]
            # hf = sol.y[3]
            
            #fit to get a representative equation for theta
            thet= np.polyfit(s, tf, 4)
            
            #  #fit to get a representative equation for phi       
            ph= np.polyfit(s, pf, 4)
            
            
                 
            #solve ODE for x, y and z
            X0 = [R*cos(t0)*cos(p0)]
            Y0 = [-R*cos(t0)*sin(p0)]
            Z0 = [R*sin(t0)]
            #
            def X(s,x):
                theta = thet[0]*s**4+thet[1]*s**3+thet[2]*s**2+thet[3]*s+thet[4]
                phi = ph[0]*s**4+ph[1]*s**3+ph[2]*s**2+ph[3]*s+ph[4]
                dxds = cos(theta)*cos(phi)
                return dxds
            
            #solve ode 
            sol1 = solve_ivp(X, sspan, X0, t_eval = s)

            A.append(sol1.y[0])
            
            
            def Y(s,y):
                theta = thet[0]*s**4+thet[1]*s**3+thet[2]*s**2+thet[3]*s+thet[4]
                phi = ph[0]*s**4+ph[1]*s**3+ph[2]*s**2+ph[3]*s+ph[4]
                dyds = -cos(theta)*sin(phi)
                
                return dyds
            
            #solve ode 
            sol2 = solve_ivp(Y, sspan, Y0, t_eval = s)

            B.append(sol2.y[0])
            
            
            def Z(s,z):
                theta = thet[0]*s**4+thet[1]*s**3+thet[2]*s**2+thet[3]*s+thet[4]
            
                dzds = sin(theta)

                return dzds
            
            #solve ode 
            sol3 = solve_ivp(Z, sspan, Z0, t_eval = s)

            C.append(sol3.y[0])
            
    return A,B,C



#solve for hair root positiios in 3D
def headXYZ(theta0,phi0):
    """
    Compute the 3D hair positions on the head from range of values from theta0 and phi0
    
    Parmeters
    ---------
    theta0: 1 x n array of floats.
        The initial lattitude angle, theta0 of each hair determined by n number of hairs.
    phi0: 1 x n array of floats.
        The initial longitude angle,  phi0 of each hair determined by n number of hairs.
    
    Returns
    -------
    X: list,
    The x-coordinate of the hair position on head
    Y: list,
    The y-coordinate of the hair position on head
    Z: list,
    The z-coordinate of the hair position on head
    """
    assert type(theta0) == np.ndarray
    assert type(phi0) == np.ndarray
    assert theta0.size >= 3
    assert phi0.size >= 3
    
    X = []
    Y = []
    Z = []
    for i in range(10):
        for j in range(10):
            X.append(R*cos(theta0[j]) * cos(phi0[i]) )
            Y.append(-R*cos(theta0[j])*sin(phi0[i]))
            Z.append(R*sin(theta0[j]))
    return X, Y, Z


def Task2():
    """
    Task2 solves the IVP using shooting method and root finding with inputs of 
     L, R, fg, fx, theta0 , guess1, guess 2. guess1 guess2 were estimated using
     the 20 plots. 
    Returns (x,z) plots: figure 1- 20 hair in (x,z) plane, fg = 0.1, fx =0
                     
    """
    fg = 0.1
    fx = 0 #no wind
    
    theta0 = np.linspace(0,pi,20)
    #guess1 and guess2 can be seen as smart guesses with help from the 20plots which 
    #can be referred by running test_hair2D
    guess1 = -0.6
    guess2 = -1*guess1
    
    #x-z coordinates of 20 hairs
    x,z = hair2D(L, R, fg, fx, theta0, guess1, guess2)
    #iterate plot 20 hairs
    for i in range(20):
        plt.plot(x[i],z[i], 'k', linewidth = 1)
    #plot full circle
    t1 = np.linspace(0,2*pi,20)
    plt.plot(R*cos(t1), R*sin(t1))
    #plot hair-head interface
    X, Z = head2DXZ(theta0)
    plt.plot(X,Z, 'ro')
    
    plt.xlabel(r'$x$')
    plt.ylabel(r'$z$')
    plt.title('Task2 fg = 0.1, fx = 0,guess1=-0.6, guess2=0.6, without wind')
    axes = plt.gca()
    axes.set_aspect(1)
    plt.show()


def Task3():
    """
    Task3 solves the IVP using shooting method and root finding with inputs of 
     L, R, fg, fx, theta0, guess1, guess 2. guess1 guess2 were estimated using
     the 20 plots. 
    Returns (x,z) plots: figure 1- 20 hair in (x,z) plane, fg = 0.1, fx =0.1
                     
    """
    fg = 0.1
    fx = 0.1 #wind active
    theta0 = np.linspace(0,pi,20)
    #guess1 remains the same as in task 2 due to on the other side of head, hence same value
    guess1 = -0.6
    #guess2 is no longer negative of guess1 to model a more realistic hair movement in wind, 
    #where half of the hair on left hand side gets pushed down and the other half blown upwards
    guess2 = 0.52
    
#    compute x, z coordinates for all 20 hair
    x,z = hair2D(L, R, fg, fx, theta0, guess1, guess2)
#    iterate to plot 20 hair
    for i in range(20):
        plt.plot(x[i],z[i], 'k', linewidth = 1)

    #plot full circle
    t1 = np.linspace(0,2*pi,20)
    plt.plot(R*cos(t1), R*sin(t1))
    #plot hair-head interface
    X, Z = head2DXZ(theta0)
    plt.plot(X,Z, 'ro')
    
    plt.xlabel(r'$x$')
    plt.ylabel(r'$z$')
    plt.title('Task3 fg = 0.1, fx = 0.1,guess1=-0.6, guess2=0.52, with wind')
#    plt.xlabel(r'$step size log10(h)$' )
#    plt.ylabel(r'$Error, Log10()$')
    axes = plt.gca()
    axes.set_aspect(1)
    plt.show()

#

def Task5XZ():
    """
    Task5XZ solves the IVP using shooting method and root finding with inputs of 
    L, R, fg, fx, theta0, phi0 , guess1, guess 2. guess1 guess2 were estimated using
    dz/ds vs z and dh/ds vs h plots. 
    --------
    Returns (x,z) plots: figure 3- 100 hair in (x,z) plane, fg = 0.1, fx =0.05
                     
    """
    phi0 = np.linspace(0, pi, 10) #phi0 values
    theta0 = np.linspace(0,0.49*pi,10) #theta0 values
    
    #Compute the 3D hair positions and unpack into A, B, C representing x, y, z coordinates
    #change guess1 and guess2 to acheive convergence
    fg = 0.1
    fx = 0.05
    A,B,C = hair3D(L, R, fg, fx, theta0, phi0, -0.6, 0.7)
    
    for i in range(100):
        plt.plot(A[i],C[i], 'k', linewidth = 1)

    t1 = np.linspace(0,2*pi,20)
    X, Y, Z = headXYZ(theta0,phi0)
    plt.plot(X,Z, 'ro')
    plt.plot(10*cos(t1), 10*sin(t1))
    plt.title('Task5 fg = 0.1, fx = 0.05, XZ plane')
    axes = plt.gca()
    axes.set_aspect(1)
    plt.show()




def Task5YZ():
    """
    Task5YZ solves the IVP using shooting method and root finding with inputs of 
    L, R, fg, fx, theta0, phi0 , guess1, guess 2. guess1 guess2 were estimated using
    dz/ds vs z and dh/ds vs h plots. 
    --------
    Returns (y,z) plots: figure 4- 100 hair in (y,z) plane, fg = 0.1, fx =0.05
                     
    """
    fg = 0.1
    fx = 0.05
    phi0 = np.linspace(0, pi, 10) #phi0 values
    theta0 = np.linspace(0,0.49*pi,10) #theta0 values
    
    #Compute the 3D hair positions and unpack into A, B, C representing x, y, z coordinates
    #change guess1 and guess2 to acheive convergence

    A,B,C = hair3D(L, R, fg, fx, theta0, phi0, -0.6, 0.7)
    
    for i in range(100):
        plt.plot(B[i],C[i], 'k', linewidth = 1)
   
    X, Y, Z = headXYZ(theta0,phi0)
    plt.plot(Y,Z, 'ro')
    plt.plot(10*cos(t1), 10*sin(t1))
    plt.title('Task5 fg = 0.1, fx = 0.05, YZ plane')
    axes = plt.gca()
    axes.set_aspect(1)
    plt.show()
##
##



def Task5_3D():
    """
    Task5_3D solves the IVP using shooting method and root finding with inputs of 
    L, R, fg, fx, theta0, phi0 , guess1, guess 2. guess1 guess2 were estimated using
    dz/ds vs z and dh/ds vs h plots. 
    --------
    Returns 3D(x, y, z) plots: figure 5- 100 hair in 3D space with wire sphere fg = 0.1, fx =0.05
            Clearly show hair being blown to +ve x direction.          
    """
    # define theta and phi for whole sphere
    phi1 = np.linspace(0, 2*pi,20)
    theta1 = np.linspace(0,2*pi,40)
    
    phi0 = np.linspace(0, pi, 10) #phi0 values
    theta0 = np.linspace(0,0.49*pi,10) #theta0 values
    
    # Generate points to plot wire sphere
    def headline(theta1,phi1):
        X = []
        Y = []
        Z = []
        for i in range(20):
            for j in range(40):
                X.append(R*cos(theta1[j])*cos(phi1[i]))
                Y.append(-R*cos(theta1[j])*sin(phi1[i]))
                Z.append(R*sin(theta1[j]))
        return X, Y, Z
            
    x, y, z = headline(theta1,phi1)
    X, Y, Z = headXYZ(theta0,phi0)
    
    
    #Compute the 3D hair positions and unpack into A, B, C representing x, y, z coordinates
    #change guess1 and guess2 to acheive convergence
    fg = 0.1
    fx = 0.05
    A,B,C = hair3D(L, R, fg, fx, theta0, phi0, -0.6, 0.7)
    ax = plt.axes(projection='3d')
    #   add red dot 
    ax.scatter(X, Y, Z, color = "r", marker = 'o')
    #   add hair
    ax.scatter(A, B, C, color = "k", marker = '.', linewidth = 0.001)
    #add sphere
    ax.plot3D(x, y, z, color = "k", linewidth = 0.2)
    ax.view_init(elev=15, azim=-45)
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_zlim(-10, 10)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    
    plt.title('Task5 3D fg = 0.1, fx = 0.05')
    plt.show()
    
    


##change fg and fx when doing error testing
fg = 0.1
fx = 0
theta0 = np.linspace(0,pi,20)
#print(test_hair2D(L, R, fg, fx, theta0))
#print('test_hair2D produces plots that help to identify guess1 and guess2,\
#      we can see different guess value will have root converges to different value')

#theta and phi values for plotting 2D circle and 3D sphere
t1 = np.linspace(0,2*pi,20)
p1 = np.linspace(0,2*pi,20)

print("Solution to Task 2: Without wind, all hair is under influence of gravity force only")
print(Task2())
print("Solution to Task 3:With wind, we can see the chosen guess allows more realistic modeeling where half of the hair near\
 the top of the head is blown upwards and the bottom hair were pressed down due to the mixed effect of both gravity and wind")
print(Task3())
print("Solution to Task 5 in XZ plane:hair being blown to the positive x direction")
print(Task5XZ())
print("Solution to Task 5 in YZ plane:hair in front being blown into the plane but the hair behind (overlapped) is blown further\
 into the plane result in this ")
print(Task5YZ())
print("Solution to Task 5_3D:Clearly show hair being blown to +ve x direction in a 3D plot")
print(Task5_3D())
    
    

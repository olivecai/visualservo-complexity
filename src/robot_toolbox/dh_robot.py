#!/usr/bin/env python
'''
Oct 20 2025
'''

import sympy as sp
import numpy as np
import roboticstoolbox as rtb

import logging

# Configure basic logging to the console (default level is WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger for the current module
# logger. debug, info, warning, error, critical


class DHSympyParams:
    '''
    these must be initialized first so that we can use the parameters in the dh system
    '''
    def __init__(self):
        self.joint_vars = sp.symbols('t(:10)') #t1,t2,t3,...t9
        t0,t1,t2,t3,t4,t5,t6,t7,t8,t9 = self. joint_vars
        self.cart_space_vars = sp.symbols('x,y,z') #x,y,z
        x,y,z = self.cart_space_vars
        self.task_space_vars = sp.symbols('u,v') #u,v
        u, v = self.task_space_vars
        
    def get_params(self):
        return (self.joint_vars, self.cart_space_vars, self.task_space_vars)
    

class DenavitHartenbergAnalytic():
    '''
    initialize a denavit hartenberg system with analytic symbols, using sympy.
    '''
    def __init__(self, dh_params: list, symbolclass : DHSympyParams):
       
        self.cartvars = symbolclass.cart_space_vars
        self.taskvars = symbolclass.task_space_vars
        self.jntvars = symbolclass.joint_vars
        #dh_params is a double nested list, where each row is one joint.

        transforms=[]
        for param in dh_params: # for each joint 
            transforms.append(self.transformation_matrix_DH(*param)) #transformation_matrix_DH returns the DH matrix for that single joint...
        ee_matrix = sp.eye(4)
        for i in range(len(transforms)): #chain multiply the DH matrices to get the final position
            ee_matrix *= transforms[i]

        ee_translation = ee_matrix[:,3][:3] #translation final position 
        self.ee_matrix = ee_matrix #entire matrix including rotations etc
        self.ee_translation = sp.Matrix(ee_translation)
        self.dof = len(dh_params) #degree of freedom correlates to the number of dh parameter rows

        self.dh_params = dh_params
        self.rtb_robot = self.rtb_model()

        self.jointlimits = [(0, np.pi/2)] * self.dof

        #print("ee_translation:", self.ee_translation)
        self.F = sp.Matrix(self.ee_translation[:3]) - sp.Matrix(self.cartvars)
        #print("F:", self.F)
        
        self.J_analytic = sp.Matrix(self.ee_translation[:3]).jacobian(self.jntvars[:self.dof])
        self.J = (sp.utilities.lambdify(self.jntvars[:self.dof], self.J_analytic, 'numpy'))

        variables = self.jntvars[: self.dof] + self.cartvars
        self.fkin_eval = (sp.utilities.lambdify(self.jntvars[:self.dof], self.ee_translation, 'numpy'))
        self.errfn_eval= (sp.utilities.lambdify(variables, self.F, 'numpy'))

    def transformation_matrix_DH(self, theta_i, alpha_i, r_i, d_i):
        '''
        Returns the general denavit hartenberg transformation matrix for one link, i.
        theta_i is the angle about the z(i-1) axis between x(i-1) and x(i).
        alpha_i is the angle about the x(i) axis between z(i-1) and z(i).
        r_i is the distance between the origin of frame i-1 and i along the x(i) direction.
        d_i is the distance from x(i-1) to x(i) along the z(i-1) direction.
        '''
        alpha, r, theta, d = sp.symbols('alpha, r, theta, d', real=True)
        general_DH = sp.Matrix([[sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha), r*sp.cos(theta)],
                                [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), r*sp.sin(theta)],
                                [0, sp.sin(alpha), sp.cos(alpha), d],
                                [0,0,0,1]])
        #print(general_DH)
        DH = general_DH.subs([(alpha, alpha_i), (r, r_i), (theta, theta_i), (d, d_i)])
        #print(DH)
        return DH
    
    def central_differences(self, Q, desP, epsilon=None):
        '''
        Returns the central differences Jacobian and the value of epsilon to perturb by
        '''
        Q= np.array((Q))

        if epsilon == None:
            epsilon = 1e-4
        
        p= Q.shape[0]
        d= self.F.shape[0]
        #print("pxd:", p,"x",d)

        k=1
        Jt = np.zeros((p,d))
        I = np.identity(p)
    
        for i in range(p):
     
            forward = self.errfn_eval(*(Q + epsilon * I[i]) , *desP)
            #print(forward)
            backward = self.errfn_eval(*(Q - epsilon * I[i]) , *desP)
            #print(backward)
            diff = (forward-backward).T
            #print(diff)
            Jt[i] = diff / (2*epsilon)

        logger.info("Central differences Jacobian: {Jt.T}")

        return Jt.T
    
    def rtb_model(self):
        '''
        Just to verify everything is okay, I have the rtb DH robot initialized here.
        '''
        links = []

        for i in range(len(self.dh_params)):
            for j in range(4):
                try:
                    self.dh_params[i][j] = float(self.dh_params[i][j])
                except:
                    pass
            
            links.append(rtb.RevoluteDH(alpha = self.dh_params[i][1], a=self.dh_params[i][2], d=self.dh_params[i][3]))

        robot = rtb.DHRobot(links, name=f"robot_{self.dof}dof")
        logger.info("RTB robot: {robot}")

        return robot
    
    def plot(self, Q=None):
        if Q is None:
            logger.info("Q = None: Sending 0 to all joints.")
            Q = [0.0] * self.dof
        blocking = True
        logger.info(f"Plotting robot, block == {blocking}")
        self.rtb_robot.plot(Q, block=blocking)
        
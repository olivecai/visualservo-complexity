#!/usr/bin/env python
'''
October 20 2025

Uncalibrated Visual Servoing:
Import dh_robot.py and camera.py 
'''

from .dh_robot import DenavitHartenbergAnalytic
import numpy as np
import sympy as sp


import logging

# Configure basic logging to the console (default level is WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger for the current module
# logger. debug, info, warning, error, critical


class DenavitHartenberg_Cameras_Analytic():
    '''
    INVOKED BY CREATE_UVS 

    initialize a denavit hartenberg system plus camera(s) with analytic symbols, using sympy.

    Compare the lipschitz constant of with vs without cameras to see whether cameras majorly affect the complexity (hopefully they do not.)

    '''
    # it is important to get the error function F to reduce to 0
    def __init__(self, cameras: list, analytic_cameras:list, dh_robot: DenavitHartenbergAnalytic):
        '''
        cameras is a list of Camera objects
        dh_robot is a DenavitHartenbergAnalytic object
        '''
        if type(cameras) is not list:
            cameras = list(cameras)

        self.cameras =cameras
        self.dh_robot = dh_robot
        self.jntvars, self.cartvars, self.taskvars = dh_robot.jntvars, dh_robot.cartvars, dh_robot.taskvars

        logger.info(f"ANALYTIC FORWARD KINEMATICS WITHOUT CAMERA PROJECTIONS: {self.dh_robot.F}")

        self.F = []# is a factor of 2, since each camera gives two projected points.
        for camera in cameras:
            projected_point = camera.projectpoint(self.dh_robot.F)
            self.F.append(projected_point)  
        self.F = sp.Matrix(self.F)
        logger.info(f"ANALYTIC FORWARD KINEMATICS WITH CAMERA PROJECTIONS: {self.F}")

        self.F_with_cam_params = []
        for camera in analytic_cameras:
            projected_point = camera.projectpoint(self.dh_robot.F)
            self.F_with_cam_params.append(projected_point)  
        self.F_with_cam_params = sp.Matrix(self.F_with_cam_params)
        logger.info(f"ANALYTIC FORWARD KINEMATICS WITH CAMERA PARAM PROJECTIONS: {self.F_with_cam_params}")
        self.J_for_params_regression = self.F_with_cam_params.jacobian(self.dh_robot.jntvars[:self.dh_robot.dof])
        logger.info(f"ANALYTIC JACOBIAN WITH CAMERA PROJECTIONS: {self.J_for_params_regression}")


        self.J = self.F.jacobian(self.dh_robot.jntvars[:self.dh_robot.dof])
        logger.info(f"ANALYTIC JACOBIAN WITH CAMERA PROJECTIONS: {self.J}")



        variables = dh_robot.jntvars[: dh_robot.dof] + dh_robot.cartvars
        self.errfn_eval= (sp.utilities.lambdify(variables, self.F, 'numpy')) #this uses the TRUE desired point... but that's not accessible in real VS, so we should move away from this function
        self.jacobian_eval = (sp.utilities.lambdify(variables, self.J, 'numpy')) #ah, same goes for this function...
        #now self.F should be the equation of the projection, errfn_eval will evaluate F given the params


    def projected_errfn_eval(self, initQ, desPP):
        #for each camera project the end effector point and subtract the desired point
        errfn = []
        eeRP = self.dh_robot.fkin_eval(*initQ)
        eePP = self.projected_world_point(eeRP)

        errfn = np.subtract(desPP, eePP)

        return errfn
    

    def const_jac_inv_kin_pp(self, desP, initQ, J=None):
        '''
        desP is a PROJECTED POINT through the camera already!!!!!

        Q = initQ
        
        '''
        logger.info("Newton Method")

        # self.F = fkin_x - x, fkin_y - y, fkin_z - z
        currQ = initQ
        tolerance = 1e-3
        maxiter= 30

        traj = [currQ]
        
        if J is None:
            J = self.central_differences_pp(currQ, desP)
        logger.info("Jacobian:\n", J)

        errors=[]

        ret= -1

        for i in range(maxiter):
            logger.info("i:", i)
            currError = self.projected_errfn_eval(currQ, desP) #first calculate the error
            #print("currError:", currError.flatten())
            logger.info("current error:\n", currError)
            errors.append(currError)

            if np.linalg.norm(currError) <= tolerance: #if error is near 0 then we can end the iterations
                ret = i
                break
            
            Jinv =np.linalg.pinv(J) 
            logger.info("J inverse:", Jinv)
            logger.info("Norm of J Inv:", np.linalg.norm(Jinv))

            newtonStep = (Jinv @ currError).flatten()
            logger.info("newtonStep\n", newtonStep)
            currQ = currQ - self.dh_robot.alpha * newtonStep

            logger.info("currQ:\n", currQ)
            traj.append(currQ)

        traj=np.array(traj)
        if 1:
            self.dh_robot.rtb_robot.plot(traj, block=False)
            
        

        return ret, currQ
    
    
    def central_differences_pp(self, Q:list, epsilon=None):
        '''
        Pass the PROJECTED POINTS directly, NOT the real point. 

        Returns the central differences Jacobian and the value of epsilon to perturb by

        the matrix J should be (number of cameras * 2) x (dof)
        '''

        

        Q= np.array((Q))

        if epsilon == None:
            epsilon = 1e-1
        
        p= Q.shape[0]
        d= self.F.shape[0]
        #print("pxd:", p,"x",d)

        k=1
        Jt = np.zeros((p,d))
        I = np.identity(p)
    
        for i in range(p):

            
            forward = self.projected_world_point(self.dh_robot.fkin_eval(*(Q + epsilon * I[i])))
            
            backward = self.projected_world_point(self.dh_robot.fkin_eval(*(Q - epsilon * I[i])))
            

            diff = np.array((forward-backward)).T
            #print(diff)
            Jt[i] = diff / (2*epsilon)

        return -Jt.T

    def projected_world_point(self, real_world_point):
        '''
        project a real world point and get the image projection in all cameras
        '''
        #print("@@@@@@@@@@@@@@@@@@@@@@@@\n", real_world_point)
        projected_points =[]
        for camera in self.cameras:
            proj_pnt = camera.projectpoint(real_world_point)
            projected_points.append(np.array(proj_pnt, dtype=float).flatten())

        return (np.array(projected_points).flatten())
    
    
    def central_differences(self, Q, desP, epsilon=None):
        '''
        DesP is the REAL WORLD POINT.
        Returns the central differences Jacobian and the value of epsilon to perturb by

        the matrix J should be (number of cameras * 2) x (dof)
        '''
        Q= np.array((Q))

        if epsilon == None:
            epsilon = 1e-1
        
        p= Q.shape[0]
        d= self.F.shape[0]
        #print("pxd:", p,"x",d)

        k=1
        Jt = np.zeros((p,d))
        I = np.identity(p)
    
        for i in range(p):
     
            forward = self.errfn_eval(*(Q + epsilon * I[i]) , *desP)
            print("f", forward)
            backward = self.errfn_eval(*(Q - epsilon * I[i]) , *desP)
            print("b", backward)
            diff = (forward-backward).T
            #print(diff)
            Jt[i] = diff / (2*epsilon)

        return Jt.T
    
    


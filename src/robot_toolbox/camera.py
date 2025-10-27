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


class Camera:
    '''
    creates a camera with actual parameters
    '''
    def __init__(self, rot_xaxis, rot_yaxis, rot_zaxis, translation, fx, fy, cx, cy):
        '''
        rot axis is a radian rotation around a specified axis.
        '''
        self.K=sp.Matrix([[fx, 0, cx],[0, fy, cy], [0,0,1]]) #intrinsic matrix

        rx = sp.Matrix([[1,0,0],[0,sp.cos(rot_xaxis), -sp.sin(rot_xaxis)],[0,sp.sin(rot_xaxis), sp.cos(rot_xaxis)]])
        ry= sp.Matrix([[sp.cos(rot_yaxis), 0, sp.sin(rot_yaxis)],[0,1,0],[-sp.sin(rot_yaxis), 0, sp.cos(rot_yaxis)]])
        rz = sp.Matrix([[sp.cos(rot_zaxis), -sp.sin(rot_zaxis), 0], [sp.sin(rot_zaxis), sp.cos(rot_zaxis),0],[0,0,1]])
        
        R = rx*ry*rz
        t=sp.Matrix([translation[0],translation[1],translation[2]])
        
        E = R.col_insert(3,t)

        self.E = E

        self.P = self.K*self.E

        logger.info(f"Projection Camera: {self.P}")

    def projectpoint(self, worldpoint):
        '''
        Projects a sympy Matrix object of shape (4, 1)
        '''
        worldpoint=sp.Matrix(worldpoint)
        if worldpoint.shape[0] != self.P.shape[1]:
            worldpoint = worldpoint.row_insert(worldpoint.shape[0], sp.Matrix([[1]]))
        try:
            x = self.P * worldpoint
        except:
            logger.error(Exception, worldpoint)
        #print("projection point before flatten:")
        #print(x)
        x[0]=x[0]/x[2]
        x[1]=x[1]/x[2]
        x[2]=1#'''
        logger.info(f"Projected Point: {sp.Matrix([[x[0]],[x[1]]])}")
        return sp.Matrix([[x[0]],[x[1]]])
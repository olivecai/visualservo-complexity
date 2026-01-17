#!/usr/bin/env python
import sympy as sp
from .camera import Camera
from .dh_robot import DHSympyParams, DenavitHartenbergAnalytic
from .uvs import DenavitHartenberg_Cameras_Analytic
import numpy as np

P = DHSympyParams()
jntspace, cartspace, taskspace, c, l = P.get_params()
t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = jntspace
x, y, z = cartspace
u, v = taskspace



dof2_params = [
                [t0, 0, 0.5, 0], 
                [t1, 0, 0.5, 0]
                ]

dylan_dof3_params=[
                [ t0, sp.pi/2, 0 , 0 ],
                [ t1,  0  ,  0.55, 0 ],
                [ t2,  0  ,  0.3, 0 ]
                ]


kinova_dof7_params = [
    [t0,      sp.pi,   0.0,   0.0],
    [t1,      sp.pi/2, 0.0, -(.1564 + 0.1284)],
    [t2 +sp.pi, sp.pi/2, 0.0, -(0.0054 + 0.0064)],
    [t3 +sp.pi, sp.pi/2, 0.0, -(0.2104 + 0.2104)],
    [t4 +sp.pi, sp.pi/2, 0.0, -(0.0064 + 0.0064)],
    [t5 +sp.pi, sp.pi/2, 0.0, -(0.2084 + 0.1059)],
    [t6 +sp.pi, sp.pi/2, 0.0, 0.0],
    [t7 +sp.pi,    sp.pi, 0.0, -(0.1059 + 0.0615)],
]


dof4_params = [
    [t0,  sp.pi/2,  0.2,  0.0],   # Base rotates, link 1 vertical offset
    [t1,      0.0,  0.5,  0.0],   # Link 2 extends forward
    [t2,      0.0,  0.4,  0.0],   # Link 3 extends forward
    [t3,      0.0,  0.2,  0.0]    # Wrist rotation, small link for gripper
]

jaco_dh_params = [
    [ -t0,        1.5708,   0.0,    0.2755 ],   # Joint 1
    [ t1+1.5708,  3.1416,   0.4100, 0.0    ],   # Joint 2
    [ t2-1.5708,  1.5708,  -0.0098, 0.0    ],   # Joint 3
    [ t3,         0.9599,   0.0,   -0.0642 ],   # Joint 4
    [ t4+3.1416,  0.9599,   0.0,   -0.0642 ],   # Joint 5
    [ t5-1.1868,  3.1416,   0.0,   -0.1456 ]    # Joint 6
]


puma_dh_params = [
    [t0, -sp.pi/2, 0.0,     0.0     ],  # Joint 1
    [t1,     0.0 , 0.4318,  0.0     ],  # Joint 2
    [t2, -sp.pi/2, 0.0203,  0.15005 ],  # Joint 3
    [t3,  sp.pi/2, 0.0,     0.4318  ],  # Joint 4
    [t4, -sp.pi/2, 0.0,     0.0     ],  # Joint 5
    [t5,     0.0 , 0.0,     0.0     ]   # Joint 6
]


cam0 = Camera(0,0,0,[0,0,4], 4,4, 0, 0) 
cam1 = Camera(0.1,0.05,0,[0,0,4], 4,4, 0, 0) 
cam2 = Camera(-sp.pi/2, 0, 0.5, [0,0,4], 4,4,0,0) #looks at scene from the y axis, world z is cam2 y, world x is cam2 x 


analytic_cameras = [Camera(c[0],c[1],c[2],[c[3],c[4],c[5]], c[6],c[7], c[8], c[9]), Camera(c[10],c[11],c[12],[c[13],c[14],c[15]], c[16],c[17], c[18], c[19]) , Camera(c[20], c[21],c[22], [c[23],c[24],c[25]], c[26],c[27],c[28],c[29])] #looks at scene from the y axis, world z is cam2 y, world x is cam2 x 


cameras=[cam0, cam1, cam2]

class UVS:

    def __init__(self, name, cam_idx: list):
        self.cameras=[]
        self.cameras_analytic=[]
        for i in cam_idx:
            self.cameras.append(cameras[i])
            self.cameras_analytic.append(analytic_cameras[i])
        robot=None
        match name:
            case 'puma':
                robot = DenavitHartenbergAnalytic(puma_dh_params, P)
                Q = [.524,-1.047,2.094,0.,-1.571,1.571]
                
            case 'dof2':
                robot = DenavitHartenbergAnalytic(dof2_params, P)
                Q = [0.1,2.0]
            case 'dof3':
                robot = DenavitHartenbergAnalytic(dylan_dof3_params, P)
                Q = [0.,1.,-1.8]            
            case 'kinova':
                robot = DenavitHartenbergAnalytic(kinova_dof7_params, P)
                Q = np.deg2rad(np.array([-0.1336059570312672, -28.57940673828129, -179.4915313720703, -147.7, 0.06742369383573531, -57.420898437500036, 89.88030242919922, 0.5])).tolist()
            case 'jaco':
                robot = DenavitHartenbergAnalytic(jaco_dh_params, P)
                Q = [0.0, 0.6, -1.0, 1.5, 0.0, 0.0]
            case 'dof4':
                robot = DenavitHartenbergAnalytic(dof4_params, P)
                initQ=[0. ,1.2, -1.0, -1.2]
            case _:
                raise ValueError("Unknown robot name")  
        
        dof = len(Q)
    
        delta = np.pi / 4
        jointranges = [(q - delta, q + delta) for q in  Q]       

        self.uvs_model = DenavitHartenberg_Cameras_Analytic(cameras=self.cameras, analytic_cameras=self.cameras_analytic, dh_robot=robot)
        self.dh_robot = robot
        self.jointlimits = jointranges
        self.dof=dof

    def get_projections(self, Q:list):
        '''
        Q: list of lists
        Given Q, a list of joint angles, return the numpy matrix of all camera projections of the points, projections.
        projections is shape (2 * num_cameras, num angle pairs in Q)
        '''
        projections = []
        for cam in self.cameras:
            cam : Camera = cam
            cam_projections = []
            for q in Q: 
                cam_projections.append(cam.projectpoint(self.dh_robot.fkin_eval(*q)).flat())
            projections.append(cam_projections)
        return projections


if __name__ == "__main__":
    uvs = UVS('dof3', [0])
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(uvs.dh_robot.F)
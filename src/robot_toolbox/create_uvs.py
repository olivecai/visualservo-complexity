import sympy as sp
import dh_robot as dh
import uvs


P = dh.DHSympyParams()
jntspace, cartspace, taskspace = P.get_params()
t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = jntspace
x, y, z = cartspace
u, v = taskspace

dof2_params = [
                [t0, 0, .5, 0], 
                [t1, 0, .5, 0]
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

cam1 = dh.Camera(0.1,0.05,0,[0,0,4], 4,4, 0, 0) 
cam2 = dh.Camera(-sp.pi/2, 0, 0.5, [0,0,4], 4,4,0,0) #looks at scene from the y axis, world z is cam2 y, world x is cam2 x 
cameras=[cam1, cam2]

class UVS:

    def __init__(self, name, cam_idx: list):
        self.cameras=[]
        for i in cam_idx:
            self.cameras.append(cameras[i])
        robot=None
        match name:
            case 'puma':
                robot = dh.DenavitHartenbergAnalytic(puma_dh_params, P)
            case 'dof2':
                robot = dh.DenavitHartenbergAnalytic(dof2_params, P)
            case 'dof3':
                robot = dh.DenavitHartenbergAnalytic(dylan_dof3_params, P)
            case 'kinova':
                robot = dh.DenavitHartenbergAnalytic(kinova_dof7_params, P)
            case 'jaco':
                robot = dh.DenavitHartenbergAnalytic(jaco_dh_params, P)
            case 'dof4':
                robot = dh.DenavitHartenbergAnalytic(dof4_params, P)
            case _:
                raise ValueError("Unknown robot name")  
        self.uvs_model = uvs.DenavitHartenberg_Cameras_Analytic(cameras=self.cameras, dh_robot=robot)
        self.dh_robot = robot
        
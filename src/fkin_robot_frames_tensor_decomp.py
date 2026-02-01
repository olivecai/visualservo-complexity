'''
Jan 28 2026

This class will inherent methods from forward_kinematics.py
but instead of just the end effector, we will get all the robot frames.
'''

from sklearn.preprocessing import PolynomialFeatures
from robot_toolbox.create_uvs import UVS, analytic_cameras
from robot_toolbox.dh_robot import DHSympyParams, DenavitHartenbergAnalytic
from robot_toolbox.kinematic_structure import get_kinematic_structure
import matplotlib.pyplot as plt

from itertools import combinations, product

from basis import Basis
from forward_kinematics import ForwardKinematics, generate_symbolic_library_additive, generate_symbolic_library_multiply

import numpy as np
import sympy as sp
import logging

import os
from datetime import datetime
import sys

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

# Configure basic logging to the console (default level is WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger for the current module
# logger. debug, info, warning, error, critical

sin=np.sin
cos=np.cos
pi=np.pi

P = DHSympyParams()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_symbolic_library_incremental_additive(
    q,
    max_order= None,
    include_constant=True,
    primitive_sympy_functions=[sp.sin, sp.cos],
):
    primitive_sympy_functions=[sp.sin, sp.cos] #hardcode for now

    library = []
    active_where = []
    n = len(q)

    if include_constant:
        library.append(sp.Integer(1))
        active_where.append([1]*n)

    group = 0

    print(primitive_sympy_functions)

    for i in range(n):
        group = sp.Add(group, q[i])   # cumulative sum q1 + ... + qi
        print(group)
        for p in primitive_sympy_functions:
            print(p)
            library.append(p(group))
            print(library)

            mask = [0]*n
            mask[:i+1] = [1]*(i+1)   # joints 0..i active
            active_where.append(mask)

    return library, active_where

def rank_one_tensor(factor_vectors):
    """
    factor_vectors: list of 1D arrays [v1, v2, ..., vN]
    returns: N-way rank-1 tensor

    ex:
    rank_one_tensor([A[:, r], B[:, r], C[:, r]])
    rank_one_tensor([A[:, r], B[:, r], C[:, r], D[:, r]])

    """
    T = factor_vectors[0].reshape(-1, 1)
    for v in factor_vectors[1:]:
        T = np.tensordot(T, v, axes=0)
    return T.squeeze()


def normalize(x, lower=0, upper=1, axis=0):
    return (x - x.min(axis=axis)) / (x.max(axis=axis) - x.min(axis=axis))


def reconstruct(factors, weights=None):
    """
    factors: list of factor matrices [U1, U2, ..., UN]
    weights: optional CP weights
    """
    R = factors[0].shape[1]
    shape = [U.shape[0] for U in factors]
    T_hat = np.zeros(shape)

    for r in range(R):
        vectors = [U[:, r] for U in factors]
        T_r = rank_one_tensor(vectors)
        if weights is not None:
            T_r *= weights[r]
        T_hat += T_r

    return T_hat


def plot_factors(factors, mode_names=None):
    """
    factors: list of factor matrices

    use it like
    plot_factors(
    factors,
    mode_names=["samples", "frames", "xyz"]
        )
    """
    n_modes = len(factors)
    R = factors[0].shape[1]

    fig, axes = plt.subplots(R, n_modes, figsize=(4*n_modes, 2*R))
    if R == 1:
        axes = axes.reshape(1, -1)

    for mode, U in enumerate(factors):
        for r in range(R):
            ax = axes[r, mode]
            ax.plot(U[:, r])
            if r == 0:
                ax.set_title(mode_names[mode] if mode_names else f"Mode {mode}")
            if mode == 0:
                ax.set_ylabel(f"Factor {r+1}")

    plt.tight_layout()
    plt.show()

    
    
def compare_factors(factors, factors_actual, factors_ind=[0, 1, 2], fig=None):

    a_actual, b_actual, c_actual = factors_actual
    a, b, c = factors
    rank = a.shape[1]
    
    fig, axes = fig, np.array(fig.axes).reshape(rank, -1) if fig else plt.subplots(rank, 3, figsize=(8, int(rank * 1.2 + 1)))
    sns.despine(top=True)

    f_ind = factors_ind

    for ind, ax in enumerate(axes):
        ax1, ax2, ax3 = ax
        label, label_actual = ("Estimate", "Ground truth") if ind==0 else (None, None)
        ax1.plot(a_actual[:, ind], lw=5, c='b', alpha=.8, label=label_actual);  # a
        ax1.plot(a[:, f_ind[ind]], lw=2, c='red', label=label);  # a
        ax2.plot(b_actual[:, ind], lw=5, c='b', alpha=.8);  # b
        ax2.plot(b[:, f_ind[ind]], lw=2, c='red');  # a
        ax3.plot(c_actual[:, ind], lw=5, c='b', alpha=.8);  # c
        ax3.plot(c[:, f_ind[ind]], lw=2, c='red');  # a
        
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        ax3.set_yticklabels([])
        ax3.set_yticks([])
        ax1.set_ylabel("Factor {}".format(ind+1), fontsize=15)
        
        if ind != 2:
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax3.set_xticks([])
            ax3.set_xticklabels([])
        else:
            ax1.set_xlabel("Time", fontsize=15)
            ax2.set_xlabel("Neuron", fontsize=15)
            ax3.set_xlabel("Trial", fontsize=15)

    fig.tight_layout()
    fig.legend(loc='lower left', bbox_to_anchor= (0.08, -0.02), ncol=2, 
               borderaxespad=0, fontsize=15, frameon=False)
    
    return fig, axes


class ForwardKinematic_AllFrames(ForwardKinematics):
    def forward_kin_all_frames(self, q):
        # q is a list
        Ts= self.robot.rtb_robot.fkine_all(q=np.array(q))[1:]
        # self.robot.rtb_robot.plot(q, block=True)
        print("Ts:\n",Ts)
        frames = [T.t.tolist() for T in Ts]

        return frames
    
    def get_data(self, num_trajectories=50, num_pnts_per_traj=5, rng :np.random.Generator =None, add_input_noise=0.0, add_output_noise=0.05):
        '''
        Collect data for fitting the basis function.
        num_samples: number of samples to collect
        q_limits: list of tuples specifying the min and max for each joint variable
        '''
        num_samples = num_trajectories * num_pnts_per_traj
        logger.info(f"Collecting {num_samples} with add_output_noise={add_output_noise}, add_input_noise={add_input_noise}")

        joint_configs = self.generate_joint_samples_via_trajectory(num_trajectories, num_pnts_per_traj, rng=rng)
        fkin_data = [self.forward_kin_all_frames(q) for q in joint_configs]

        joint_configs = [(q + rng.normal(0, add_input_noise, len(q))).tolist() for q in joint_configs]
        fkin_data = [(fkin + rng.normal(0, add_output_noise, np.array(fkin).shape)).tolist() for fkin in fkin_data]
        logger.info(f"First 3 joint configs: {joint_configs[:5]}\nFirst 3 fkin data: {fkin_data[:5]}")
        return joint_configs, fkin_data
    
    def create_tensor(self,num_trajectories=50, num_pnts_per_traj=5, rng :np.random.Generator =None, add_input_noise=0.0, add_output_noise=0.05):
        joint_configs, fkin_data = self.get_data(num_trajectories=num_trajectories, num_pnts_per_traj=num_pnts_per_traj, rng=rng, add_input_noise=add_input_noise, add_output_noise=add_output_noise)
        # example of ONE datapoint for a 2dof arm in cartesian space is: [0, pi/2], [[0,0,0],[0.5,0,0],[0.5,0.5,0.0]]
        T = np.array(fkin_data) 
        logger.info(f"Tensor shape: {T.shape}")
        return T, joint_configs
    
    def iterative_model(self, fkin_frames, joint_configs, basis_library, active_where):
        '''
        Docstring for iterative_model
        
        :param self: 
        :param fkin_frames: an (N,F,S) shape tensor for Number of datapoints, F frames of the robot, S dimensions for workspace (ie, N=100 datapoint samples, F=2 frames of a 2 dof robot, S=3 for x,y,z cartesian directions or S=2 for u,v camera projection)
        :param joint_configs: an (N,D) shape matrix for Number of datapoints, D degrees of freedom of the robot
        :param basis_library: a list of length P, ie [1, sin, cos, etc etc]

        perform regression for each frame and only use the terms that are allowed ie if we are solving for frame 2 then only joints 1 and 2 are utilized, so only terms that involve joints 1 and 2 are involved
       
        
        or

        for each joint frame, see the joints it uses
        make a library of terms for those joints, and perform regression to select out of those combs
        then we only need to include those terms in the next regression 

        round 1
        basis = [1, sin(q0), cos(q0)]
        activated cos(q0)

        round 2
        basis = [cos(q0+q1), cos(q0-q1), cos(q0), sin(q0), cos(q1), sin(q1)]
        activated cos(q0+q1), cos(q0)

        round 3
        basis = [cos(q0+q1+q2), cos(q0+q1-q2), cos(q0+q1), etc...]
        activated ?

        results in model?'''



    def model_all_frames(self, fkin_frames, joint_configs, basis_library):

        print(f"frames: {fkin_frames}")
        print(f"joint configs: {joint_configs}")
        for i in range(fkin_frames.shape[1]):
            print("i",i)
            frame_i = fkin_frames[:,i,:]
            joint_configs_i = joint_configs[:,:i]
            print(f"frame i: {frame_i}")
            print(f"joint configs i: {joint_configs}")

       

robot_name='dof2'
robot = UVS(robot_name, cam_idx=[0]).dh_robot # ignore camera for true cartesian real world fkin
    
rng = np.random.default_rng(1)
a = ForwardKinematic_AllFrames(robot=robot)

T, joint_configs = a.create_tensor(5, 10, rng=rng, add_input_noise=0, add_output_noise=0)

basis_library = generate_symbolic_library_multiply(a.params, max_order=2, include_constant=True, primitive_sympy_functions=[sp.sin, sp.cos])



exit
print("T:")
print(T)
print("joint configs:\n",joint_configs)
R=5
weights, factors = parafac(
    tl.tensor(T),
    rank=R,
    normalize_factors=True
)

print("-"*8)
print(f"weights: {weights}")
print(f"factors: {factors}")

M_tl = reconstruct(factors)
plot_factors(factors, mode_names=["N samples", "Frames", "Cartesian motion"])

'''
[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]
[[0.0, 0.0, 0.0], [0.3535533905932738, 0.35355339059327373, 0.0], [0.3535533905932739, 0.8535533905932737, 0.0]]
[[0.0, 0.0, 0.0], [0.25000000000000006, 0.4330127018922193, 0.0], [-0.18301270189221924, 0.6830127018922194, 0.0]]

[0,pi/2]
[pi/4,pi/4]
[pi/3,pi/2]

[.5, .3536, 0.25]
[0.5, 0.3536, -0.18]



'''

'''
AX=B
where A is joints, X is weights, B is fkin. X will show weights of which bases (inherent in A) have been activated.

Now suppose we use multiple frames as part of our forward kinematics modelling: then
A is N x D:
[[t1 t2], [t1,t2], [t1,t2]] etc. N samples, each is D dof length

B is [[[frame1x,frame1y,frame1z],[frame2x,frame2y,frame2z]], 
      [[frame1x,frame1y,frame1z],[frame2x,frame2y,frame2z]],
      ...
      ]
B is then N x F x S, since there are N samples, F frames (TODO should F == D?), and S spatial directions (in this case x y z but for UBVS it can be u,v, or u1,v1,u2,v2)

Then, can we incrementally create the forward kinematics:
Regression over joint data to get fkin basis for frame 1 ==> AX


For one sample to find its values:
A is 1 x D
B is 1 x S

X is D x S ????

Or what if we had a matrix multiplication chain where each degree of freedom had three rotation matrices in R3 and we performed tensor decomposition:
Ax=B, so
D number of 3x3 rotation matrices * X = B (F x S) 

What is happening to my dimensions...

OR can i do SVD or tensor decomposition on each frame individually and then STACK those results together?

1. get the datapoints
2. transform the joints through the trig basis
3. for each joint configuration, get every frame of the robot
4. Then try to rebuild the kinematic structure using group regression
'''

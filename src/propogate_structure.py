'''
Jan 31 2026

This class will inherent methods from forward_kinematics.py
but instead of just the end effector, we will get all the robot frames.
'''

from sklearn.preprocessing import PolynomialFeatures
from robot_toolbox.create_uvs import UVS, analytic_cameras
from robot_toolbox.dh_robot import DHSympyParams, DenavitHartenbergAnalytic
from robot_toolbox.kinematic_structure import get_kinematic_structure
import matplotlib.pyplot as plt
# import cvxpy as cp

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

class ForwardKinematic_AllFrames(ForwardKinematics):
    def forward_kin_all_frames(self, q):
        # q is a list
        Ts= self.robot.rtb_robot.fkine_all(q=np.array(q))[1:]
        # self.robot.rtb_robot.plot(q, block=True)
        # print("Ts:\n",Ts)
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
        joint_configs = [(q + rng.normal(0, add_input_noise, len(q))).tolist() for q in joint_configs]
        fkin_data = [self.forward_kin_all_frames(q) for q in joint_configs]

        fkin_data = [(fkin + rng.normal(0, add_output_noise, np.array(fkin).shape)).tolist() for fkin in fkin_data]
        logger.info(f"First 3 joint configs: {joint_configs[:5]}\nFirst 3 fkin data: {fkin_data[:5]}")
        return joint_configs, fkin_data
    
    def create_tensor(self,num_trajectories=50, num_pnts_per_traj=5, rng :np.random.Generator =None, add_input_noise=0.0, add_output_noise=0.05):
        joint_configs, fkin_data = self.get_data(num_trajectories=num_trajectories, num_pnts_per_traj=num_pnts_per_traj, rng=rng, add_input_noise=add_input_noise, add_output_noise=add_output_noise)
        # example of ONE datapoint for a 2dof arm in cartesian space is: [0, pi/2], [[0,0,0],[0.5,0,0],[0.5,0.5,0.0]]
        T = np.array(fkin_data) 
        logger.info(f"Tensor shape: {T.shape}")
        return T, joint_configs
        

    def model_all_frames(self, fkin_frames, joint_configs, basis_library):
        # UNFINISHED
        print(f"frames: {fkin_frames}")
        print(f"joint configs: {joint_configs}")
        for i in range(fkin_frames.shape[1]):
            print("i",i)
            frame_i = fkin_frames[:,i,:]
            joint_configs_i = joint_configs[:,:i]
            print(f"frame i: {frame_i}")
            print(f"joint configs i: {joint_configs}")

       
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

def hierarchical_sparse_regression(
    Phi,
    y,
    groups_ext,
    groups_root,
    lambda_ext=1e-2,
    lambda_root=5e-2,
):
    """
    Currently not in use, since robotics toolbox and cvxpy have conflicting numpy versions

    Phi: (N, M) design matrix
    y:   (N,)   target
    groups_ext:  list of lists of column indices (extensions)
    groups_root: list of lists of column indices (new roots)
    """

    M = Phi.shape[1]
    x = cp.Variable(M)

    # data fidelity
    loss = cp.sum_squares(Phi @ x - y)

    # group sparsity penalties
    penalty = 0

    for g in groups_ext:
        penalty += lambda_ext * cp.norm(x[g], 2)

    for g in groups_root:
        penalty += lambda_root * cp.norm(x[g], 2)

    problem = cp.Problem(cp.Minimize(loss + penalty))
    problem.solve(solver=cp.OSQP)

    print("x value: \n", x)
    return x.value

def group_soft_threshold(x, groups, lam, step):
    x_new = x.copy()
    for g in groups:
        v = x[g]
        norm = np.linalg.norm(v)
        if norm > lam * step:
            x_new[g] *= (1 - lam * step / norm)
        else:
            x_new[g] = 0.0
    return x_new


def hierarchical_sparse_regression_numpy(
    Phi,
    y,
    groups_ext,
    groups_root,
    lambda_ext=1e-2,
    lambda_root=5e-2,
    max_iter=200,
    step=1e-3,
):
    """
    NumPy-only hierarchical group sparse regression
    """

    N, M = Phi.shape
    x = np.zeros(M)

    for _ in range(max_iter):
        # gradient of ||Phi x - y||^2
        grad = 2 * Phi.T @ (Phi @ x - y)

        # gradient descent step
        x -= step * grad

        # root groups (stronger penalty)
        x = group_soft_threshold(x, groups_root, lambda_root, step)

        # extension groups (weaker penalty)
        x = group_soft_threshold(x, groups_ext, lambda_ext, step)

    return x


def recover_forward_kin(model: ForwardKinematic_AllFrames, joint_configs: np.array, fkin_data : np.array, primitives=[sp.sin, sp.cos]):
   
    q = model.params #sympy symbols q0,q1,q2, ... len==dof
    TOLERANCE=0.3
    fkin_data=np.array(fkin_data)
    joint_configs=np.array(joint_configs)

    print("RECOVER FORWARD MODEL")
    
    # since the fkin data is a tensor size (N,F,S):
    recovered_bases = [] #should end up being size (1, S, P), where S is dim(workspace) and P is the number of basis elements
    print(fkin_data.shape)
    for workspace_dim in range(fkin_data.shape[2]):
        
        # initialization stage:
        funcs = []
        args = []

        # funcs and args are parallel lists


        ############### Now we have set up
        for frame_idx in range(fkin_data.shape[1]): #for each frame (workspace_dimension[i] is each frame in that workspace dimension)
            print("SOLVING FOR ", workspace_dim, frame_idx)
            funcs_next=[]
            args_next=[]
            indicator=[]
            parents= []
            
            funcs = list(funcs)
            args = list(args)

            for i in range(len(funcs)):
                funcs_next.append(funcs[i])
                args_next.append(args[i])
                indicator.append(0) #current active
                parents.append(i)

            for i in range(len(funcs)):
                funcs_next.append(funcs[i])
                args_next.append(args[i]+q[frame_idx])
                indicator.append(1) # extensions for next
                parents.append(i) 

                funcs_next.append(funcs[i])
                args_next.append(args[i]-q[frame_idx])
                indicator.append(1) # extensions for next
                parents.append(i) 

            for p in primitives:
                funcs_next.append(p)
                args_next.append(q[frame_idx])
                indicator.append(2) #root
                parents.append(None)
            
            for i in range(len(funcs_next)):
                print(f"\nelement i: {i}")
                print(f"function : {funcs_next[i]}")
                print(f"argument : {args_next[i]}")
                print(f"indicator: {indicator[i]}\n")
                

            print("ARGS:", args)
            print("FUNCS:", funcs)
            print("ARGS NEXT:", args_next)
            print("FUNCS NEXT:", funcs_next)   
            print("INDICATOR:", indicator)
            
            phi=[] #contains everything!
            propogated = [] #contains the previous basis
            extension = [] #contains the terms where we extend the argument of the primitive
            root =[] #a new basis element of just one term

            # designate the propogated, extension, and root components:
            for i in range(len(funcs_next)):
                element = funcs_next[i](args_next[i])
                if indicator[i] == 0:
                    propogated.append(element)
                if indicator[i] == 1:
                    extension.append(element)
                if indicator[i] == 2:
                    root.append(element)
                
                phi.append(element)
            print(f"propogated:\n{propogated}\nextension:\n{extension}\nroot:\n{root}")

            # joint_configs: (N, dof)
            N = joint_configs.shape[0]
            M = len(phi)

            Phi = np.zeros((N, M))

            for j, expr in enumerate(phi):
                f = sp.lambdify(q, expr, "numpy")
                Phi[:, j] = f(*joint_configs.T)

            groups_ext = []
            groups_root = []

            for child_idx, parent_idx in enumerate(parents):
                if parent_idx is None:
                    groups_root.append([child_idx])
                else:
                    groups_ext.append([parent_idx, child_idx])

            # fkin_data shape: (N, F, S)
            y = fkin_data[:, frame_idx, workspace_dim]  

            Phi_mean = Phi.mean(axis=0, keepdims=True)
            Phi_std  = Phi.std(axis=0, keepdims=True) + 1e-12
            Phi_n = (Phi - Phi_mean) / Phi_std


            # solve for the frame and get back the vector weights
            weights_n = hierarchical_sparse_regression_numpy(
                Phi=Phi_n,
                y=y,
                groups_ext=groups_ext,
                groups_root=groups_root,
                lambda_ext=1e-2,
                lambda_root=5e-2,
            )
            weights = weights_n / Phi_std.ravel()

            print(f"WEIGHTS: \n{weights}")

            # active_mask = []
            # min_w = min(abs(weights))
            # max_w = max(abs(weights))
            # TOLERANCE = (max_w+min_w)/2
            # TOLERANCE = max(TOLERANCE, 0.1)
            
            # for i in range(len(weights)):
            #     if abs(weights[i]) > TOLERANCE:
            #         active_mask.append(1)
            #     else:
            #         active_mask.append(0)

            abs_w = np.abs(weights)
            order = np.argsort(abs_w)[::-1]
            energy = np.cumsum(abs_w[order]**2)
            energy /= (energy[-1] + 1e-12)

            keep = order[energy <= 0.99]          # keep terms explaining 99% of weight energy
            active_mask = np.zeros_like(weights, dtype=int)
            active_mask[keep] = 1

            print(f"ACTIVE MASK:\n{active_mask}")
            
            funcs=[]
            args=[]
            for i in range(len(active_mask)):
                if active_mask[i]==1:
                    print("TRUE")
                    print(funcs_next[i])
                    print(args_next[i])
                    funcs.append(funcs_next[i])
                    args.append(args_next[i])

            recovered_basis=[]
            for i in range(len(funcs)):
                element = funcs[i](args[i])           
                recovered_basis.append(element)
        
            print(f"basis at frame idx {frame_idx}: {recovered_basis}")

        print(f"recovered basis: {recovered_basis}")
        recovered_bases.append(recovered_basis)

    return recovered_bases

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



robot_name='kinova'
robot = UVS(robot_name, cam_idx=[0]).dh_robot # ignore camera for true cartesian real world fkin
    
rng = np.random.default_rng(1)
a = ForwardKinematic_AllFrames(robot=robot)

# T, joint_configs = a.create_tensor(5, 10, rng=rng, add_input_noise=0, add_output_noise=0)

basis_library = generate_symbolic_library_multiply(a.params, max_order=2, include_constant=True, primitive_sympy_functions=[sp.sin, sp.cos])

###
joint_configs, fkin_data = a.get_data(50, 1, rng, 0.0, 0.0)
recovered_bases =recover_forward_kin(a, joint_configs, fkin_data)
print("DONE")
print(recovered_bases)

# validate the data

basis_objects =[]
basis_objects.append(Basis('x'))
basis_objects.append(Basis('y'))
basis_objects.append(Basis('z'))
rng_eval=np.random.default_rng(2)
joint_configs_eval, fkin_data_eval = a.get_data(100, 1, rng, 0.0, 0.0)

ee_idx = a.dof - 1
dims = ["x", "y", "z"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

for i in range(len(basis_objects)):
    b:Basis=basis_objects[i]
    b.setup(a.params, 1e-1, recovered_bases[i])
    symbolic_basis, weights = b.train(train_b=np.array(fkin_data)[:,a.dof-1, i],train_a=np.array(joint_configs))
    print(f"TRAINING: symbolic basis {symbolic_basis}, weights: {weights}")
    RSS = b.evaluate(eval_b=np.array(fkin_data_eval)[:,a.dof-1, i], eval_a=joint_configs_eval)
    print("EVALUATED RSS:", RSS)

    y_training_pred=[]
    for joint in joint_configs:
        y_training_pred.append(b.get_prediction(joints=joint))
    y_eval_pred =[]
    for joint in joint_configs_eval:
        y_eval_pred.append(b.get_prediction(joints=joint))

    y_eval_true = np.array(fkin_data_eval)[:,a.dof-1, i]
    y_training_true =np.array(fkin_data)[:,a.dof-1, i]


    ax = axes[i]
    ax.plot(y_eval_pred, "k.", alpha=0.6, label="y_eval_pred")
    ax.plot(y_eval_true, "r.", alpha=0.3, label="y_eval_true")
    ax.plot(y_training_pred, "g.", alpha=0.6, label="y_training_pred")
    ax.plot(y_training_true, "b.", alpha=0.3, label="y_training_true")

    ax.set_title(f"End-effector {dims[i]}")
    ax.set_ylabel("Position")
    ax.set_xlabel("Sample index")
    ax.grid(True)
    

axes[0].legend()
plt.tight_layout()
plt.show()
exit()
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

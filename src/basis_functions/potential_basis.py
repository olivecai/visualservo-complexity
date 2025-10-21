'''
Oct 20 2025

First attempt at creating a basis function to represent the visual servoing function.
We can first try with many different basis functions and analyze the coefficients to determine 
which functions are effective bases.

Some notes on Jacobian basis function fitting:
- We will likely need to create region-specific functions to cover the joint space well,
but for now we can use one function for the entire space (thus choose a slightly smaller joint region)

'''
from ..robot_toolbox.create_uvs import UVS
import numpy as np
import sympy as sp
import logging

uvs = UVS('dof2', [0])

# create a basis to approximate the UVS function
class jacobian_basis:
    def __init__(self, uvs: UVS, phi):

        self.uvs = uvs
        self.phi = phi # basis function, should be a function that takes in joint variables and returns a scalar
        
    

    def collect_data(self, num_trajectories=50, num_pnts_per_traj=5):
        '''
        Collect data for fitting the basis function.
        num_samples: number of samples to collect
        q_limits: list of tuples specifying the min and max for each joint variable
        '''
        num_samples = num_trajectories * num_pnts_per_traj
        logging.info(f"Collecting {num_samples} samples for basis function fitting...")
        # first we collect sample joint-jacobian pairs to fit our basis.
        joint_configs = self.generate_joint_samples_via_trajectory(num_trajectories, num_pnts_per_traj)
        jacobian_data = self.generate_jacobians_given_joint_configs(joint_configs)
        return joint_configs, jacobian_data
    
    def fit_basis_function(self, joint_configs, jacobian_data, phi):
        '''
        Given joint configs and their corresponding Jacobians,
        fit a basis function to approximate the UVS Jacobian. 

        Joint configs and Jacobian data are parallel lists collected from class method collect_data().
        We will 'unravel' the Jacobian into one long vector for fitting.

        phi is the basis function to fit for each entry of the Jacobian: J_ij = phi(q).
        '''
        logging.info("Fitting basis function to collected data...")
        # unravel jacobian: 
        J_vectors = [J.flatten() for J in jacobian_data]
        J_matrix = np.array(J_vectors)  # shape: (num_samples, m*n) where J is m x n

        phi = np.array([self.phi(q) for q in joint_configs]) # shape: (num_samples,dof)
        logging.info(f"phi shape: {phi.shape}, J_matrix shape: {J_matrix.shape}")
        # perform the linear regression to obtain coefficient matrix A:
        A, residuals, rank, s = np.linalg.lstsq(phi, J_matrix, rcond=None) # shape: (dof, m*n)
        logging.info(f"Fitted basis function with residuals: {residuals}")
        logging.info(f"Coefficient matrix: {A}")
        return A  # shape: (dof, m*n)

    def generate_joint_samples_via_trajectory(self, num_trajectories, num_pnts_per_traj):
        '''
        Returns a list of joint configurations sampled over the joint space with the uvs.jointlimits
        
        For num_trajectories, generate random linear trajectories in joint space, and sample num_pnts_per_traj points along each trajectory.
        '''
        joint_configs = []
        for _ in range(num_trajectories):
            start_q = np.array([np.random.uniform(low, high) for (low, high) in self.uvs.jointlimits])
            end_q = np.array([np.random.uniform(low, high) for (low, high) in self.uvs.jointlimits])
            for t in np.linspace(0, 1, num_pnts_per_traj):
                q = (1 - t) * start_q + t * end_q
                joint_configs.append(q)
        return joint_configs
    
    def generate_jacobians_given_joint_configs(self, joint_configs):
        '''
        Given a list of joint configurations, compute the corresponding Jacobians using UVS central differences.
        '''
        jacobians = []
        for q in joint_configs:
            J = self.uvs.uvs_model.central_differences_pp(q)
            jacobians.append(J)
        return jacobians



#!/usr/bin/env python

'''
Oct 20 2025

First attempt at creating a basis function to represent the visual servoing function.
We can first try with many different basis functions and analyze the coefficients to determine 
which functions are effective bases.

Some notes on Jacobian basis function fitting:
- We will likely need to create region-specific functions to cover the joint space well,
but for now we can use one function for the entire space (thus choose a slightly smaller joint region)

'''
from robot_toolbox.create_uvs import UVS

import numpy as np
import sympy as sp
import logging

# Configure basic logging to the console (default level is WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger for the current module
# logger. debug, info, warning, error, critical



# create a basis to approximate the UVS function
class JacobianBasis:
    def __init__(self, uvs: UVS):

        self.uvs = uvs
        '''
        User intstructions:

        Initialize a jacobian_basis object,
        set; collect data, get the coefficient matrix, then validate that model by get_approximate_jacobian_from_regression_model.
        
        '''
        self.m : int = self.uvs.dof
        self.K :int = (self.phi([0]*self.m).shape[0]) 
        self.n : int = len(self.uvs.cameras) * 2


    def phi(self, q):
        '''
        phi is the vector that contains our basis columns. 
        to modify the basis, change the variables below for polynomial_degree, trigonometric, etc.
        '''

        ##### CHANGE BASIS VARIABLEs/COLUMNS BELOW #####
        polynomial_degree : int = 2 # 1 for linear, 2 for quadratic, so on so forth
        trigonometric : bool = True
        ################################################

        # q is some vector: q1, q2, q3, ..., qdof
        # for now our basis can be very basic:
        phi_vector = []
        for k in range(1,polynomial_degree+1):
            for i in range(len(self.uvs.dof)):
                phi_vector.append(q[i]**k)
        if trigonometric: # for now let's not implement this: let's see how far the polynomial basis gets us.
            pass

        phi_vector = np.array(phi_vector)
        logger.info("Phi vector:",phi_vector)

        return phi_vector

    def collect_data(self, num_trajectories=50, num_pnts_per_traj=5):
        '''
        Collect data for fitting the basis function.
        num_samples: number of samples to collect
        q_limits: list of tuples specifying the min and max for each joint variable
        '''
        num_samples = num_trajectories * num_pnts_per_traj
        logger.info(f"Collecting {num_samples} samples for basis function fitting...")
        # first we collect sample joint-jacobian pairs to fit our basis.
        joint_configs = self.generate_joint_samples_via_trajectory(num_trajectories, num_pnts_per_traj)
        jacobian_data = self.generate_jacobians_given_joint_configs(joint_configs)
        return joint_configs, jacobian_data

    def generate_joint_samples_via_trajectory(self, num_trajectories, num_pnts_per_traj):
        '''
        TO BE USED IN "collect_data"

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
        TO BE USED IN "collect_data"

        Given a list of joint configurations, compute the corresponding Jacobians using UVS central differences.
        '''
        jacobians = []
        for q in joint_configs:
            J = self.uvs.uvs_model.central_differences_pp(q)
            jacobians.append(J)
        return jacobians
    
    def get_coefficient_matrix(self, joint_configs, jacobian_data, phi: function):
        '''

        Given joint configs and their corresponding Jacobians,
        fit a basis function to approximate the UVS Jacobian. 

        Joint configs and Jacobian data are parallel lists collected from class method collect_data().
        We will 'unravel' the Jacobian into one long vector for fitting.

        phi is the basis function to fit for each entry of the Jacobian: J_ij = phi(q).

        Dimensions sanity check:
        N is the number of datapoints we collect.
        m is the number of task space coordinates
        n is the number of dof 
        K is the number of basis columns

        Then, 
        Jacobian is shape (m, n)
        Flattened Jacobian is shape (1, m*n)
        Flattened Jacobian Matrix is shape (N, m*n)

        Phi shape is (1,K)
        Phi matrix is shape (N,K)

        In Ax=B, we have x=A_inv @ B. Here, x is our coefficient matrix, A is phi matrix, and B is J matrix.
        x = A\B, so coeff matrix = phi matrix inverse \ J matrix ==> phi matrix inverse is shape (K,N), J matrix is shape (N, m*n)
        Then x is shape (K, m*n):
        Coeff matrix is shape (K, m*n).

        Then if we want to solve for any arbitrary new joint q:
        q is shape (1, n)
        phi is shape (1,K)
        J_new = phi @ coeff matrix, so J_new shape (1,m*n). Then we may reshape the Jacobian to become (m,n).

        '''
        logger.info("Fitting basis function to collected data...")
        # unravel jacobian: 
        J_vectors = [J.flatten() for J in jacobian_data]
        J_matrix = np.array(J_vectors)  # shape: (N, m*n) where J is m x n

        phi = np.array([self.phi(q) for q in joint_configs]) # shape: (N,K)
        logger.info("phi:", phi)
        logger.info("J_matrix:", J_matrix)
        logger.info(f"phi shape: {phi.shape}, J_matrix shape: {J_matrix.shape}")
        # perform the linear regression to obtain coefficient matrix A:
        coeff_matrix, residuals, rank, s = np.linalg.lstsq(phi, J_matrix, rcond=None) # shape: (K, m*n) 
        logger.info(f"Fitted basis function with residuals: {residuals}")
        logger.info(f"Coefficient matrix: {coeff_matrix}")
        return coeff_matrix  # shape: (K, m*n)

    
    def get_approximate_jacobian_from_regression_model(self, coefficient_matrix:np.ndarray):
        '''
        Get user input q and evaluate those joints, then evaluate a series of joints and return the accuracy/goodness of the model.
        '''
        logger.info("Evluating the regression model with various joint configurations q:")
    
        try:
            while True:
                q_input_str = input("Enter joints in radians as q1,q2, ..., qdof:")
                
                q = [float(x) for x in q_input_str.split(',')]
                J_approximated = self.phi(q) @ coefficient_matrix # (1,K) @ (K, (m*n))
                J_approximated.reshape(self.m, self.n)

                J_true = self.uvs.uvs_model.central_differences_pp(Q=q)
                L2_error = np.linalg.norm(J_approximated-J_true, ord=2)

                logging.info("JOINT CONFIGURATION:\n", q)
                logging.info("COMPUTED JACOBIAN:\n", J_approximated)
                logging.info("TRUE JACOBIAN:\n", J_true)
                logging.info("L2 ERROR:", L2_error)

                
        except Exception as e:
            logging.info("\nError occurred: {}; exiting program.".format(e))

def main():
    uvs = UVS('dof2', [0]) #2 dof arm with a direct projection onto the scene from above
    jacobian_basis = JacobianBasis(uvs)
    sample_joints, sample_jacobians = jacobian_basis.collect_data(10, 5)
    coeff_mat = jacobian_basis.get_coefficient_matrix(sample_joints, sample_jacobians)
    jacobian_basis.get_approximate_jacobian_from_regression_model(coefficient_matrix=coeff_mat)

    



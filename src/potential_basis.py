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
from sklearn.preprocessing import PolynomialFeatures
from robot_toolbox.create_uvs import UVS
import matplotlib.pyplot as plt

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
        self.K :int = (self.phi(np.array([[0]*self.m])).shape[0]) 
        self.n : int = len(self.uvs.cameras) * 2


    def phi(self, q : np.array):
        '''
        pass q as np.array([[q1_1,...q1_dof],[q2_1,...,q2_dof,...qN_11,...qN_dof]]) where N is the number of sample points to regress over.
        phi is the vector that contains our basis columns. 
        to modify the basis, change the variables below for polynomial_degree, trigonometric, etc.
        '''

        ##### CHANGE BASIS VARIABLEs/COLUMNS BELOW #####
        polynomial_degree : int = 2 # 1 for linear, 2 for quadratic, so on so forth
        trigonometric : bool = True
        ################################################

        # q is some vector: q1, q2, q3, ..., qdof
        # for now our basis can be very basic:
        poly = PolynomialFeatures(polynomial_degree)
        phi_vector = poly.fit_transform(q)

        if trigonometric: # for now let's not implement this: let's see how far the polynomial basis gets us.
            phi_vector = np.hstack([
                phi_vector,
                np.sin(q),       # sin(x1), sin(x2)
                np.cos(q)        # cos(x1), cos(x2)
            ])

        logger.info(f"Phi vector: {phi_vector}")

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
    
    def get_coefficient_matrix(self, joint_configs, jacobian_data):
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
        print(np.array(joint_configs))
        phi = self.phi(np.array(joint_configs)) # shape: (N,K)

        logger.info(f"phi: {phi}")
        logger.info(f"J_matrix: {J_matrix}")
        logger.info(f"phi shape: {phi.shape}, J_matrix shape: {J_matrix.shape}")
        # perform the linear regression to obtain coefficient matrix A:
        coeff_matrix, residuals, rank, s = np.linalg.lstsq(phi, J_matrix, rcond=None) # shape: (K, m*n) 
        logger.info(f"Residuals: {residuals}")
        logger.info(f"Coefficient matrix:\n {coeff_matrix}")

        return coeff_matrix  # shape: (K, m*n)

    
    def get_approximate_jacobian_from_regression_model(self, coefficient_matrix:np.ndarray):
        '''
        Get user input q and evaluate those joints, then evaluate a series of joints and return the accuracy/goodness of the model.
        '''
        logger.info("Evaluating the regression model with various joint configurations q:")
    
        try:
            while True:
                q_input_str = input("Enter joints in radians as q1,q2, ..., qdof:")
                
                q = [float(x) for x in q_input_str.split(',')]
                logger.info(f"JOINT CONFIGURATION:\n{q}")
                J_true = self.uvs.uvs_model.central_differences_pp(Q=q)
                logger.info(f"TRUE JACOBIAN:\n{J_true}")
                J_approximated = self.phi([np.array(q)]) @ coefficient_matrix # (1,K) @ (K, (m*n))
                J_approximated =J_approximated.reshape(self.m, self.n)
                logger.info(f"COMPUTED JACOBIAN:\n{J_approximated}")

                L2_error = np.linalg.norm(J_approximated-J_true, ord=2)

                logger.info(f"L2 ERROR:\n{L2_error}")

                
        except Exception as e:
            logging.info("\nError occurred: {}; exiting program.".format(e))

    def evaluate_goodness_of_fit(self, coefficient_matrix:np.ndarray, num_pnts_per_traj=int, num_trajectories = int):
        '''
        Evaluate goodness of fit of regression model (given coefficient matrix) over a series of random joint trajectories.
        '''
        joint_configs, jacobians = self.collect_data(num_pnts_per_traj=num_pnts_per_traj, num_trajectories=num_trajectories)
        total_error = 0.0
        for q, J_true in zip(joint_configs, jacobians):
                J_approximated = self.phi([np.array(q)]) @ coefficient_matrix # (1,K) @ (K, (m*n))
                J_approximated =J_approximated.reshape(self.m, self.n)
                L2_error = np.linalg.norm(J_approximated-J_true, ord=2)
                total_error += L2_error
        avg_error = total_error / len(joint_configs)
        
        # plot fkin(q) points with the jacobian goodness
        





def main():
    uvs = UVS('dof2', [0]) #2 dof arm with a direct projection onto the scene from above
    jacobian_basis = JacobianBasis(uvs)
    sample_joints, sample_jacobians = jacobian_basis.collect_data(50, 5)
    coeff_mat = jacobian_basis.get_coefficient_matrix(sample_joints, sample_jacobians)
    jacobian_basis.get_approximate_jacobian_from_regression_model(coefficient_matrix=coeff_mat)


main()
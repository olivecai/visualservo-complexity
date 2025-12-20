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

import os
from datetime import datetime
import sys

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
    
        self.m : int = self.uvs.dof #dof
        self.phi_degree = None
        self.n : int = len(self.uvs.cameras) * 2
        logger.info(f"m: {self.m}, phi_degree: {self.phi_degree}, n: {self.n}")

    def set_phi(self, phi_deg, phi_type):
        self.phi_degree = phi_deg
        self.phi=phi_type
        self.K :int = (self.phi(np.array([[0]*self.m])).shape[1]) 


    def polynomial_phi(self, q : np.array):
        '''
        pass q as np.array([[q1_1,...q1_dof],[q2_1,...,q2_dof,...qN_11,...qN_dof]]) where N is the number of sample points to regress over.
        phi is the vector that contains our basis columns. 
        to modify the basis, change the variables below for polynomial_degree, trigonometric, etc.
        '''
        poly = PolynomialFeatures(self.phi_degree)
        phi_vector = poly.fit_transform(q)

        return phi_vector
    
    def trigonometric_phi(self, q : np.array):
        '''
        pass q as np.array([[q1_1,...q1_dof],[q2_1,...,q2_dof,...qN_11,...qN_dof]]) where N is the number of sample points to regress over.
        phi is the vector that contains our basis columns. 
        to modify the basis, change the variables below for polynomial_degree, trigonometric, etc.
        '''
        # q is some vector: q1, q2, q3, ..., qdof
        # for now our basis can be very basic:

        sin_q = np.sin(q)
        cos_q = np.cos(q)

        logger.info(f"sin_q: {sin_q}")
        logger.info(f"cos q: {cos_q}")

        trig_q = np.concatenate((sin_q, cos_q), axis=1)
        logger.info(f"trig q: {trig_q}")
        
        
        trig_poly = PolynomialFeatures(self.phi_degree)
        trig_phi_vector = trig_poly.fit_transform(trig_q)

        logger.info(f"Phi vector: {trig_phi_vector}")

        return trig_phi_vector
        

    def collect_data(self, num_trajectories=50, num_pnts_per_traj=5, rng=None):
        '''
        Collect data for fitting the basis function.
        num_samples: number of samples to collect
        q_limits: list of tuples specifying the min and max for each joint variable
        '''
        num_samples = num_trajectories * num_pnts_per_traj
        logger.info(f"Collecting {num_samples} samples for basis function fitting...")
        # first we collect sample joint-jacobian pairs to fit our basis.
        joint_configs = self.generate_joint_samples_via_trajectory(num_trajectories, num_pnts_per_traj, rng=rng)
        jacobian_data = self.generate_jacobians_given_joint_configs(joint_configs)
        return joint_configs, jacobian_data
        
    def generate_joint_samples_via_trajectory(self, num_trajectories, num_pnts_per_traj, rng=None):
        '''
        Reproducible random generator-based sampling.
        '''
        if rng is None:
            rng = np.random.default_rng()

        joint_configs = []

        for _ in range(num_trajectories):
            start_q = np.array([rng.uniform(low, high) 
                                for (low, high) in self.uvs.jointlimits])
            end_q = np.array([rng.uniform(low, high) 
                            for (low, high) in self.uvs.jointlimits])

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
        n is the number of task space coordinates
        m is the number of dof 
        K is the number of basis columns

        Then, 
        Jacobian is shape (n, m)
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
        J_new = phi @ coeff matrix, so J_new shape (1,m*n). Then we may reshape the Jacobian to become (n,m).

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
        Used for testing small examples
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



    def evaluate_goodness_of_fit(self, coefficient_matrix:np.ndarray,  num_trajectories:int, num_pnts_per_traj:int, rng:np.random.Generator, output_folder = 'test', output_name='test'):
        '''
        Evaluate goodness of fit of regression model (given coefficient matrix) over a series of random joint trajectories.
        '''

        joint_configs, jacobians = self.collect_data(num_pnts_per_traj=num_pnts_per_traj, num_trajectories=num_trajectories, rng=rng)
        total_error = 0.0
        colors=[]
        for q, J_true in zip(joint_configs, jacobians):
                J_approximated = self.phi([np.array(q)]) @ coefficient_matrix # (1,K) @ (K, (m*n))
                J_approximated =J_approximated.reshape(self.n, self.m)
                L2_error = np.linalg.norm(J_approximated-J_true, ord=2)
                total_error += L2_error
                colors.append(L2_error)
        avg_error = total_error / len(joint_configs)
        logger.info(f"Average error:\n{avg_error}")
        # plot fkin(q) points with the jacobian goodness
        projections = self.uvs.get_projections(np.array(joint_configs).tolist())
        
        #projections is [cam1, cam2]
        number_of_cameras = int (self.n / 2)
        
        ncols = number_of_cameras
        nrows=1
        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), layout='constrained')
        
        for i in range(number_of_cameras):
            camera_i = np.array(projections[i]).T
            try:
                ax= axs[i]
            except:
                ax=axs
            mappable = ax.scatter(camera_i[0],camera_i[1], c=colors, cmap = 'viridis')
            ax.set_title(f"Camera {i} Projection Goodness of Fit")

        try:
            fig.colorbar(mappable, ax=axs.ravel().tolist())
        except:
            fig.colorbar(mappable, ax=axs)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        path  = os.path.join(output_folder, output_name)
        plt.savefig(path)

        return avg_error
        # plt.show()
        # plt.close()
            
        





def main():
    '''
    Usage example: `python3 ./src/potential_basis.py dof2 0,1 2 5 10 500 10 888 testing`
    '''

    params = sys.argv
    for i in range(0,len(params)):
        try:
            params[i] = int(params[i])
        except:
            pass
    _, robot, camera_setup, num_trajs_sample, num_pnts_per_traj_sample, num_trajs_eval, num_pnts_per_traj_eval, phi_degrees, phi_types, random_seed, output_folder= params

    camera_setup = [int(i) for i in camera_setup.split(',')]
    phi_degrees = [int(i) for i in phi_degrees.split(',')]
    phi_types = [int(i) for i in phi_types.split(',')] 
    random_seed = random_seed if random_seed >=0 else None

    logger.info(f"robot: {robot}\ncamera_setup: {camera_setup}\nnum_trajs_sample: {num_trajs_sample}\nnum_pnts_per_traj_sample: {num_pnts_per_traj_sample}\nnum_trajs_eval: {num_trajs_eval}\nnum_pnts_per_traj_eval: {num_pnts_per_traj_eval}\nrandom_seed: {random_seed}\noutput_folder:{output_folder}")

    uvs = UVS(robot, camera_setup) 
    jacobian_basis = JacobianBasis(uvs)
    rng = np.random.default_rng(seed=random_seed)
    sample_joints, sample_jacobians = jacobian_basis.collect_data(num_trajectories=num_trajs_sample,num_pnts_per_traj=num_pnts_per_traj_sample,rng=rng)
    
        # jacobian_basis.get_approximate_jacobian_from_regression_model(coefficient_matrix=coeff_mat) #user io test for individual points
  
    
    for type in phi_types:
        for deg in phi_degrees:
            if type==0:
                phi_type_name = 'polynomial_phi'
                phi_type=jacobian_basis.polynomial_phi
            if type==1:
                phi_type_name = 'trigonometric_phi'
                phi_type=jacobian_basis.trigonometric_phi
        
            jacobian_basis.set_phi(phi_deg=deg, phi_type=phi_type)
            coeff_mat = jacobian_basis.get_coefficient_matrix(sample_joints, sample_jacobians)

            output_name = f"{robot}-{camera_setup}-{phi_type_name}-{deg}-{num_trajs_sample}-{num_pnts_per_traj_sample}-{num_trajs_eval}-{num_pnts_per_traj_eval}-{random_seed}.png"
            avg_error = jacobian_basis.evaluate_goodness_of_fit(coeff_mat,num_trajs_eval,num_pnts_per_traj_eval,rng,output_folder=output_folder, output_name=output_name)

main()
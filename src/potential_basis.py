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
from robot_toolbox.create_uvs import UVS, analytic_cameras
from robot_toolbox.dh_robot import DHSympyParams
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

sin=np.sin
cos=np.cos

P = DHSympyParams()

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
        self.linearized_params_jacobian_regressor_matrix = None
        logger.info(f"m: {self.m}, phi_degree: {self.phi_degree}, n: {self.n}")

    def set_phi(self, phi_deg, phi_type):
        self.phi_degree = phi_deg
        self.phi=phi_type
        self.K :int = (self.phi(np.array([[0]*self.m])).shape[1]) 

    def kinematic_structure_phi(self, q : np.array):
        '''
        suppose Jacobian(q)= N(q)/D(q). 
        Assume D(q) is constant.
        '''
        pass

    def custom_phi(self,q:np.array):
        '''
        custom phi: specify the basis for each entry of the jacobian.

        '''
        #TODO


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

        Then if we want to solve the Jacobian for any arbitrary new joint q:
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



    def evaluate_goodness_of_fit(self, coefficient_matrix:np.ndarray, joint_configs, jacobians, output_folder = 'test', output_name='test'):
        '''
        Evaluate goodness of fit of regression model (given coefficient matrix) over a series of random joint trajectories.
        '''

        total_error = 0.0
        colors=[]
        for q, J_true in zip(joint_configs, jacobians):
                J_approximated = self.phi([np.array(q)]) @ coefficient_matrix # (1,K) @ (K, (m*n))
                # the commented out lines below are from jan 9 2026, testing the reduced jacobian.
                # q0=q[0]
                # q1=q[1]

                # J_approximated = np.array([ 0.49916708323414*sin(q0)*cos(q1) + 0.499167083234143*sin(q0) + 0.499167083234144*sin(q1)*cos(q0), 0.499167083234141*sin(q0)*cos(q1) + 0.499167083234139*sin(q1)*cos(q0), 0.499167083234142*sin(q0)*sin(q1) - 0.499167083234145*cos(q0)*cos(q1) - 0.499167083234133*cos(q0), 0.499167083234141*sin(q0)*sin(q1) - 0.499167083234143*cos(q0)*cos(q1) ])

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
            
        

def symbolic_trigonometric_phi(dof, degree):
    """
    Returns a list of symbolic basis terms corresponding
    to trigonometric_phi + PolynomialFeatures(degree)
    """

    # symbolic joints
    q = sp.symbols(f'q0:{dof}')  # (q0, q1, ..., q_{dof-1})

    # sin/cos stack (same structure as numeric code)
    sin_q = sp.Matrix([sp.sin(qi) for qi in q])
    cos_q = sp.Matrix([sp.cos(qi) for qi in q])

    trig_q = sp.Matrix.vstack(sin_q, cos_q)

    # sklearn polynomial feature structure
    poly = PolynomialFeatures(degree, include_bias=True)

    # dummy numeric input JUST to get powers_
    dummy = np.zeros((1, trig_q.shape[0]))
    poly.fit(dummy)

    powers = poly.powers_  # shape: (K, 2*dof)

    # build symbolic monomials
    basis = []
    for row in powers:
        term = 1
        for base, power in zip(trig_q, row):
            if power != 0:
                term *= base**power
        basis.append(sp.simplify(term))

    return basis

def symbolic_trigonometric_phi(dof, degree):
    """
    Returns a list of symbolic basis terms corresponding
    to trigonometric_phi + PolynomialFeatures(degree)
    """

    # symbolic joints
    q = sp.symbols(f'q0:{dof}')  # (q0, q1, ..., q_{dof-1})

    # sin/cos stack (same structure as numeric code)
    sin_q = sp.Matrix([sp.sin(qi) for qi in q])
    cos_q = sp.Matrix([sp.cos(qi) for qi in q])

    trig_q = sp.Matrix.vstack(sin_q, cos_q)

    # sklearn polynomial feature structure
    poly = PolynomialFeatures(degree, include_bias=True)

    # dummy numeric input JUST to get powers_
    dummy = np.zeros((1, trig_q.shape[0]))
    poly.fit(dummy)

    powers = poly.powers_  # shape: (K, 2*dof)

    # build symbolic monomials
    basis = []
    for row in powers:
        term = 1
        for base, power in zip(trig_q, row):
            if power != 0:
                term *= base**power
        basis.append(term)

    return basis


def symbolic_polynomial_phi(dof, degree):
    """
    Returns a list of symbolic basis terms corresponding
    to PolynomialFeatures(degree)
    """

    # symbolic joints
    q = sp.symbols(f'q0:{dof}')  # (q0, q1, ..., q_{dof-1})
    q_vec = sp.Matrix([qi for qi in q])
 

    # sklearn polynomial feature structure
    poly = PolynomialFeatures(degree, include_bias=True)

    # dummy numeric input JUST to get powers_
    dummy = np.zeros((1, q_vec.shape[0]))
    poly.fit(dummy)

    powers = poly.powers_  # shape: (K, 2*dof)

    # build symbolic monomials
    basis = []
    for row in powers:
        term = 1
        for base, power in zip(q, row):
            if power != 0:
                term *= base**power
        basis.append(term)

    return basis


def main():
    '''
    Usage example: `python3 ./src/potential_basis.py dof2 0,1 10 3 100 10 1,2,3 0,1 888 modularity_test`

    dof2 : robot type
    0,1 : use camera 0 and camera 1
    10 : number of trajectories to use in the sample 
    3: number of points along each trajectory
        NOTE: the total number of points sampled is then 10x3=30
    100 : the number of trajectories in the evaluation 
    10 : the number of points per trajectory to evaluate
    1,2,3 : run the test on the polynomial/trig basis with degree 1, then 2, then 3
    0,1 : 0 == polynomial_phi, 1 == trigonometric_phi. run the test with the polynomial basis with the degrees specified above. then test with trig basis with the degrees specified above.
        NOTE: number of output tests is 3 x 2, since there are 3 degrees for 2 different types of basis to test on.
    '''

    params = sys.argv
    for i in range(0,len(params)):
        try:
            params[i] = int(params[i])
        except:
            pass
    _, robot, camera_setup_str, num_trajs_sample, num_pnts_per_traj_sample, num_trajs_eval, num_pnts_per_traj_eval, phi_degrees_str, phi_types_str, random_seed, output_folder= params

    camera_setup = [int(i) for i in str(camera_setup_str).split(',')]
    phi_degrees = [int(i) for i in str(phi_degrees_str).split(',')]
    phi_types = [int(i) for i in str(phi_types_str).split(',')] 
    random_seed = int(random_seed) if int(random_seed) >=0 else None

    logger.info(f"robot: {robot}\ncamera_setup: {camera_setup}\nnum_trajs_sample: {num_trajs_sample}\nnum_pnts_per_traj_sample: {num_pnts_per_traj_sample}\nnum_trajs_eval: {num_trajs_eval}\nnum_pnts_per_traj_eval: {num_pnts_per_traj_eval}\nrandom_seed: {random_seed}\noutput_folder:{output_folder}")

    uvs = UVS(robot, camera_setup) 
    jacobian_basis = JacobianBasis(uvs)
    rng = np.random.default_rng(seed=random_seed)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

        # jacobian_basis.get_approximate_jacobian_from_regression_model(coefficient_matrix=coeff_mat) #user io test for individual points
    evaluated_errors = []

    sample_joints, sample_jacobians = jacobian_basis.collect_data(num_trajectories=num_trajs_sample,num_pnts_per_traj=num_pnts_per_traj_sample,rng=rng)
    test_joint_configs, test_jacobians = jacobian_basis.collect_data(num_trajectories=num_trajs_eval, num_pnts_per_traj=num_pnts_per_traj_eval, rng=np.random.default_rng(seed=random_seed+1))
       
    np.save(output_folder+'/sample_joints', sample_joints, allow_pickle=True)
    np.save(output_folder+'/sample_jacobians', sample_jacobians, allow_pickle=True)
    np.save(output_folder+'/eval_joints', test_joint_configs, allow_pickle=True)
    np.save(output_folder+'/eval_jacobians', test_jacobians, allow_pickle=True)

    path  = os.path.join(output_folder, output_folder)
    with open(path, 'w') as f:
        
        for type in phi_types:
            for deg in phi_degrees:
                if type==0:
                    phi_type_name = 'polynomial_phi'
                    phi_type=jacobian_basis.polynomial_phi
                    
                    basis = symbolic_polynomial_phi(uvs.dof, deg)


                if type==1:
                    phi_type_name = 'trigonometric_phi'
                    phi_type=jacobian_basis.trigonometric_phi

                    basis = symbolic_trigonometric_phi(uvs.dof, deg)

                if type==2:
                    phi_type_name = 'kinematic_structure_phi'
                    phi_type = jacobian_basis.kinematic_structure_phi

                logger.info("Symbolic basis:")
                for i, b in enumerate(basis):
                    logger.info(f"phi[{i}] = {b}")


                jacobian_basis.set_phi(phi_deg=deg, phi_type=phi_type)
                coeff_mat = jacobian_basis.get_coefficient_matrix(sample_joints, sample_jacobians)

                output_name = f"{robot}-{camera_setup_str}-{phi_type_name}-{deg}-{num_trajs_sample}-{num_pnts_per_traj_sample}-{num_trajs_eval}-{num_pnts_per_traj_eval}-{random_seed}.png"
                avg_error = jacobian_basis.evaluate_goodness_of_fit(coeff_mat,test_joint_configs,test_jacobians,output_folder=output_folder, output_name=output_name)
                evaluated_errors.append(avg_error)

                J_model = (basis @ coeff_mat).reshape(jacobian_basis.n, jacobian_basis.m).tolist()

                f.write(str(avg_error) +' '+output_name+' '+ str(J_model) + str(basis )+'\n')

    


main()
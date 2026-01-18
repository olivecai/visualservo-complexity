'''
Jan 17 2025

Use SINDy and AIC to find the ideal model for the forward kinematics of our robots.

Terminology:
LIBRARY ==> the terms that compose the basis 

'''
from sklearn.preprocessing import PolynomialFeatures
from robot_toolbox.create_uvs import robot, analytic_cameras
from robot_toolbox.dh_robot import DHSympyParams, DenavitHartenbergAnalytic
import matplotlib.pyplot as plt

from basis import Basis

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

# create a basis to approximate the robot function
class ForwardKinematics:
    def __init__(self, robot: DenavitHartenbergAnalytic):

        self.robot = robot
        '''
        User instructions:

        Initialize a ForwardKinematics object (TRUE REAL WORLD FORWARD KIN, NO CAMERA PROJECTION)
        set; collect data, get the coefficient matrix, then validate that model.
        
        '''
    
        self.m : int = self.robot.dof #dof
        self.phi_degree = None
        self.n : int = 3 #cartesian real world

        self.basis_obj_list = []
        for i in range(self.n): #x,y,z each gets its own basis
            new_entry= Basis(name = f"{i}")
            self.basis_obj_list.append(new_entry)
            
        logger.info(f"m: {self.m}, phi_degree: {self.phi_degree}, n: {self.n}")

    def set_phi(self, phi_deg, phi_type):
        self.phi_degree = phi_deg
        self.phi=phi_type
        self.K :int = (self.phi(np.array([[0]*self.m])).shape[1]) 

    def collect_data(self, num_trajectories=50, num_pnts_per_traj=5, rng=None):
        '''
        Collect data for fitting the basis function.
        num_samples: number of samples to collect
        q_limits: list of tuples specifying the min and max for each joint variable
        '''
        num_samples = num_trajectories * num_pnts_per_traj
        logger.info(f"Collecting {num_samples} samples for basis function fitting...")
        # first we collect sample joint-fkin pairs to fit our basis.
        joint_configs = self.generate_joint_samples_via_trajectory(num_trajectories, num_pnts_per_traj, rng=rng)
        fkin_data = self.generate_fkins_given_joint_configs(joint_configs)
        return joint_configs, fkin_data
        
    def generate_joint_samples_via_trajectory(self, num_trajectories, num_pnts_per_traj, rng=None):
        '''
        Reproducible random generator-based sampling.
        '''
        if rng is None:
            rng = np.random.default_rng()

        joint_configs = []

        for _ in range(num_trajectories):
            start_q = np.array([rng.uniform(low, high) 
                                for (low, high) in self.robot.jointlimits])
            end_q = np.array([rng.uniform(low, high) 
                            for (low, high) in self.robot.jointlimits])

            for t in np.linspace(0, 1, num_pnts_per_traj):
                q = (1 - t) * start_q + t * end_q
                joint_configs.append(q)

        return joint_configs

    def generate_fkins_given_joint_configs(self, joint_configs):
        '''
        TO BE USED IN "collect_data"

        Given a list of joint configurations, compute the corresponding fkins using robot central differences.
        '''
        fkins = []
        for q in joint_configs:
            fkin = self.robot.fkin_eval(q)
            fkins.append(fkin)
        return fkins
            


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
    Usage:
    python3 forward_kinematics.py dof2 10 3 100 10 1,2,3 0,1 888 results
    '''

    params = sys.argv
    for i in range(len(params)):
        try:
            params[i] = int(params[i])
        except:
            pass

    (
        _,
        robot,
        num_trajs_sample,
        num_pnts_per_traj_sample,
        num_trajs_eval,
        num_pnts_per_traj_eval,
        phi_degrees_str,
        phi_types_str,
        activation_threshold,
        random_seed,
        output_folder,
    ) = params

    phi_degrees = [int(i) for i in str(phi_degrees_str).split(',')]
    phi_types = [int(i) for i in str(phi_types_str).split(',')]
    rng = np.random.default_rng(random_seed)

    logger.info(
        f"\nrobot={robot}"
        f"\nsample_trajs={num_trajs_sample}"
        f"\nsample_pts={num_pnts_per_traj_sample}"
        f"\neval_trajs={num_trajs_eval}"
        f"\neval_pts={num_pnts_per_traj_eval}"
        f"\nphi_degrees={phi_degrees}"
        f"\nphi_types={phi_types}"
        f"\nactivation_threshold={activation_threshold}"
        f"\nseed={random_seed}"
        f"\nout={output_folder}"
    )

    os.makedirs(output_folder, exist_ok=True)

    robot = DenavitHartenbergAnalytic(robot)
    fkin_basis = ForwardKinematics(robot)

    # -------------------------
    # Data collection (ONCE)
    # -------------------------
    train_joints, train_fkins = fkin_basis.collect_data(
        num_trajectories=num_trajs_sample,
        num_pnts_per_traj=num_pnts_per_traj_sample,
        rng=rng
    )

    eval_joints, eval_fkins = fkin_basis.collect_data(
        num_trajectories=num_trajs_eval,
        num_pnts_per_traj=num_pnts_per_traj_eval,
        rng=np.random.default_rng(random_seed + 1)
    )

    np.save(f"{output_folder}/train_joints.npy", train_joints, allow_pickle=True)
    np.save(f"{output_folder}/train_fkins.npy", train_fkins, allow_pickle=True)
    np.save(f"{output_folder}/eval_joints.npy", eval_joints, allow_pickle=True)
    np.save(f"{output_folder}/eval_fkins.npy", eval_fkins, allow_pickle=True)
  

    for phi_type in phi_types:
        for deg in phi_degrees:
            output_name = f"{robot}-{phi_type}-{deg}-{num_trajs_sample}-{num_pnts_per_traj_sample}-{num_trajs_eval}-{num_pnts_per_traj_eval}-{activation_threshold}-{random_seed}.txt"
            results_path = os.path.join(output_folder, output_name)
            with open(results_path, "w") as f:

                phi_name = "poly" if phi_type == 0 else "trig"
                logger.info(f"\n=== Running {phi_name} basis, degree={deg} ===")

                # -------------------------
                # Setup basis
                # -------------------------
                fkin_basis.set_activation_threshold(activation_threshold)
                fkin_basis.set_phi(phi_deg=deg, phi_type=phi_type)
                f.write(f"Robot: {robot}\n")
                f.write(f"=== {phi_name} basis, degree={deg} ===\n")
                f.write(f"Initial basis elements per Jacobian entry: {fkin_basis.jacobian_entry_basis_objects[0].symbolic_basis}\n")

                # -------------------------
                # Train
                # -------------------------
                f.write("TRAINING WITH FULL BASIS:")
                fkin_basis.train(train_fkins, train_joints)
                for entry in fkin_basis.basis_obj:
                    f.write(
                        f"  J[{entry.name}]: "
                        f"number of basis elements={entry.number_of_basis_elements}, "
                        f"{entry.symbolic_basis} "
                        f"weights={entry.weights.tolist()}\n"

                    )

                # -------------------------
                # Evaluate (before reduction)
                # -------------------------
                _, rmse_before = fkin_basis.evaluate(eval_fkins, eval_joints)

                # -------------------------
                # Reduce + retrain
                # -------------------------
                f.write("TRAINING WITH REDUCED BASIS:")
                fkin_basis.reduce_basis()
                fkin_basis.train(train_jacobians, train_joints)

                _, rmse_after = fkin_basis.evaluate(eval_jacobians, eval_joints)

                # -------------------------
                # Log symbolic structure
                # -------------------------

                f.write(
                    f"{phi_name}, deg={deg}, "
                    f"rmse_before={rmse_before:.4e}, "
                    f"rmse_after={rmse_after:.4e}\n"
                )

                for entry in fkin_basis.jacobian_entry_basis_objects:
                    f.write(
                        f"  J[{entry.name}]: "
                        f"number of basis elements={entry.number_of_basis_elements}, "
                        f"{entry.symbolic_basis} "
                        f"weights={entry.weights.tolist()}\n"

                    )

                f.write("\n")

    logger.info("Done.")


def main():
    '''
    Usage example: `python3 ./src/forward_kinematics.py dof2 10 3 100 10 1,2,3 0,1 888 modularity_test`

    dof2 : robot type
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
    _, robot, num_trajs_sample, num_pnts_per_traj_sample, num_trajs_eval, num_pnts_per_traj_eval, phi_degrees_str, phi_types_str, random_seed, output_folder= params

    phi_degrees = [int(i) for i in str(phi_degrees_str).split(',')]
    phi_types = [int(i) for i in str(phi_types_str).split(',')] 
    random_seed = int(random_seed) if int(random_seed) >=0 else None

    logger.info(f"robot: {robot}\nnum_trajs_sample: {num_trajs_sample}\nnum_pnts_per_traj_sample: {num_pnts_per_traj_sample}\nnum_trajs_eval: {num_trajs_eval}\nnum_pnts_per_traj_eval: {num_pnts_per_traj_eval}\nrandom_seed: {random_seed}\noutput_folder:{output_folder}")

    robot = DenavitHartenbergAnalytic(robot) 
    fkin_basis = ForwardKinematics(robot)
    rng = np.random.default_rng(seed=random_seed)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

        # fkin_basis.get_approximate_fkin_from_regression_model(coefficient_matrix=coeff_mat) #user io test for individual points
    evaluated_errors = []

    sample_joints, sample_fkins = fkin_basis.collect_data(num_trajectories=num_trajs_sample,num_pnts_per_traj=num_pnts_per_traj_sample,rng=rng)
    test_joint_configs, test_fkins = fkin_basis.collect_data(num_trajectories=num_trajs_eval, num_pnts_per_traj=num_pnts_per_traj_eval, rng=np.random.default_rng(seed=random_seed+1))
       
    np.save(output_folder+'/sample_joints', sample_joints, allow_pickle=True)
    np.save(output_folder+'/sample_fkins', sample_fkins, allow_pickle=True)
    np.save(output_folder+'/eval_joints', test_joint_configs, allow_pickle=True)
    np.save(output_folder+'/eval_fkins', test_fkins, allow_pickle=True)

    path  = os.path.join(output_folder, output_folder)
    with open(path, 'w') as f:
        
        for type in phi_types:
            for deg in phi_degrees:
                if type==0:
                    phi_type_name = 'polynomial_phi'
                    phi_type=fkin_basis.polynomial_phi
                    
                    basis = symbolic_polynomial_phi(robot.dof, deg)


                if type==1:
                    phi_type_name = 'trigonometric_phi'
                    phi_type=fkin_basis.trigonometric_phi

                    basis = symbolic_trigonometric_phi(robot.dof, deg)

                if type==2:
                    phi_type_name = 'kinematic_structure_phi'
                    phi_type = fkin_basis.kinematic_structure_phi

                logger.info("Symbolic basis:")
                for i, b in enumerate(basis):
                    logger.info(f"phi[{i}] = {b}")


                fkin_basis.set_phi(phi_deg=deg, phi_type=phi_type)
                coeff_mat = fkin_basis.get_coefficient_matrix(sample_joints, sample_fkins)

                output_name = f"{robot}-{camera_setup_str}-{phi_type_name}-{deg}-{num_trajs_sample}-{num_pnts_per_traj_sample}-{num_trajs_eval}-{num_pnts_per_traj_eval}-{random_seed}.png"
                avg_error = fkin_basis.evaluate_goodness_of_fit(coeff_mat,test_joint_configs,test_fkins,output_folder=output_folder, output_name=output_name)
                evaluated_errors.append(avg_error)

                fkin_model = (basis @ coeff_mat).reshape(fkin_basis.n, fkin_basis.m).tolist()

                f.write(str(avg_error) +' '+output_name+' '+ str(fkin_model) + str(basis )+'\n')

    


main()
'''
Jan 17 2025

Use SINDy and AIC to find the ideal model for the forward kinematics of our robots.

Terminology:
LIBRARY ==> the terms that compose the basis 

'''
from sklearn.preprocessing import PolynomialFeatures
from robot_toolbox.create_uvs import UVS, analytic_cameras
from robot_toolbox.dh_robot import DHSympyParams, DenavitHartenbergAnalytic
from robot_toolbox.kinematic_structure import get_kinematic_structure
import matplotlib.pyplot as plt

from itertools import combinations, product

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



def generate_symbolic_library_multiply(q, max_order=2, include_constant=True, primitive_sympy_functions=[sp.sin, sp.cos]):
    """
    q: list of sympy symbols [q0, q1, ..., qn]
    """
    primitives = []
    if len(primitive_sympy_functions)>0:
        for qi in q:
            for p in primitive_sympy_functions:
                primitives.append(p(qi))
    # elif len(primitive_sympy_functions)==0:
    #     # then do polynomials only
    #     for order in range(1, max_order + 1):
    #         for qi in q:
    #             primitives.append(qi**order)

    library = []

    if include_constant:
        library.append(sp.Integer(1))

    # single terms
    library.extend(primitives)

    # interaction terms
    for order in range(2, max_order + 1):
        for combo in combinations(primitives, order):
            library.append(sp.Mul(*combo))

    logger.info(f"Generated symbolic library:{library}")
    logger.info(f"Library size: {len(library)}")

    return library #type: list[sp.Expr]

def generate_symbolic_library_additive(q, max_order=2, include_constant=True, primitive_sympy_functions=[sp.sin, sp.cos]):
    '''
    for (q0,q1), generates [sin(q0), cos(q0), sin(q1), cos(q1), sin(q0+q1), cos(q0+q1)]
    '''
    
    library = []

    
    library.append(sp.Integer(1))

    # single terms

    # create the additive functions
    for order in range(1, len(q)+1):
        for combo in combinations(q, order):
            for p in primitive_sympy_functions:
                library.append(p(sp.Add(*combo)))

    return library



# create a basis to approximate the robot function
class ForwardKinematics:
    def __init__(self, robot: DenavitHartenbergAnalytic, name=None, activation_threshold = 0.1):

        self.robot = robot

        '''
        User instructions:

        Initialize a ForwardKinematics object (TRUE REAL WORLD FORWARD KIN, NO CAMERA PROJECTION)
        set; collect data, get the coefficient matrix, then validate that model.

        kinematic_structure = [] if not mode 2
        if mode 2, then the robot's actual kinematic structure will be captured from the basis
        
        '''
        self.m : int = 1 #dof
        self.dof : int = self.robot.dof
        self.phi_degree = None
        self.n : int = 3 #cartesian real world
        self.params = [sp.symbols(f"q{i}") for i in range(self.dof)]


        self.basis_obj_list = []
        for i in range(self.n): #x,y,z each gets its own basis
            new_entry= Basis(name = f"{i}")
            if name:
                print("KINEMATIC STRUCTURE INDEXED AT i:",self.get_kinematic_structure(name)[i])
                new_entry.setup(params=self.params, activation_threshold=activation_threshold, symbolic_basis_expression=self.get_kinematic_structure(name)[i])

            self.basis_obj_list.append(new_entry)
        for i in range(self.n):
            print("num basis element", self.basis_obj_list[i].number_of_basis_elements)
            print("sym basis", self.basis_obj_list[i].symbolic_basis)


        logger.info(f"m: {self.m}, phi_degree: {self.phi_degree}, n: {self.n}")

    def get_kinematic_structure(self, name):
        q0 = self.params[0]
        q1 = self.params[1]

        #true
        dof2 = [[sp.sin(self.params[0])*sp.sin(self.params[1]), sp.cos(self.params[0])*sp.cos(self.params[1]), sp.cos(self.params[0])], [sp.sin(self.params[0])*sp.cos(self.params[1]), sp.sin(self.params[0]), sp.sin(self.params[1])*sp.cos(self.params[0])], []]
        if name == 'dof2':
            return dof2

    def set_phi(self,phi_deg,phi_type, activation_threshold, primitive_sympy_functions=[]):
        '''
        initialize the basis functions for each Jacobian entry
        phi_type: 0 for polynomial, 1 for trigonometric
        phi_deg: degree of the basis functions

        Call after initialization and before collect_data.
        Sets ALL entries to be the SAME
        '''

        self.phi_degree = phi_deg

        if primitive_sympy_functions==[]:
            if phi_type == 0:
                primitive_sympy_functions = []
            elif phi_type == 1 or phi_type ==3:
                primitive_sympy_functions = [sp.sin, sp.cos]
    
        if phi_type<=1:
            phi_func = generate_symbolic_library_multiply(
                q=self.params,
                max_order=phi_deg,
                include_constant=True,
                primitive_sympy_functions=primitive_sympy_functions)
        if phi_type==3:
            phi_func  = generate_symbolic_library_additive(
                q=self.params,
                max_order=phi_deg,
                include_constant=True,
                primitive_sympy_functions=primitive_sympy_functions)
        
            
        for entry in self.basis_obj_list:
            entry.setup(params=self.params, activation_threshold=activation_threshold, symbolic_basis_expression=phi_func)

    def collect_data(self, num_trajectories=50, num_pnts_per_traj=5, rng :np.random.Generator =None, add_input_noise=0.0, add_output_noise=0.05):
        '''
        Collect data for fitting the basis function.
        num_samples: number of samples to collect
        q_limits: list of tuples specifying the min and max for each joint variable
        '''
        num_samples = num_trajectories * num_pnts_per_traj
        logger.info(f"Collecting {num_samples} with add_output_noise={add_output_noise}, add_input_noise={add_input_noise}")

        joint_configs = self.generate_joint_samples_via_trajectory(num_trajectories, num_pnts_per_traj, rng=rng)
        fkin_data = self.generate_fkins_given_joint_configs(joint_configs)

        joint_configs = [(q + rng.normal(0, add_input_noise, len(q))).tolist() for q in joint_configs]
        fkin_data = [(fkin + rng.normal(0, add_output_noise, np.array(fkin).shape)).tolist() for fkin in fkin_data]
        logger.info(f"First 3 joint configs: {joint_configs[:5]}\nFirst 3 fkin data: {fkin_data[:5]}")
        return joint_configs, fkin_data
        
    def generate_joint_samples_via_trajectory(self, num_trajectories, num_pnts_per_traj, rng=None):
        '''
        TO BE USED IN "collect_data"

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
                joint_configs.append(q.tolist())

        return joint_configs

    def generate_fkins_given_joint_configs(self, joint_configs):
        '''
        TO BE USED IN "collect_data"

        Given a list of joint configurations, compute the corresponding fkins using robot central differences.
        '''
        fkins = []
        for q in joint_configs:
            fkin = self.robot.fkin_eval(*q)
            fkin=fkin.reshape((3,))
            fkins.append(fkin.tolist())
        return np.array(fkins)
    
    def train(self, train_pos, train_joints):
        pos_matrix = np.array(train_pos)
        for i in range(self.n):
            for j in range(self.m):
                pos_entry_i_j : Basis = self.basis_obj_list[i*self.m+j]
                symbolic_basis_elements, weights = pos_entry_i_j.train(pos_matrix[:,i*self.m+j], train_joints)

    def evaluate(self, eval_pos, eval_joints):
        '''
        Evaluate the forward kinematics model on evaluation data.
        Returns total_rss, total RMSE across all entries, total_count of params
        '''
        rss_list = []
        total_rss = 0.0
        count_list= []
        total_count = 0
        eval_pos=np.array(eval_pos)
        for i in range(self.n):
            for j in range(self.m):
                pos_entry_i_j : Basis = self.basis_obj_list[i*self.m+j]
                # print(eval_pos)
                # print(eval_pos[:,i*self.m+j])
                rss = pos_entry_i_j.evaluate(eval_pos[:,i*self.m+j], eval_joints)
                total_rss += rss
                total_count += len(eval_joints)
                rss_list.append(rss)
                count_list.append(len(eval_joints))
        rmse = np.sqrt(total_rss / total_count)
        return total_rss, rmse, total_count, rss_list, count_list

    def reduce_basis(self):
        '''does not recalculate the parameters, only reduces the basis'''
        for i in range(self.n):
            for j in range(self.m):
                pos_entry_i_j : Basis = self.basis_obj_list[i*self.m+j]
                pos_entry_i_j.reduce_basis()

    def sindy(self, train_pos, train_joints, lambda_vals=[]):
        train_pos=np.array(train_pos)

        for i in range(self.n):
            for j in range(self.m):
                pos_entry_i_j :  Basis = self.basis_obj_list[i*self.m+j]
                if lambda_vals:
                    pos_entry_i_j.activation_threshold = lambda_vals[i*self.m+j]
                    print("LOGGING",pos_entry_i_j.activation_threshold)
                _, pos_entry_i_j.weights = pos_entry_i_j.sindy_stlsq(train_pos[:,i*self.m+j], train_joints, lambda_val=pos_entry_i_j.activation_threshold)
        
    def pareto_frontier(self, train_pos, train_joints, eval_pos, eval_joints, lambda_values=np.linspace(0,1, 11), output_folder=None):
        train_pos=np.array(train_pos)
        eval_pos=np.array(eval_pos)

        best_lambdas = []
        
        for i in range(self.n):
            for j in range(self.m):
                if output_folder is not None:
                    os.makedirs(output_folder, exist_ok=True)
                    plotname = f"{output_folder}/pareto_frontier_i_j.png"
                else:
                    plotname = None
                pos_entry_i_j : Basis = self.basis_obj_list[i*self.m+j]
                lambda_val, weights, RSS, num_basis_elements, min_aicc =  pos_entry_i_j.pareto_frontier(train_b=train_pos[:,i*self.m+j], train_a=train_joints,eval_b=eval_pos[:,i*self.m+j], eval_a=eval_joints,lambda_values=lambda_values, plot_name=plotname)
                logger.info(f"Basis Component: {pos_entry_i_j.name}, Best lambda: {lambda_val}, Weights: {weights}, RSS: {RSS}, Num Basis Elements: {num_basis_elements}, Min AICC: {min_aicc}")
                best_lambdas.append(lambda_val)

        return best_lambdas
    
def main():
    '''
    Usage:
    python3 forward_kinematics.py dof2 50 1 100 1 1,2 1 0.1 888 test_jan20
    '''

    params = sys.argv
    for i in range(len(params)):
        try:
            params[i] = int(params[i])
        except:
            pass

    (
        _,
        robot_name,
        train_num_trajs,
        train_num_pnts_per_traj,
        eval_num_trajs,
        eval_num_pnts_per_traj,
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
        f"\nrobot={robot_name}"
        f"\nsample_trajs={train_num_trajs}"
        f"\nsample_pts={train_num_pnts_per_traj}"
        f"\neval_trajs={eval_num_trajs}"
        f"\neval_pts={eval_num_pnts_per_traj}"
        f"\nphi_degrees={phi_degrees}"
        f"\nphi_types={phi_types}"
        f"\nactivation_threshold={activation_threshold}"
        f"\nseed={random_seed}"
        f"\nout={output_folder}"
    )

    os.makedirs(output_folder, exist_ok=True)

    

    robot = UVS(robot_name, cam_idx=[0]).dh_robot # ignore camera for true cartesian real world fkin
    
    linearized_params_basis_flag=None
    if phi_types[0]==2:
        linearized_params_basis_flag = robot_name
    
    fkin_basis = ForwardKinematics(robot, linearized_params_basis_flag)

    # -------------------------
    # Data collection (ONCE)
    # -------------------------
    train_joints, train_fkins = fkin_basis.collect_data(
        num_trajectories=train_num_trajs,
        num_pnts_per_traj=train_num_pnts_per_traj,
        rng=rng,
        add_output_noise=0.05 #note: noise follows normal distribution
    )
    # print("TRAIN JOINTS")
    # print(train_joints)
    # print("TRAIN FKINS")
    # print(train_fkins)

    eval_joints, eval_fkins = fkin_basis.collect_data(
        num_trajectories=eval_num_trajs,
        num_pnts_per_traj=eval_num_pnts_per_traj,
        rng=np.random.default_rng(random_seed + 1)
    )

    np.save(f"{output_folder}/train_joints.npy", train_joints, allow_pickle=True)
    np.save(f"{output_folder}/train_fkins.npy", train_fkins, allow_pickle=True)
    np.save(f"{output_folder}/eval_joints.npy", eval_joints, allow_pickle=True)
    np.save(f"{output_folder}/eval_fkins.npy", eval_fkins, allow_pickle=True)
  
    for phi_type in phi_types:
        for deg in phi_degrees:
            output_name = f"{robot_name}-{phi_type}-{deg}-{train_num_trajs}-{train_num_pnts_per_traj}-{eval_num_trajs}-{eval_num_pnts_per_traj}-{activation_threshold}-{random_seed}"

            ith_output_folder = os.path.join(output_folder, output_name)
            os.makedirs(ith_output_folder, exist_ok=True)

            results_path = os.path.join(output_folder, f"RESULTS-{output_name}.txt")
            with open(results_path, "w") as f:

                if phi_type == 0:
                    phi_name = "poly"
                if phi_type == 1:
                    phi_name = "trig multiplicative"
                if phi_type == 2:
                    phi_name = "linearized_kinematic_structure"
                if phi_type == 3:
                    phi_name = "trig additive"
                    
                logger.info(f"\n=== {phi_name} basis, degree={deg} ===")

                # -------------------------
                # Setup basis
                # -------------------------
                if phi_type !=2:
                    fkin_basis.set_phi(phi_deg=deg, phi_type=phi_type, activation_threshold=float(activation_threshold))
                f.write(f"Robot: {robot_name}\n")
                f.write(f"=== {phi_name} basis, degree={deg} ===\n")
                f.write(f"Initial basis elements per fkin entry: {fkin_basis.basis_obj_list[0].symbolic_basis}\n")
                print(f"Initial basis elements per fkin entry: {fkin_basis.basis_obj_list[0].symbolic_basis}\n")
                # -------------------------
                # Train
                # -------------------------
                f.write("TRAINING WITH FULL BASIS:")
                fkin_basis.train(train_fkins, train_joints)
                for entry in fkin_basis.basis_obj_list:
                    f.write(
                        f"  J[{entry.name}]: "
                        f"number of basis elements={entry.number_of_basis_elements}, "
                        f"{entry.symbolic_basis} "
                        f"weights={entry.weights.tolist()}\n"

                    )

                # -------------------------
                # Evaluate (before reduction)
                # -------------------------
                RMSE_total, RSS_total, param_count_total, rss_list, count_list = fkin_basis.evaluate(eval_fkins, eval_joints)
                f.write(f"Pre-reduction evaluation: RMSE={RMSE_total:.4e}, RSS={RSS_total:.4e}, param_count={param_count_total}\n")
                f.write(f"Per-entry RSS and counts:\n")
                for i in range(fkin_basis.n):
                    for j in range(fkin_basis.m):
                        idx = i * fkin_basis.m + j
                        f.write(f"  entry[{i},{j}]: RSS={rss_list[idx]:.4e}, count={count_list[idx]}\n")

                # -------------------------
                # Reduce + retrain
                # -------------------------
                f.write("PARETO CURVE:")
                best_lambdas = fkin_basis.pareto_frontier(
                    train_pos=train_fkins,
                    train_joints=train_joints,
                    eval_pos=eval_fkins,    
                    eval_joints=eval_joints,
                    lambda_values=np.linspace(0, 1, 11),
                    output_folder=ith_output_folder
                )

                f.write("TRAINING WITH SINDy BASIS:")
                fkin_basis.sindy(train_fkins, train_joints, best_lambdas)

                # -------------------------
                # Evaluate (after sindy)
                # -------------------------
                RMSE_total_SINDY, RSS_total_SINDY, param_count_total_SINDY, rss_list_SINDY, count_list_SINDY = fkin_basis.evaluate(eval_fkins, eval_joints)
                f.write(f"Sindy evaluation: RMSE={RMSE_total_SINDY:.4e}, RSS={RSS_total_SINDY:.4e}, param_count={param_count_total_SINDY}\n")
                f.write(f"Sindy RSS and counts:\n")
                for i in range(fkin_basis.n):
                    for j in range(fkin_basis.m):
                        idx = i * fkin_basis.m + j
                        f.write(f"  entry[{i},{j}]: RSS={rss_list_SINDY[idx]:.4e}, count={count_list_SINDY[idx]}\n")


                # -------------------------
                # Log symbolic structure
                # -------------------------

                f.write(
                    f"{phi_name}, deg={deg}, "
                    f"RMSE_total_full_library={RMSE_total:.4e}, "
                    f"RSS_total_SINDY={RSS_total_SINDY:.4e}\n"
                )

                for entry in fkin_basis.basis_obj_list:
                    f.write(
                        f"Name: [{entry.name}]: "
                        f"number of basis elements={entry.number_of_basis_elements}, "
                        f"{entry.symbolic_basis} "
                        f"weights={entry.weights.tolist()}\n"

                    )

                f.write("\n")

    logger.info("Done.")

if __name__ == '__main__':
    main()
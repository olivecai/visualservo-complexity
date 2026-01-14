'''
Jan 10 2026

Overview: 
Using regression, create a model to approximate the Jacobian given any set of joint angles.
We suppose that for revolute robots, a trigonometric basis can sufficiently model the forward kinematics function; 
and by further assuming that most of the complexity and structure of the jacobian comes from the forward kinematics, 
we can then suppose that this trigonometric basis might sufficiently model the jacobian function.

We have already empirically come across a few interesting behaviors relating to the basis formulations:
1. The products of trigonometric functions basis indeed works far more efficiently than the products of monomials basis; it only takes a few training samples and only a few products. Nonetheless, we still achieve good results with the polynomials.
2. The trigonometric basis is very accurate for the 2 dof, but performance stagnates for the 7 dof.
3. Unsurprisingly, only some basis elements are activated, and removing the trivial components does not significantly change the goodness of fit. 

We now have two new characteristics we would like to implement into our basis model:
1. JACOBIAN ENTRY INDEPENDENCE: The basis model assigned to each entry of the Jacobian can (and should) be unique. 
2. SEQUENTIAL ATTRITION: A computed basis can be 'recomputed' with only activated basis elements as to avoid overfitting and reduce the number of parameters in the final basis.

For instance, the 2 DOF Jacobian in projected space = [[fit_1_1, fit_1_2],[fit_2_1, fit_2_2]]

Then there should be the unique, independent basis for each:
fit_1_1_basis
fit_1_2_basis
fit_2_1_basis
fit_2_2_basis

[fit_1_1_basis, fit_1_2_basis, fit_2_1_basis, fit_2_2_basis].T

Solve for basis_weights in Phi_stack @ basis_weights = J_stack.

Regress over each entry independently by having each entry of the Jacobian being its own object. 
These Basis objects, which in this specific context are Jacobian entries, have attributes:
- name/entry of the jacobian (ie 1_1, 1_2, 2_1, or 2_2, etc. Naming in matrix row_col style)
- basis (ie [sin(t0),cos(t0),sin(t1),cos(t1), sin(t0)*cos(t1)])
- number of elements in the basis (ie 5)
- threshold for magnitude of activated basis component (ie 1e-1)
- phi weights (intialized as list of None, to be set by invoking self.train)
- discarded basis elements (initialized as empty list, but is mutated by reduce_basis. ie if sin(t0) is discarded from basis, then it is noted here)
As well as methods:
- reduce_basis() (if a basis [sin(t0) cos(t0)] has weights [0.01, 5] then reduce basis to [cos(t0)]) and keep tracked of discarded basis elements 
- train(train_jacobians, train_joints) -> return AND set phi_weights. this is where the regression occurs, and train_phi = phi(train_q)
- evaluate(eval_jacobians, eval_joints) -> save error plot, return list of all error
- get_symbolic_basis -> ie returns [cos(t0), sin(t0)]
- get_basis_weights -> ie returns [2.0, 4.5]

Then, the JacobianBasis class can use each Basis to easily construct its Jacobian and evaluate as before.

'''
from sklearn.preprocessing import PolynomialFeatures
from robot_toolbox.create_uvs import UVS, analytic_cameras
from robot_toolbox.dh_robot import DHSympyParams
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

import numpy as np
import sympy as sp
import logging
from itertools import combinations

import os
from datetime import datetime
import sys

from basis import Basis

# Configure basic logging to the console (default level is WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger for the current module
# logger. debug, info, warning, error, critical

sin=np.sin
cos=np.cos

P = DHSympyParams()

def construct_trig_basis(params, degree):
    '''
    Construct a trigonometric basis up to the specified degree.
    For example, for degree=2 and params=[t0, t1], the basis will include:
    [sin(t0), cos(t0), sin(t1), cos(t1), sin(t0)*sin(t1), sin(t0)*cos(t1), cos(t0)*sin(t1), cos(t0)*cos(t1)]
    '''
    print(params)
    basis_elements = []

    num_params = len(params)

    # Single terms
    for d in range(1, degree + 1):
        for i in range(num_params):
            basis_elements.append(sp.sin(d * params[i]))
            basis_elements.append(sp.cos(d * params[i]))

    # Product terms
    for d in range(2, degree + 1):
        for indices in combinations(range(num_params), d):  # Changed from multiset_partitions
            term_sin = 1
            term_cos = 1
            for index in indices:
                term_sin *= sp.sin(params[index])
                term_cos *= sp.cos(params[index])
            basis_elements.append(term_sin)
            basis_elements.append(term_cos)

    # Remove duplicates and create a sympy expression
    unique_basis_elements = list(set(basis_elements))
    symbolic_basis_expression = sum(unique_basis_elements)
    logger.info(f"Constructed trigonometric basis with {len(unique_basis_elements)} elements: {symbolic_basis_expression}")

    return symbolic_basis_expression

def construct_poly_basis(params, degree):
    '''
    Construct a polynomial basis up to the specified degree.
    For example, for degree=2 and params=[t0, t1], the basis will include:
    [1, t0, t1, t0^2, t0*t1, t1^2]
    '''
    poly = PolynomialFeatures(degree)
    # Generate all combinations of parameters up to the specified degree
    param_combinations = poly.fit_transform(np.zeros((1, len(params))))
    
    # Create sympy expressions for each combination
    basis_elements = []
    for comb in param_combinations[0]:
        term = 1
        for i, power in enumerate(comb):
            term *= params[i] ** power
        basis_elements.append(term)
    
    # Remove duplicates and create a sympy expression
    unique_basis_elements = list(set(basis_elements))
    symbolic_basis_expression = sum(unique_basis_elements)

    return symbolic_basis_expression

class JacobianBasis:
    '''
    Uses a Basis object to find each entry of the Jacobian.

    USAGE:
    1. Initialize: jacobian_basis = JacobianBasis(uvs)
    2. Set the phi type and degree: jacobian_basis.set_phi(phi_deg, phi_type)
    3. Collect data: train_joints, train_jacobians = jacobian_basis.collect_data(num_trajectories, num_pnts_per_traj)
    4. Train the basis: jacobian_basis.train(train_jacobians, train_joints)
    5. Optionally reduce the basis: jacobian_basis.reduce_basis()
    6. Evaluate the basis: errors, overall_rmse = jacobian_basis.evaluate(eval_jacobians, eval_joints)
    7. Use the basis to predict Jacobians for new joint configurations.
    '''
    def __init__(self, uvs: UVS):
    
        self.uvs = uvs

        self.activation_threshold = 1e-5 #can be set by user. For dynamic percentiles (like get the 10% lowest values) set as NEGATIVE. (negative acts as a flag)
    
        self.m : int = self.uvs.dof #dof
        self.phi_degree = None
        self.n : int = len(self.uvs.cameras) * 2
        logger.info(f"m: {self.m}, phi_degree: {self.phi_degree}, n: {self.n}")
        self.jacobian_entry_basis_objects: list = []

        # sympy symbols
        self.params = sp.symbols(f'q0:{self.m}')

        # for each entry of the Jacobian, we need a Basis.
        # Naming will be [0,0], [0,1], [0,2], etc
        for i in range(self.n):
            for j in range(self.m):
                new_entry= Basis(name = f"{i}_{j}")
                self.jacobian_entry_basis_objects.append(new_entry)

        self.set_activation_threshold(self.activation_threshold)


        
    def set_activation_threshold(self, threshold):
        '''
        Set the threshold for basis element reduction.
        threshold: float, basis elements with absolute weight below this value will be discarded.
        '''
        self.activation_threshold = threshold
        for element in self.jacobian_entry_basis_objects:
            element.activation_threshold = threshold

    def set_phi(self,phi_deg,phi_type):
        '''
        initialize the basis functions for each Jacobian entry
        phi_type: 0 for polynomial, 1 for trigonometric
        phi_deg: degree of the basis functions

        Call after initialization and before collect_data.
        Sets ALL entries to be the SAME
        '''

        self.phi_degree = phi_deg

        if phi_type == 0:
            phi_func = construct_poly_basis
        elif phi_type == 1:
            phi_func = construct_trig_basis
            
        for entry in self.jacobian_entry_basis_objects:
            entry.setup(params=self.params, activation_threshold=3e-1, symbolic_basis_expression=phi_func(self.params, degree=self.phi_degree))


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
    
    def train(self, train_jacobians, train_joints):
        train_jacobians_vectors = [J.flatten() for J in train_jacobians]
        train_jacobian_matrix = np.array(train_jacobians_vectors)
        for i in range(self.n):
            for j in range(self.m):
                jac_entry_i_j : Basis = self.jacobian_entry_basis_objects[i*self.m+j]
                symbolic_basis_elements, weights = jac_entry_i_j.train(train_jacobian_matrix[:,i*self.m+j], train_joints)


    def reduce_basis(self):
        '''does not recalculate the parameters, only reduces the basis'''
        for i in range(self.n):
            for j in range(self.m):
                jac_entry_i_j : Basis = self.jacobian_entry_basis_objects[i*self.m+j]
                jac_entry_i_j.reduce_basis()

    def evaluate(self, eval_jacobians, eval_joints):
        eval_jacobians_vectors = [J.flatten() for J in eval_jacobians]
        eval_jacobian_matrix = np.array(eval_jacobians_vectors)
        all_errors = {} # dict of errors for each entry, key is "i_j", value is LIST of errors
        for i in range(self.n):
            for j in range(self.m):
                jac_entry_i_j : Basis = self.jacobian_entry_basis_objects[i*self.m+j]
                errors = jac_entry_i_j.evaluate(eval_jacobian_matrix[:,i*self.m+j], eval_joints)
                all_errors[f"{i}_{j}"] = errors
        # get overall RMSE (root mean squared error) across all entries
        total_squared_error = 0.0
        total_count = 0
        for errors in all_errors.values():
            total_squared_error += np.sum(errors**2)
            total_count += len(errors)
        overall_rmse = np.sqrt(total_squared_error / total_count)
        logger.info(f"Overall RMSE across all Jacobian entries: {overall_rmse}")
        return all_errors, overall_rmse
    
    def symbolic_basis(self):
        '''
        Return the symbolic basis for each Jacobian entry as a MATRIX.
        '''
        basis_matrix = []
        for i in range(self.n):
            row = []
            for j in range(self.m):
                jac_entry_i_j : Basis = self.jacobian_entry_basis_objects[i*self.m+j]
                row.append(jac_entry_i_j.symbolic_basis)
            basis_matrix.append(row)
        return basis_matrix


def verify_basis_object():
    '''
    Just used for quick verification of Basis class functionality.
    '''

    q0, q1 = sp.symbols('q0 q1')

    expr = sp.sin(q0) + sp.cos(q1)
    logger.info(f"Expression: {expr}")
    b = Basis("0_0")
    b.setup(activation_threshold=5e-1, params=[q0, q1], symbolic_basis_expression=expr)

    b.weights = np.array([2.0, 0.0005])

    # generate some fake data to test training. given  expr = sp.sin(q0) + 0.001*sp.cos(q1) + sp.sin(q0)*sp.cos(q1), 
    # generate dummy data with weights [2.0, 0.0005, 1.3] and add noise:
    rng = np.random.default_rng(42)
    num_samples = 100
    train_joints = rng.uniform(low=-np.pi, high=np.pi, size=(num_samples, 2))
    # print(train_joints)
    train_jacobian_entry = []
    for q in train_joints:
        true_value = 2.0 * np.sin(q[0]) + 0.0005 * np.cos(q[1]) 
        noise = rng.normal(loc=0.0, scale=0.01)
        train_jacobian_entry.append(true_value + noise)
    train_jacobian_entry = np.array(train_jacobian_entry)
    b.lasso_regression(train_jacobian_entry, train_joints)
    print(f"Trained LASSO weights: {b.weights}")
    b.train(train_jacobian_entry, train_joints)
    print(f"Trained LS weights: {b.weights}")
    b.reduce_basis()
    print(f"Reduced basis symbolic elements: {b.symbolic_basis}")
    print(f"Reduced basis weights: {b.weights}")
    b.train(train_jacobian_entry, train_joints)
    print(f"Retrained weights after reduction: {b.weights}")

    # now evaluate:
    eval_joints = rng.uniform(low=-np.pi, high=np.pi, size=(20, 2))
    eval_jacobian_entry = []
    for q in eval_joints:
        true_value = 2.0 * np.sin(q[0]) + 0.0005 * np.cos(q[1]) 
        noise = rng.normal(loc=0.0, scale=0.01)
        eval_jacobian_entry.append(true_value + noise)
    eval_jacobian_entry = np.array(eval_jacobian_entry)
    errors = b.evaluate(eval_jacobian_entry, eval_joints)
    print(f"Evaluation errors: {errors}")

def verify_jacobian_basis():
    robot = 'dof2'
    camera_setup=[0,1]
    uvs= UVS(robot, camera_setup)
    jacobian_basis = JacobianBasis(uvs)
    rng = np.random.default_rng(888)
    jacobian_basis.set_phi(phi_deg=2, phi_type=1) #trig basis of degree 2

    train_joints, train_jacobians = jacobian_basis.collect_data(num_trajectories=30, num_pnts_per_traj=1, rng=rng)
    eval_joints, eval_jacobians = jacobian_basis.collect_data(num_trajectories=10, num_pnts_per_traj=1, rng=rng)
    
    jacobian_basis.train(train_jacobians, train_joints)
    errors, overall_rmse = jacobian_basis.evaluate(eval_jacobians, eval_joints)
    print(f"Basis: {jacobian_basis.symbolic_basis}")

    print(f"Overall RMSE: {overall_rmse}")
    print("Reducing basis...")
    jacobian_basis.reduce_basis()

    jacobian_basis.train(train_jacobians, train_joints)
    jacobian_basis.evaluate(eval_jacobians, eval_joints)
    print(f"Basis: {jacobian_basis.symbolic_basis}")
    print(f"Overall RMSE: {overall_rmse}")

    



def main():
    '''
    Usage:
    python3 basis_by_attrition.py dof2 0,1 10 3 100 10 1,2,3 0,1 888 results
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
        camera_setup_str,
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

    camera_setup = [int(i) for i in str(camera_setup_str).split(',')]
    phi_degrees = [int(i) for i in str(phi_degrees_str).split(',')]
    phi_types = [int(i) for i in str(phi_types_str).split(',')]
    rng = np.random.default_rng(random_seed)

    logger.info(
        f"\nrobot={robot}"
        f"\ncameras={camera_setup}"
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

    uvs = UVS(robot, camera_setup)
    jacobian_basis = JacobianBasis(uvs)

    # -------------------------
    # Data collection (ONCE)
    # -------------------------
    train_joints, train_jacobians = jacobian_basis.collect_data(
        num_trajectories=num_trajs_sample,
        num_pnts_per_traj=num_pnts_per_traj_sample,
        rng=rng
    )

    eval_joints, eval_jacobians = jacobian_basis.collect_data(
        num_trajectories=num_trajs_eval,
        num_pnts_per_traj=num_pnts_per_traj_eval,
        rng=np.random.default_rng(random_seed + 1)
    )

    np.save(f"{output_folder}/train_joints.npy", train_joints, allow_pickle=True)
    np.save(f"{output_folder}/train_jacobians.npy", train_jacobians, allow_pickle=True)
    np.save(f"{output_folder}/eval_joints.npy", eval_joints, allow_pickle=True)
    np.save(f"{output_folder}/eval_jacobians.npy", eval_jacobians, allow_pickle=True)


  

    for phi_type in phi_types:
        for deg in phi_degrees:
            output_name = f"{robot}-{camera_setup_str}-{phi_type}-{deg}-{num_trajs_sample}-{num_pnts_per_traj_sample}-{num_trajs_eval}-{num_pnts_per_traj_eval}-{activation_threshold}-{random_seed}.txt"
            results_path = os.path.join(output_folder, output_name)
            with open(results_path, "w") as f:

                phi_name = "poly" if phi_type == 0 else "trig"
                logger.info(f"\n=== Running {phi_name} basis, degree={deg} ===")

                # -------------------------
                # Setup basis
                # -------------------------
                jacobian_basis.set_activation_threshold(activation_threshold)
                jacobian_basis.set_phi(phi_deg=deg, phi_type=phi_type)
                f.write(f"Robot: {robot}, Cameras: {camera_setup}\n")
                f.write(f"=== {phi_name} basis, degree={deg} ===\n")
                f.write(f"Initial basis elements per Jacobian entry: {jacobian_basis.jacobian_entry_basis_objects[0].symbolic_basis}\n")

                # -------------------------
                # Train
                # -------------------------
                f.write("TRAINING WITH FULL BASIS:")
                jacobian_basis.train(train_jacobians, train_joints)
                for entry in jacobian_basis.jacobian_entry_basis_objects:
                    f.write(
                        f"  J[{entry.name}]: "
                        f"number of basis elements={entry.number_of_basis_elements}, "
                        f"{entry.symbolic_basis} "
                        f"weights={entry.weights.tolist()}\n"

                    )

                # -------------------------
                # Evaluate (before reduction)
                # -------------------------
                _, rmse_before = jacobian_basis.evaluate(eval_jacobians, eval_joints)

                # -------------------------
                # Reduce + retrain
                # -------------------------
                f.write("TRAINING WITH REDUCED BASIS:")
                jacobian_basis.reduce_basis()
                jacobian_basis.train(train_jacobians, train_joints)

                _, rmse_after = jacobian_basis.evaluate(eval_jacobians, eval_joints)

                # -------------------------
                # Log symbolic structure
                # -------------------------

                f.write(
                    f"{phi_name}, deg={deg}, "
                    f"rmse_before={rmse_before:.4e}, "
                    f"rmse_after={rmse_after:.4e}\n"
                )

                for entry in jacobian_basis.jacobian_entry_basis_objects:
                    f.write(
                        f"  J[{entry.name}]: "
                        f"number of basis elements={entry.number_of_basis_elements}, "
                        f"{entry.symbolic_basis} "
                        f"weights={entry.weights.tolist()}\n"

                    )

                f.write("\n")

    logger.info("Done.")


if __name__ == "__main__":
    # verify_basis_object()
    # verify_jacobian_basis()
    main()




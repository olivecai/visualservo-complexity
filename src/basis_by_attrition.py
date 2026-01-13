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

def construct_trig_basis(params, degree):
    '''
    Construct a trigonometric basis up to the specified degree.
    For example, for degree=2 and params=[t0, t1], the basis will include:
    [sin(t0), cos(t0), sin(t1), cos(t1), sin(t0)*sin(t1), sin(t0)*cos(t1), cos(t0)*sin(t1), cos(t0)*cos(t1)]
    '''
    basis_elements = []

    num_params = len(params)

    # Single terms
    for d in range(1, degree + 1):
        for i in range(num_params):
            basis_elements.append(sp.sin(d * params[i]))
            basis_elements.append(sp.cos(d * params[i]))

    # Product terms
    for d in range(2, degree + 1):
        for indices in sp.utilities.iterables.multiset_partitions(range(num_params), d):
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
    '''
    def __init__(self, uvs: UVS):
    
        self.uvs = uvs
    
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
            entry.setup(params=self.params, activation_threshold=1e-2, symbolic_basis_expression=phi_func(self.params, degree=self.phi_degree))


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
                symbolic_basis_elements, weights = jac_entry_i_j.compute_basis(train_jacobian_matrix[:,i*self.m+j])


    def reduce_basis(self):
        pass


class Basis:
    '''
    This Basis object is used for storing, calculating, modifying, etc the regression basis and its weights.
    Any regression basis can be created by doing the following:
    1. Initialize: basis_obj = Basis("a")
    2. Set the parameters of the basis, ie `basis_obj.set_params([t0,t1,t2])`
    3. Set the phi expression to be the basis, ie `basis_obj.set_basis(expr)` where expr is a sympy expression
    4. Compute the weights given training data, ie `basis_obj.compute_basis(train_jacobian_entry, train_joints)`
    5. Optionally reduce the basis by removing trivial components, ie `basis_obj.reduce_basis()`
    6. Evaluate the basis on evaluation data, ie `basis_obj.evaluate(eval_jacobian_entry, eval_joints)`
    7. Get predictions on new data, ie `basis_obj.get_prediction(joints)`

    '''
    def __init__(self, name):
        self.name = name
        self.basis=None #function
        self.weights=None
        self.number_of_basis_elements = None
        self.activation_threshold = None
        self.discarded_basis_elements=[]
        self.symbolic_basis = None # vector shape 1,number_of_basis_elements
        self.params = None # the sympy parameters for the joints: t0, ..., t(dof-1)

    def setup(self, params, activation_threshold, symbolic_basis_expression):
        self.set_params(params)
        self.activation_threshold = activation_threshold
        self.set_basis(symbolic_basis_expression)

    def set_params(self, params):
        self.params=params

    def set_basis(self, symbolic_basis_expression):
        '''
        symbolic_basis_expression = scalar form expression like cos(t1)+sin(t0)+cos(t0)*sin(t1)
        self.symbolic_basis = vector form of basis
        self.basis = callable function (give joints get basis evaluation)
        '''

        self.symbolic_basis = list(symbolic_basis_expression.as_ordered_terms())
        self.number_of_basis_elements = len(self.symbolic_basis)

        self.basis = sp.lambdify(
            self.params,
            sp.Matrix(self.symbolic_basis),
            modules="numpy"
        )

    def eval_phi(self, joints:np.array): # return shape (number_of_basis_elements, )
        return np.asarray(self.basis(*joints), dtype=float).reshape(-1) #matrix of elements

    def reduce_basis(self):
        """
        Remove basis elements whose absolute weight is below activation_threshold.
        Mutates:
        - self.symbolic_basis
        - self.weights
        - self.discarded_basis_elements
        - self.number_of_basis_elements
        - self.basis (callable)
        """

        assert self.weights is not None, "Cannot reduce basis before training"
        assert self.symbolic_basis is not None, "Symbolic basis not set"

        kept_basis = []
        kept_weights = []

        for phi_k, w_k in zip(self.symbolic_basis, self.weights):
            if abs(w_k) >= self.activation_threshold:#if the weight is >= threshold, then keep it
                kept_basis.append(phi_k)
                kept_weights.append(w_k)
            else:
                self.discarded_basis_elements.append(phi_k)

        if len(kept_basis) == 0:
            raise RuntimeError(
                f"All basis elements discarded for Jacobian entry {self.name}"
            )

        # Update internal state
        self.symbolic_basis = kept_basis
        self.weights = np.array(kept_weights)
        self.number_of_basis_elements = len(kept_basis)

        # Rebuild callable basis function
        # basis(q) -> [phi_1(q), ..., phi_K(q)]
        self.basis = sp.lambdify(
            self.params,
            sp.Matrix(self.symbolic_basis),
            modules="numpy"
        )

        # at this point, the basis has been reduced and we can either. USE the current reduced basis OR recompute the weights given the new reduced basis. The nice thing is we can keep reusing the same data we collected.

    def train(self, train_jacobian_entry, train_joints):
        '''
        REGRESSION AND RET WEIGHTS

        return the symbolic basis elements and the corresponding weights.

        train_jacobian_entry is a N x 1 col vec,
        basis is a 1 x number_of_basis_elements vec,
        and the basis itself should be [number_of_basis_elements x 1]
        '''
        

        logger.info(f"Computing basis for Jacobian entry {self.name}:")
        # print(np.array(self.eval_phi(t) for t in train_joints))
        weights, residuals, rank, s = np.linalg.lstsq( np.vstack([self.eval_phi(q) for q in train_joints]), train_jacobian_entry, rcond=None)
        self.weights = weights.flatten()
        logger.info(f"Computed weights: {self.weights}")

    def get_prediction(self, joints):
        '''
        EVALUATE BASIS GIVEN JOINTS AND WEIGHTS
        '''
        return self.eval_phi(joints) @ self.weights
    
    def evaluate(self, eval_jacobian_entry, eval_joints):
        '''
        Evaluate the basis on the evaluation data and return the error list.

        eval_jacobian_entry   
        '''
        predictions = np.array([self.get_prediction(q) for q in eval_joints]).flatten()
        eval_jacobian_entry = np.array(eval_jacobian_entry).flatten()
        logging.info(f"Evaluating basis for {self.name}:")
        logging.info(f"Predictions: {predictions}")
        logging.info(f"Ground Truth: {eval_jacobian_entry}")
    
        errors = eval_jacobian_entry - predictions
        logging.info(f"Errors: {errors}")
        return errors


def verify_test():
    '''
    Just used for quick verification of Basis class functionality.
    '''

    q0, q1 = sp.symbols('q0 q1')

    expr = sp.sin(q0) + sp.cos(q1)
    logging.info(f"Expression: {expr}")
    b = Basis("0_0")
    b.setup(activation_threshold=1e-2, params=[q0, q1], symbolic_basis_expression=expr)

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
    b.train(train_jacobian_entry, train_joints)
    print(f"Trained weights: {b.weights}")
    b.reduce_basis()
    print(f"Reduced basis symbolic elements: {b.symbolic_basis}")
    print(f"Reduced basis weights: {b.weights}")
    b.train(train_jacobian_entry, train_joints)
    print(f"Retrained weights after reduction: {b.weights}")

if __name__ == "__main__":
    verify_test()




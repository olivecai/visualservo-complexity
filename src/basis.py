
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


# Configure basic logging to the console (default level is WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger for the current module
# logger. debug, info, warning, error, critical

sin=np.sin
cos=np.cos

P = DHSympyParams()


class Basis:
    '''
    This Basis object is used for storing, calculating, modifying, etc the regression basis and its weights.
    Any regression basis can be created by doing the following:
    1. Initialize: basis_obj = Basis("a")
    2. Set the parameters of the basis, ie `basis_obj.set_params([t0,t1,t2])`
    3. Set the phi expression to be the basis, ie `basis_obj.set_basis(expr)` where expr is a sympy expression
    4. Compute the weights given training data, ie `basis_obj.compute_basis(train_b, train_a)`
    5. Optionally reduce the basis by removing trivial components, ie `basis_obj.reduce_basis()`
    6. Evaluate the basis on evaluation data, ie `basis_obj.evaluate(eval_b, eval_a)`
    7. Get predictions on new data, ie `basis_obj.get_prediction(joints)`

    '''
    def __init__(self, name):
        self.name = name
        self.basis=None #callable function
        self.weights=None
        self.number_of_basis_elements = None
        self.activation_threshold = None
        self.discarded_basis_elements=[]
        self.symbolic_basis = None # the library of symbols. vector shape 1,number_of_basis_elements
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

    def lasso_regression(self, train_b, train_a, alpha=None):
        '''
        LASSO CV REGRESSION

        return the symbolic basis elements and the corresponding weights.

        train_b is a N x 1 col vec,
        basis is a 1 x number_of_basis_elements vec,
        and the basis itself should be [number_of_basis_elements x 1]
        '''
        if alpha is None:
            alpha = self.activation_threshold

        logger.info(f"Computing basis for Jacobian entry {self.name} using LASSO regression:")
        phi_matrix = np.vstack([self.eval_phi(q) for q in train_a])
        lasso = LassoCV(alphas=[alpha], cv=5).fit(phi_matrix, train_b)
    
        lasso.fit(phi_matrix, train_b)
        self.weights = lasso.coef_.flatten()
        logger.info(f"Computed weights: {self.weights}")
        return self.symbolic_basis, self.weights
        
    
    def train(self, train_b, train_a):
        '''
        REGRESSION AND RET WEIGHTS

        return the symbolic basis elements and the corresponding weights.

        train_b is a N x 1 col vec,
        basis is a 1 x number_of_basis_elements vec,
        and the basis itself should be [number_of_basis_elements x 1]
        '''
        

        logger.info(f"Computing basis for Jacobian entry {self.name}:")
        # print(np.array(self.eval_phi(t) for t in train_a))
        weights, residuals, rank, s = np.linalg.lstsq( np.vstack([self.eval_phi(q) for q in train_a]), train_b, rcond=None)
        self.weights = weights.flatten()
        logger.info(f"Computed weights: {self.weights}")
        return self.symbolic_basis, self.weights

    def get_prediction(self, joints):
        '''
        EVALUATE BASIS GIVEN JOINTS AND WEIGHTS
        '''
        return self.eval_phi(joints) @ self.weights
    
    def evaluate(self, eval_b, eval_a):
        '''
        RETURNS RSS

        Evaluate the basis on the evaluation data and return the error list.

        eval_b   
        '''
        predictions = np.array([self.get_prediction(q) for q in eval_a]).flatten()
        eval_b = np.array(eval_b).flatten()
        logging.info(f"Evaluating basis for {self.name}:")
        logging.info(f"Predictions: {predictions}")
        logging.info(f"Ground Truth: {eval_b}")
    
        errors = eval_b - predictions
        RSS = np.sum(errors**2)
        logging.info(f"Residual sum of squares: {RSS}")
        return RSS
    
    def sindy_stlsq(self, train_b, train_a, lambda_val=None, max_iter=10):
        '''
        SINDy Sequential Thresholded Least Squares Regression

        return the symbolic basis elements and the corresponding weights.
        MUTATES weights, does NOT MUTATE SYMBOL LIBRARY BASIS

        train_b is a N x 1 col vec,
        basis is a 1 x number_of_basis_elements vec,
        and the basis itself should be [number_of_basis_elements x 1]
        '''
        logger.info(f"Computing basis for {self.name} using SINDy STLSQ regression:")
        phi_matrix = np.vstack([self.eval_phi(q) for q in train_a])
        
        # Initial least squares fit
        weights, residuals, rank, s = np.linalg.lstsq(phi_matrix, train_b, rcond=None)
        weights = weights.flatten()

        for iteration in range(max_iter):
            '''In one iteration, threshold small weights and refit big ones.
            If all weights below threshold, break.'''
            small_indices = np.abs(weights) < lambda_val
            weights[small_indices] = 0
            
            if np.all(small_indices):
                break  # all weights below threshold, do not refit. DO NOTHING!
            
            # Refit with remaining terms
            large_indices = ~small_indices
            logger.info(f"small_indices: {small_indices}")
            logger.info(f"large_indices: {large_indices}")

            if np.sum(large_indices) == 0:
                break  # No terms left to fit
            
            weights[large_indices], residuals, rank, s = np.linalg.lstsq(
                phi_matrix[:, large_indices], 
                train_b, 
                rcond=None
            )
            weights = weights.flatten()

        self.weights = weights
        logger.info(f"Computed weights: {self.weights}")
        logger.info(f"Symbolic basis: {self.symbolic_basis}")
        return self.symbolic_basis, self.weights
    
    def compute_aic(self, RSS, N, k):
        '''N: number of data points
        k: number of parameters (non-zero weights)'''
        return N * np.log(RSS / N) + 2 * k

    def compute_aicc(self, RSS, N, k):
        aic = self.compute_aic(RSS, N, k)
        if N > k + 1:
            return aic + (2 * k * (k + 1)) / (N - k - 1)
        else:
            return np.inf
        
    def pareto_frontier(self, train_b, train_a, eval_b, eval_a, lambda_values):
        weights_list = []
        RSS_list=[]
        num_basis_elements_list=[]
        aicc_list = []

        for lambda_val in lambda_values:
            _, weights = self.sindy_stlsq(train_b=train_b, train_a=train_a,  lambda_val=lambda_val)
            RSS= self.evaluate(eval_b=eval_b, eval_a=eval_a)
            num_basis_elements = sum(1 for x in weights if x) #count the number of activated components
            if num_basis_elements == 0:
                aicc_list.append(np.inf)
            else:
                aicc_list.append(self.compute_aicc(RSS, N=len(eval_b), k=num_basis_elements))
            weights_list.append(weights)
            RSS_list.append(RSS)
            num_basis_elements_list.append(num_basis_elements)

        
        plt.scatter(num_basis_elements_list,RSS_list)
        plt.xlabel("Number of Basis Elements")
        plt.ylabel("Residual Sum of Squares (RSS)")
        plt.title("Pareto Frontier: Number of Basis Elements vs RSS")
        for i in range(len(lambda_values)):
            logger.info(f"lambda value: {lambda_values[i]:.2f}, num_basis_elements: {num_basis_elements_list[i]}, RSS: {RSS_list[i]}, AICC: {aicc_list[i]:.2f}")
        
        plt.show()

        #return the lambda val associated with the minimum aicc:
        # lambda, weights, RSS, num_basis_elements, min_aicc
        return lambda_values[np.argmin(aicc_list)], weights_list[np.argmin(aicc_list)], RSS_list[np.argmin(aicc_list)], num_basis_elements_list[np.argmin(aicc_list)], min(aicc_list)   
        
if __name__ == "__main__":
    # Example usage
    t0, t1 = sp.symbols('t0 t1')
    basis_expr = sp.cos(t0) + sp.sin(t1) + sp.cos(t0)*sp.sin(t1)
    
    basis_obj = Basis("example_basis")
    basis_obj.setup(params=[t0, t1], activation_threshold=0.1, symbolic_basis_expression=basis_expr)
    
    # # Dummy training data
    # train_a = [np.array([0.0, 0.0]), np.array([np.pi/2, np.pi/2]), np.array([np.pi, np.pi])]
    # train_b = np.array([1.0, 0.5, -1.0])

    # call .basis(t0,t1) to generate training data...
    train_a = np.random.uniform(0, 2*np.pi, (30, 2))
    train_b = np.array([basis_obj.basis(*q).sum() for q in train_a])
    
    # Dummy evaluation data
    eval_a = np.random.uniform(0, 2*np.pi, (30, 2))
    eval_b = np.array([basis_obj.basis(*q).sum() for q in eval_a])
    # eval_a = [np.array([np.pi/4, np.pi/4]), np.array([3*np.pi/4, 3*np.pi/4])]
    # eval_b = np.array([0.7, -0.7])
    
    basis_expr = sp.cos(t0) + sp.sin(t1) + sp.cos(t0)*sp.sin(t1) + sp.cos(t0)**2 + sp.sin(t1)**2
    basis_obj.set_basis(basis_expr)
    
    basis_obj.train(train_b, train_a)
    print(f"Symbolic Basis before reduction: {basis_obj.symbolic_basis}")
    basis_obj.reduce_basis()
    print(f"Symbolic Basis after reduction: {basis_obj.symbolic_basis}")

    errors = basis_obj.evaluate(eval_b, eval_a)
    print("Evaluation Errors:", errors)

    print("Now check SINDy")

    t0, t1 = sp.symbols('t0 t1')
    
    basis_obj = Basis("sindy")
    basis_obj.setup(params=[t0, t1], activation_threshold=0.1, symbolic_basis_expression=basis_expr)
    
    basis_obj.sindy_stlsq(train_b, train_a, lambda_val=0.1, max_iter=10)
    print(f"Symbolic Basis: {basis_obj.symbolic_basis}")
    print(f"Weights: {basis_obj.weights}")

    errors = basis_obj.evaluate(eval_b, eval_a)
    print("Evaluation Errors:", errors)

    print("Now let's see that PARETO CURVE")
    lambda_val, weights, RSS, num_basis_elements, min_aicc = basis_obj.pareto_frontier(train_b=train_b, train_a=train_a, eval_b=eval_b, eval_a=eval_a, lambda_values=np.linspace(0,1,11).tolist())
    
    print(f"Best lambda: {lambda_val}, Weights: {weights}, RSS: {RSS}, Num Basis Elements: {num_basis_elements}, Min AICC: {min_aicc}")
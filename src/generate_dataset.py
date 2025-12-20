#!/usr/bin/env python
'''
Dec 19 2205

Initialize the JacobianBasis object.

get dataset of sample joint positions and Jacobians and return it in the function "collect_data"


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
        logger.info(f"m: {self.m}, n: {self.n}")

    def set_phi_degree(self, phi_deg):
        self.phi_degree = phi_deg

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


def main():
    '''
    Usage example: `python3 ./src/potential_basis.py dof2 0,1 2 5 10 500 10 888 testing`
    '''

    _, robot, camera_setup, num_trajs_sample, num_pnts_per_traj_sample, random_seed = sys.argv
    random_seed = int(random_seed)
    num_pnts_per_traj_sample=int(num_pnts_per_traj_sample)
    num_trajs_sample=int(num_trajs_sample)

    camera_setup = [int(i) for i in camera_setup.split(',')]

    logger.info(f"robot: {robot}\ncamera_setup: {camera_setup}\nnum_trajs_sample: {num_trajs_sample}\nnum_pnts_per_traj_sample: {num_pnts_per_traj_sample}\nrandom_seed: {random_seed}")

    uvs = UVS(robot, [0]) #2 dof arm with a direct projection onto the scene from above
    jacobian_basis = JacobianBasis(uvs)
    rng = np.random.default_rng(seed=random_seed)
    sample_joints, sample_jacobians = jacobian_basis.collect_data(num_trajectories=num_trajs_sample,num_pnts_per_traj=num_pnts_per_traj_sample,rng=rng)
    return sample_joints, sample_jacobians


if __name__ == '__main__':
    main()
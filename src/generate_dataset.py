"""FEB 2 2026"""

from robot_toolbox.create_uvs import UVS, analytic_cameras
from robot_toolbox.dh_robot import DHSympyParams, DenavitHartenbergAnalytic
from robot_toolbox.kinematic_structure import get_kinematic_structure
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

from fkin_all_frames import ForwardKinematic_AllFrames

''' Generates a dataset for N joint trajectories sampled with K points along each trajectory
'''

P = DHSympyParams()

def main():
    params = sys.argv
    for i in range(len(params)):
        try:
            params[i] = int(params[i])
        except:
            pass

    (
        _,
        robot_name,
        num_trajectories,
        num_points_per_trajectory,
        random_seed,
        noise_input,
        noise_output,
        output_folder,
        get_all_frames,
    ) = params

    logger.info(
        f"\nrobot_name={robot_name}\nnum_trajectories={num_trajectories}\nnum_points_per_trajector{num_points_per_trajectory}\nrandom_seed={random_seed}\nnoise_input={noise_input}\nnoise_output={noise_output}\noutput_folder={output_folder}"
    )

    random_seed = int(random_seed)
    get_all_frames = int(get_all_frames) # 0 or 1
    num_trajectories = int(num_trajectories)
    num_points_per_trajectory = int(num_points_per_trajectory)

    noise_input=float(noise_input)
    noise_output=float(noise_output)

    rng = np.random.default_rng(random_seed)

    os.makedirs(output_folder, exist_ok=True)

    robot = UVS(robot_name, cam_idx=[0]).dh_robot # ignore camera for true cartesian real world fkin
        
    fkin_object = ForwardKinematic_AllFrames(robot)
    if get_all_frames:
        joints, fkin_all = fkin_object.get_data(num_trajectories=num_trajectories, num_pnts_per_traj=num_points_per_trajectory, rng=rng, add_input_noise=noise_input, add_output_noise=noise_output)
    else:
        joints, fkins = fkin_object.collect_data(num_trajectories=num_trajectories, num_pnts_per_traj=num_points_per_trajectory, rng=rng, add_input_noise=noise_input, add_output_noise=noise_output)

    if get_all_frames:
        output_data = fkin_all
    else:
        output_data = fkins
    np.save(f"{output_folder}/joints.npy", joints, allow_pickle=True)
    np.save(f"{output_folder}/fkin_data.npy", output_data, allow_pickle=True)
    logger.info("DONE GENERATING DATASET")

if __name__ == "__main__":
    main()
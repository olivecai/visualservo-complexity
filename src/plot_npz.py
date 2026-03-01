''' plot any npz file'''

def plot_npz_overlapping(filenames):
    import numpy as np
    import matplotlib.pyplot as plt
    
    names=['2 DOF Nonplanar','3 DOF Nonplanar', '2 DOF Planar', '3 DOF Planar']
    for i in range(len(filenames)):
        
        data = np.load(filenames[i])
        plt.plot(data['eps_lim_values'], data['convergence_errors'], label=names[i])
    
    plt.xlabel('Entry-Wise Perturbation Scalar')
    plt.ylabel('Average Convergence Error')
    plt.title('Visual Servoing Jacobian Perturbation VS Average Convergence Error over 50 Trajectories in Workspace')
    plt.legend()
    plt.show()

# plot_npz_overlapping(['jacobian_perturbation_results_dof2_alt.npz','jacobian_perturbation_results_dof3_alt.npz','jacobian_perturbation_results_dof2_planar.npz','jacobian_perturbation_results_dof3_planar.npz'])
plot_npz_overlapping(['jacobian_perturbation_results_dof2_alt_vs.npz','jacobian_perturbation_results_dof3_alt_vs.npz','jacobian_perturbation_results_dof2_planar_vs.npz','jacobian_perturbation_results_dof3_planar_vs.npz'])
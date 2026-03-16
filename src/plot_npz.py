''' plot any npz file'''
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pylab as plt
import argparse


pi=np.pi
sin = sp.sin
cos = sp.cos
sqrt = sp.sqrt
JOINT_RANGE=(pi/6, pi/2)
small_extra_range=0*pi/12
# lower=(pi/6)-2*(pi/2)-(pi/12), upper=3*(pi/2)+(pi/12)
# (-5pi/6, 3pi/2)
LOWER_SINUSOID = JOINT_RANGE[0] - 2* JOINT_RANGE[1] - small_extra_range
UPPER_SINUSOID = JOINT_RANGE[1] * 3 + small_extra_range

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

import numpy as np
import matplotlib.pyplot as plt

def compare_suite_npz(npz_paths, labels=None):
    """
    Compare multiple .npz result files from one suite.

    Parameters
    ----------
    npz_paths : list of str
        Paths to .npz files.
    labels : list of str or None
        Labels for legend. If None, filenames are used.
    """
    if labels is None:
        labels = [f"run_{i}" for i in range(len(npz_paths))]

    results = []
    for path in npz_paths:
        data = np.load(path)
        results.append({
            "path": path,
            "max_entrywise_jacobian_residual": data["max_entrywise_jacobian_residual"],
            "x_hist": data["x_hist"],
            "q_hist": data["q_hist"],
            "x_star": data["x_star"],
            "error_per_step": data["error_per_step"],
            "iters": int(data["iters"]),
        })

    # --------------------------------------------------
    # 1. max_entrywise_jacobian_residual
    # --------------------------------------------------
    plt.figure()
    for r, label in zip(results, labels):
        y = np.atleast_1d(r["max_entrywise_jacobian_residual"])
        if y.size == 1:
            plt.scatter([0], y, label=label)
        else:
            plt.plot(y, label=label)
    plt.title("Comparison: max_entrywise_jacobian_residual")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.legend()
    plt.grid(True)

    # --------------------------------------------------
    # 2. x_hist
    # one figure, one subplot per x dimension
    # --------------------------------------------------
    x_dim = results[0]["x_hist"].shape[1]
    fig, axes = plt.subplots(x_dim, 1, figsize=(8, 3 * x_dim), squeeze=False)
    axes = axes.flatten()
    for dim in range(x_dim):
        for r, label in zip(results, labels):
            axes[dim].plot(r["x_hist"][:, dim], label=label)
        axes[dim].set_title(f"Comparison: x_hist[:, {dim}]")
        axes[dim].set_xlabel("Iteration")
        axes[dim].set_ylabel(f"x_{dim}")
        axes[dim].grid(True)
        axes[dim].legend()
    plt.tight_layout()

    # --------------------------------------------------
    # 3. q_hist
    # one figure, one subplot per q dimension
    # --------------------------------------------------
    q_dim = results[0]["q_hist"].shape[1]
    fig, axes = plt.subplots(q_dim, 1, figsize=(8, 3 * q_dim), squeeze=False)
    axes = axes.flatten()
    for dim in range(q_dim):
        for r, label in zip(results, labels):
            axes[dim].plot(r["q_hist"][:, dim], label=label)
        axes[dim].set_title(f"Comparison: q_hist[:, {dim}]")
        axes[dim].set_xlabel("Iteration")
        axes[dim].set_ylabel(f"q_{dim}")
        axes[dim].grid(True)
        axes[dim].legend()
    plt.tight_layout()

    # --------------------------------------------------
    # 4. x_star
    # compare target vector values across runs
    # --------------------------------------------------
    x_star_dim = results[0]["x_star"].shape[0] if np.ndim(results[0]["x_star"]) > 0 else 1
    fig, axes = plt.subplots(x_star_dim, 1, figsize=(8, 3 * x_star_dim), squeeze=False)
    axes = axes.flatten()
    x_positions = np.arange(len(labels))

    for dim in range(x_star_dim):
        vals = []
        for r in results:
            xs = np.atleast_1d(r["x_star"])
            vals.append(xs[dim])
        axes[dim].bar(x_positions, vals)
        axes[dim].set_title(f"Comparison: x_star[{dim}]")
        axes[dim].set_xticks(x_positions)
        axes[dim].set_xticklabels(labels, rotation=20)
        axes[dim].set_ylabel(f"x_star[{dim}]")
        axes[dim].grid(True)
    plt.tight_layout()

    # --------------------------------------------------
    # 5. error_per_step
    # --------------------------------------------------
    plt.figure()
    for r, label in zip(results, labels):
        plt.plot(r["error_per_step"], label=label)
    plt.title("Comparison: error_per_step")
    plt.xlabel("Iteration")
    plt.ylabel("||x - x*||")
    plt.legend()
    plt.grid(True)

    # --------------------------------------------------
    # 6. iters
    # --------------------------------------------------
    plt.figure()
    iter_vals = [r["iters"] for r in results]
    plt.bar(labels, iter_vals)
    plt.title("Comparison: iters")
    plt.ylabel("Iterations")
    plt.xticks(rotation=20)
    plt.grid(True)

    plt.show()

def plot_pareto():
    name = "dof2_alt"
    data = np.load(f"pareto_curve_model_{name}.npz", allow_pickle=True)
    print(data.files)
    model_complexity = data["parameter_count"]
    err = data["error"]
    color_code = ['r']*len(err)
    color_code[:4] = ['b']*4
    print(model_complexity)
    print(err)
    print(color_code)

    new_data = dict(data)
    print(model_complexity)

    model_complexity[:4] = [108,200,288,380]

    new_data["parameter_count"] = model_complexity

    np.savez(f"pareto_curve_model_{name}.npz", **new_data)

    # plot parameter count VS average jacobian error
    plt.figure()
    plt.scatter(model_complexity, err, c=color_code)   
    plt.xlabel("Parameter Count of Model")
    plt.ylabel("Average Jacobian Perturbation Error")
    plt.title(f"Model Fitting Pareto Curve for {name}")
    plt.grid(True)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

LOWER = LOWER_SINUSOID
UPPER = UPPER_SINUSOID

def plot_piecewise(ax, func, knots, label):
    knots = np.array(knots, dtype=float)

    # dense curve for reference
    xs = np.linspace(LOWER, UPPER, 2000)
    ax.plot(xs, func(xs), color="0.8", linewidth=1)

    # piecewise linear segments
    ys = func(knots)
    ax.plot(knots, ys, color="blue", linewidth=2)
    ax.plot(knots, ys, "o", color="blue", markersize=5)

    ax.set_title(label)
    ax.set_xlim(LOWER, UPPER)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)


def plot_trig_approximations():

    sin_knots_A = [LOWER, -1.57079633, 1.57079633, UPPER]
    sin_knots_B = [LOWER, -2.0943951, -1.04719755, 1.04719755, 2.0943951, 4.1887902, UPPER]
    sin_knots_C = [LOWER, -2.35619449, -1.57079633, -0.78539816, 0.78539816, 1.57079633, 2.35619449, 3.92699082, UPPER]
    sin_knots_D = [LOWER, -2.51327412, -1.88495559, -1.25663706, -0.62831853, 0.62831853, 1.25663706, 1.88495559, 2.51327412, 3.76991118, 4.39822972, UPPER]

    cos_knots_A = [LOWER, 0, np.pi, UPPER]
    cos_knots_B = [LOWER, -0.52359878, 0.52359878, 2.61799388, 3.66519143, UPPER]
    cos_knots_C = [LOWER, -2.35619449, -0.78539816, 0.0, 0.78539816, 2.35619449, 3.14159265, 3.92699082, UPPER]
    cos_knots_D = [LOWER, -2.31485774, -0.99208189, -0.33069396, 0.33069396, 0.99208189, 2.31485774, np.pi-0.33, np.pi+0.33, 3.9683275671795863, UPPER]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    plot_piecewise(axes[0,0], np.sin, sin_knots_A, "sin A")
    plot_piecewise(axes[0,1], np.sin, sin_knots_B, "sin B")
    plot_piecewise(axes[0,2], np.sin, sin_knots_C, "sin C")
    plot_piecewise(axes[0,3], np.sin, sin_knots_D, "sin D")

    plot_piecewise(axes[1,0], np.cos, cos_knots_A, "cos A")
    plot_piecewise(axes[1,1], np.cos, cos_knots_B, "cos B")
    plot_piecewise(axes[1,2], np.cos, cos_knots_C, "cos C")
    plot_piecewise(axes[1,3], np.cos, cos_knots_D, "cos D")

    fig.suptitle("Piecewise Linear Sin/Cos Approximations (Increasing Fidelity)", fontsize=14)
    plt.tight_layout()
    plt.show()


# plot_trig_approximations()

# plot_pareto()
# plot_npz_overlapping(['jacobian_perturbation_results_dof2_alt.npz','jacobian_perturbation_results_dof3_alt.npz','jacobian_perturbation_results_dof2_planar.npz','jacobian_perturbation_results_dof3_planar.npz'])
# plot_npz_overlapping(['jacobian_perturbation_results_dof2_alt_vs.npz','jacobian_perturbation_results_dof3_alt_vs.npz','jacobian_perturbation_results_dof2_planar_vs.npz','jacobian_perturbation_results_dof3_planar_vs.npz'])
# plot_npz_overlapping(['jacobian_perturbation_results_dof2_alt_vs_additive_eps_signed.npz','jacobian_perturbation_results_dof3_alt_vs_additive_eps_signed.npz','jacobian_perturbation_results_dof2_planar_vs_additive_eps_signed.npz','jacobian_perturbation_results_dof3_planar_vs_additive_eps_signed.npz'])

# compare_suite_npz(["main_script_suite[False, True, False]_model_idx3_robotdof2_planar.npz", "main_script_suite[False, True, True]_model_idx3_robotdof2_planar.npz"])
# compare_suite_npz(["main_script_suite[True, False, False]_model_idx3_robotdof2_planar.npz","main_script_suite[True, False, True]_model_idx3_robotdof2_planar.npz","main_script_suite[True, True, False]_model_idx3_robotdof2_planar.npz","main_script_suite[True, True, True]_model_idx3_robotdof2_planar.npz" ])
# compare_suite_npz(["main_script_suite[True, True, True, False]_model_idx3_robotdof2_planar.npz",  "main_script_suite[True, False, True, False]_model_idx3_robotdof2_planar.npz","main_script_suite[True, False, True, True]_model_idx3_robotdof2_planar.npz"])


compare_suite_npz(["main_script_suite[True, True, True, False]_model_idx3_robotdof3_alt.npz",  "main_script_suite[True, False, True, False]_model_idx3_robotdof3_alt.npz","main_script_suite[True, False, True, True]_model_idx3_robotdof3_alt.npz"])
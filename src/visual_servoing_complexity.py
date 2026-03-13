'''
We have a few questions to explore; 
- Q1: How robust is the Jacobian to magnitude, and to directional, changes?
- Q2: What is the effect of using a constant Jacobian over the space? 
- Q3: How many linear hyperplanes can be used to coarsely approximate the visual servoing function?
- Q4: What is the effect of more finely/coarsely modelling different joint contributions to the resulting kinematic structure? 
Q1 and Q2 ask how much error can be permitted.
Q3 and Q4 are structual modelling questions and assume we know how much error is permitted.

There are a few different interesting experiments we can run.
1. visual servoing error per jacobian entry and resulting convergence error (add perturbation to each entry where perturbation ~ uniform(1,eps_limit) where eps limit is a value from -1.5 to 3) and see the upper and lower perturbations that still warrant convergence: perturb_lower, perturb_upper
2. combinatorially find the best model given different sin and cos structures: we want to explore whether the 
'''

'''
Feb 5 2025

Part A:

Approximate sin and cos by some piecewise function

Then use sin^ and cos^ instead of sin and cos in the fkin matrix chain (the rotation and translation matrices) 

extract the translation vector (which will be the end effector position) and let's call this result fkin^

compare fkin^ with fkin: how many linear segments do we need to approximate the fkin within error bound e?

not only compare fkin^ accuracy, but compute the spectral norm of the jacobian that we get from fkin^ when we evaluate the visual servoing task:
QUESTION: for each joint position, it has one unique fkin and one unique jacobian. For the spectral norm analysis:
1. choose a joint configuration q_0 to plug into model fkin^. 
2. Then, we choose some joint position q* and calculate true forward kinematics of q* and call this fkin point the desired solution x*. (q is joint configuration, x is workspace point.)
3. Then, we compute the spectral 


create piecewise functions

create a rotation matrix for the robot

create a function to plot the robot in 3d space

create a newton method function to solve visual servoing for some target in some joint workspace

'''

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
small_extra_range=pi/12
# lower=(pi/6)-2*(pi/2)-(pi/12), upper=3*(pi/2)+(pi/12)
LOWER_SINUSOID = JOINT_RANGE[0] - 2* JOINT_RANGE[1] - small_extra_range
UPPER_SINUSOID = JOINT_RANGE[1] * 3 + small_extra_range


from robot_toolbox.camera import Camera

# cam1 = Camera(sp.pi,sp.pi/16,-sp.pi/2,[-1,0,5], 5,5, 0, 0) 
# cam2 = Camera(sp.pi+sp.pi/16, sp.pi/16, -sp.pi/2, [-1,-1,5], 5,5,0,0) 
cam1 = Camera(sp.pi, sp.pi/16,-sp.pi/2,[-1,0,5], 5,5, 0, 0) 
cam2 = Camera(sp.pi, (sp.pi/4), -sp.pi/2, [-5,0,5], 5,5,0,0) 
cams=[cam1,cam2]


def create_piecewise_sinusoid(sympy_function, knots):
    """
    Returns a callable pw(arg) that produces a SymPy Piecewise linear interpolation
    of sympy_function(arg) over the knot interval [knots[0], knots[-1]].

    note: the caller is responsible for passing argmuents in acceptable domain
    """
    knots = [float(k) for k in list(knots)] 
    knots = list(knots)
    if len(knots) < 2:
        raise ValueError("Need at least 2 knots")

    def pw(arg):
        ''' Creates a line y=mx+b'''
        
        pieces = []
        for i in range(len(knots) - 1):
            x0 = sp.nsimplify(knots[i])
            x1 = sp.nsimplify(knots[i + 1])
            y0 = sympy_function(x0)
            y1 = sympy_function(x1)

            m = (y1 - y0) / (x1 - x0)
            m*=1.0
            c = y0 - m * x0

            expr = sp.simplify(m * arg + c)

            if i < len(knots) - 2:
                cond = sp.And(arg >= x0, arg < x1)
            else:
                cond = sp.And(arg >= x0, arg <= x1)

            pieces.append((expr, cond))

        pieces.append((pieces[-1][0], True))  # fallback
        return sp.Piecewise(*pieces)

    return pw


# pw = create_piecewise_sinusoid(sp.sin, np.linspace(-2*pi,2*pi,9))

# x = sp.symbols('x')
# expr = pw(x)
# f = sp.lambdify(x, expr, "numpy")
# xs = np.linspace(-2*np.pi, 2*np.pi, 1000)
# ys = f(xs)

# # plt.plot(xs, ys)
# # plt.ylim(-2, 2)
# # plt.show()
    
def newton_raphson(f, J_inv, x0, tol=1e-1, max_iter=100, bounds=None, chord_newton=False, camera_projection=0):
    """
    Finds the root of a function f(x) using the Newton-Raphson method.

    Parameters:
    f (function): The function for which to find the root.
    df (function): The derivative of the function f(x).
    x0 (float): The initial guess for the root.
    tol (float): The tolerance (stopping criterion) for convergence.
    max_iter (int): The maximum number of iterations.

    Returns:
    float: The estimated root.
    int: The number of iterations taken.
    """
    tol=1e-1
    q = np.array(x0, dtype=float).reshape(-1)

    q_hist = [q.copy()]
    e_hist = []

    if chord_newton:
        Jpinv = np.array(J_inv(q), dtype=float)
    for n in range(max_iter):
        if chord_newton is False:
            Jpinv = np.array(J_inv(q), dtype=float)
        e = np.array(f(q), dtype=float).reshape(-1)
        e_norm = float(np.linalg.norm(e))
        e_hist.append(e_norm)

        if e_norm < tol:
            return q, n, np.array(q_hist), np.array(e_hist)

        dq = Jpinv @ e
        q = q + dq

        if bounds is not None:
            for i in range(len(q)):
                if q[i] < bounds[i][0] or q[i] > bounds[i][1]:
                    return q, n, np.array(q_hist), np.array(e_hist)

        q_hist.append(q.copy())

    # print(f"Warning: Maximum iterations ({max_iter}) exceeded. Not converged to tol={tol}.")
    return q, max_iter, np.array(q_hist), np.array(e_hist)


# --------------------------
# plotting (updated: 4 curves each)
# --------------------------
def plot_cartesian_trajectory(xs_ct, xs_cm, xs_nt, xs_nm, x_star):
    """
    xs_* are either:
      - (T,2) for one camera: [u,v]
      - (T,4) for two cameras: [u1,v1,u2,v2]
    x_star is either (2,) or (4,)
    """
    xs_ct_list = split_uv(xs_ct)
    xs_cm_list = split_uv(xs_cm)
    xs_nt_list = split_uv(xs_nt)
    xs_nm_list = split_uv(xs_nm)

    x_star = np.asarray(x_star).reshape(-1)
    if x_star.size == 2:
        stars = [x_star]
    elif x_star.size == 4:
        stars = [x_star[0:2], x_star[2:4]]
    else:
        raise ValueError(f"x_star must be size 2 or 4, got {x_star.size}")

    ncam = len(xs_ct_list)

    if ncam == 1:
        plt.figure()
        plt.plot(xs_ct_list[0][:, 0], xs_ct_list[0][:, 1], marker="o", label="Chord: True")
        plt.plot(xs_cm_list[0][:, 0], xs_cm_list[0][:, 1], marker="o", label="Chord: Model")
        plt.plot(xs_nt_list[0][:, 0], xs_nt_list[0][:, 1], marker="o", label="Newton: True")
        plt.plot(xs_nm_list[0][:, 0], xs_nm_list[0][:, 1], marker="o", label="Newton: Model")
        plt.scatter([stars[0][0]], [stars[0][1]], marker="x", s=90, label="Target")
        plt.xlabel("u (pixels)")
        plt.ylabel("v (pixels)")
        plt.title("Image-plane trajectory (u,v)")
        plt.grid(True)
        plt.legend()
        plt.show()

    elif ncam == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, ax in enumerate(axes):
            ax.plot(xs_ct_list[i][:, 0], xs_ct_list[i][:, 1], marker="o", label="Chord: True")
            ax.plot(xs_cm_list[i][:, 0], xs_cm_list[i][:, 1], marker="o", label="Chord: Model")
            ax.plot(xs_nt_list[i][:, 0], xs_nt_list[i][:, 1], marker="o", label="Newton: True")
            ax.plot(xs_nm_list[i][:, 0], xs_nm_list[i][:, 1], marker="o", label="Newton: Model")
            ax.scatter([stars[i][0]], [stars[i][1]], marker="x", s=90, label="Target")
            ax.set_xlabel("u (pixels)")
            ax.set_ylabel("v (pixels)")
            ax.set_title(f"Camera {i+1}")
            ax.grid(True)
            ax.legend()
        fig.suptitle("Image-plane trajectory (u,v)")
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError(f"Only 1 or 2 cameras supported, got {ncam}")



def plot_joint_trajectory(q_ct, q_cm, q_nt, q_nm):
    plt.figure()
    dof = q_ct.shape[1]
    it_ct = np.arange(q_ct.shape[0])
    it_cm = np.arange(q_cm.shape[0])
    it_nt = np.arange(q_nt.shape[0])
    it_nm = np.arange(q_nm.shape[0])

    for j in range(dof):
        plt.plot(it_ct, q_ct[:, j], label=f"q{j} chord true")
        plt.plot(it_cm, q_cm[:, j], linestyle="--", label=f"q{j} chord model")
        plt.plot(it_nt, q_nt[:, j], label=f"q{j} newton true")
        plt.plot(it_nm, q_nm[:, j], linestyle="--", label=f"q{j} newton model")

    plt.xlabel("iteration")
    plt.ylabel("joint angle (rad)")
    plt.title("Joint-space trajectory")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_error_history(e_ct, e_cm, e_nt, e_nm):
    plt.figure()
    plt.plot(e_ct, label="Chord: ||e|| true")
    plt.plot(e_cm, label="Chord: ||e|| model")
    plt.plot(e_nt, label="Newton: ||e|| true")
    plt.plot(e_nm, label="Newton: ||e|| model")
    plt.xlabel("iteration")
    plt.ylabel("||x* - fkin(q)||")
    plt.title("Task error vs iteration")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_spectral_norm_history(sig_ct, sig_cm, sig_nt, sig_nm):
    plt.figure()
    plt.plot(sig_ct, label="Chord: ||J||2 true")
    plt.plot(sig_cm, label="Chord: ||J||2 model")
    plt.plot(sig_nt, label="Newton: ||J||2 true")
    plt.plot(sig_nm, label="Newton: ||J||2 model")
    plt.xlabel("iteration")
    plt.ylabel("spectral norm")
    plt.title("Jacobian spectral norm along trajectory")
    plt.grid(True)
    plt.legend()
    plt.show()

def get_p_and_p_true(sin_list, cos_list, vars, robot_name,cameras=[cam1,cam2]):
    # --- build symbolic FK (4x4) using your rotation-matrix chain
    T_true = get_robot_fkin_expr(name=robot_name, vars=vars, cameras=cameras)
    T = get_vs_fkin_expr(name=robot_name, vars=vars, sin_hat_list=sin_list, cos_hat_list=cos_list, cameras=cameras)
    print(f"T:{T}, T_true:{T_true}")
    #T_true is in symbolic form, so directly substitute in the cos=cos_list[0] and sin=sin_list[0] to get the piecewise approximation of the true FK, which we will use for the Jacobian and visual servoing, but we will still use the true FK for the position and target definition
    # T = get_robot_rotation_matrix(name=robot_name, sin_hat=sin_list, cos_hat=cos_list, vars=vars)
    # --- end-effector position = translation column
    # p = sp.Matrix([T[0, 3], T[1, 3], T[2, 3]])         # 3x1
    # p_true = sp.Matrix([T_true[0, 3], T_true[1, 3], T_true[2, 3]])   
    p=T;p_true=T_true
    return p, p_true


def script(sin_list, cos_list, vars, robot_name, q0, q_star, damping=1e-3, chord_newton=False, jacobian_eps_max=1.0,p=None,p_true=None, tol=1e-1):
    # --- build symbolic FK (4x4) using your rotation-matrix chain
    if p is None or p_true is None:
        p,p_true = get_p_and_p_true(sin_list, cos_list, vars, robot_name, cameras=cams)

    J = p.jacobian(vars)                               # 3 x dof

    # --- numeric callables

    p_fun = sp.lambdify(vars, p_true, "numpy")
    J_fun = sp.lambdify(vars, J, "numpy")

    
        # knot range for numeric wrapping
    k0 = -2*np.pi
    k1 = 2*np.pi
    period = k1 - k0


    def wrap_q(q):
        q = np.asarray(q, dtype=float)
        return k0 + np.mod(q - k0, period)

    # wrap inputs before calling lambdified sympy
    def p_fun_wrapped(*q):
        qw = wrap_q(q)
        return p_fun(*qw)

    def J_fun_wrapped(*q):
        qw = wrap_q(q)
        return J_fun(*qw)

    # --- define the target in Cartesian space from TRUE FK at q_star (passed in)
    # x_star = np.array(p_fun_wrapped(*q_star), dtype=float).reshape(-1)
    x_star = np.array(p_fun(*q_star), dtype=float).reshape(-1)

    # --- visual servoing / IK error function: e(q) = x* - fkin(q)
    def f(q):
        x = np.array(p_fun(*q), dtype=float).reshape(-1)
        return x_star - x

    # --- damped pseudo-inverse Jacobian (stable)
    def J_inv(q):
        
        # Jn = np.array(J_fun_wrapped(*q), dtype=float)
        Jn = np.array(J_fun(*q), dtype=float)
        J_pinv = np.linalg.pinv(Jn)
        return get_perturbed_jacobian(J_pinv, eps_max=jacobian_eps_max)
    

    # --- run Newton iterations in joint space
    q_sol, iters, q_hist, e_hist = newton_raphson(f, J_inv, q0, tol=tol, max_iter=60, chord_newton=chord_newton)

    # --- collect Cartesian trajectory and spectral norms
    x_hist = np.array([np.array(p_fun(*q), dtype=float).reshape(-1) for q in q_hist])

    sig_hist = []
    for q in q_hist:
        s = np.linalg.svd(np.array(J_fun(*q), dtype=float), compute_uv=False)
        sig_hist.append(float(s[0]))
    sig_hist = np.array(sig_hist)

    return {
        "p": p, "J": J,
        "p_fun": p_fun, "J_fun": J_fun,
        "x_star": x_star,
        "q_sol": q_sol, "iters": iters,
        "q_hist": q_hist, "x_hist": x_hist,
        "e_hist": e_hist,
        "sig_hist": sig_hist,
    }

def get_perturbed_jacobian(J, eps_max):
    ''' Given a Jacobian matrix J, return a perturbed version of J where each element is independently scaled by a random factor in the range [1, eps_max]. This simulates the effect of approximation errors in the Jacobian. '''
    J = np.array(J, dtype=float)

    scale = np.max(np.abs(J)) #normalize!!!
    if scale == 0:
        scale = 1.0

    J_norm = J / scale

    perturb = np.random.uniform(0, eps_max, J_norm.shape)
    J_perturbed = J_norm + perturb

    return J_perturbed * scale

def valid_jacobian_perturbation_bounds_main():
    '''for a given maximum error scaling epsilon_limit, generate a scaling matrix m_eps where each element is sampled independently with m_eps_i_j ~ uniform(1,eps_limit). 
    eps_lim_bounds = (-low,+high) something like (-1.5, 3)
    for each value in that range, incrementing by 0.10 each time or something, perform 30 visual servoing runs and see how much error we accumulate (how to measure path completion proportion? measure how much error we reduce by each step)
    then find the cutoff epsilon_limit value where we start to see significant degradation in convergence (e.g., we fail to reduce error by at least 50% on average across runs, or we fail to converge within max iterations, etc). This will give us an empirical bound on how much Jacobian perturbation we can tolerate before convergence degrades significantly. We can also plot the convergence curves for different epsilon_limit values to visualize the effect of Jacobian perturbation on convergence.
    '''
    # dof = 3
    # planar=0
    for dof in [2, 3]:
        for planar in [0, 1]:

            structure_type = "planar" if planar else "alt"
            vars = sp.symbols(f"q0:{dof}", real=True)
            robot_name = f"dof{dof}_{structure_type}"
            joint_ranges = [(pi/6, pi/2)] * dof

            tol=1e-1


            sin_list_true = [sp.sin] * dof
            cos_list_true = [sp.cos] * dof

            chord_flag=False

            eps_lim_values = np.linspace(-3, 3, 100)  # 46 values from -1.5 to 3 in steps of 0.1
            results = []

            p_true = get_robot_fkin_expr(name=robot_name,vars=vars, cameras=[cam1, cam2])
            for eps in eps_lim_values:
                #make q0 and q_star random within jointn lim:
                q0 = np.random.uniform(joint_ranges[0][0], joint_ranges[0][1], size=dof)
                q_star = np.random.uniform(joint_ranges[0][0], joint_ranges[0][1], size=dof)
                    # --- damped pseudo-inverse Jacobian (stable)
                runs=[]
                for run in range(50):
                    runs.append(script(sin_list_true, cos_list_true, vars, robot_name, q0, q_star, damping=1e-3, chord_newton=chord_flag, jacobian_eps_max=eps,p=p_true,p_true=p_true, tol=tol))
                avg=np.mean([r["e_hist"][-1] for r in runs])
                print(f"eps_lim: {eps:.2f}, avg final error: {avg:.4f}")
                results.append(avg)

            #now the convergence error for each script can be the last entry of e_hist, and we can plot the convergence error vs eps_lim_values to see how the Jacobian perturbation affects convergence. We can also look at the number of iterations taken for convergence as another metric.
            convergence_errors = results
            # plt.figure()
            # plt.plot(eps_lim_values, convergence_errors, marker="o")
            # plt.xlabel("Jacobian perturbation limit (eps_lim)")
            # plt.ylabel("Final convergence error ||e||")
            # plt.title("Effect of Jacobian perturbation on convergence")
            # plt.grid(True)
            # plt.show()

            # save the results with np 
            np.savez(f"jacobian_perturbation_results_{robot_name}_vs_additive_eps_signed.npz", eps_lim_values=eps_lim_values, convergence_errors=convergence_errors)

            

def PLOT_using_chord_for_convergence_true_real():
    ''' plot over joint space using chord method and the real world true model'''
    dof = 3
    planar=0
    structure_type = "planar" if planar else "alt"
    vars = sp.symbols(f"q0:{3}", real=True)
    robot_name = f"dof{dof}_{structure_type}"
    joint_ranges = [JOINT_RANGE] * dof


    q0     = np.array([pi/5, pi/3, pi/3][:dof], dtype=float)
    q_star = np.array([pi/5, pi/2, pi/4][:dof], dtype=float)
    

    p_true=get_robot_fkin_expr(robot_name, vars=vars, cameras=None)
    print(f"p_true: {p_true}")
    p = p_true
    print(f"p: {p}")
    real_p_true_fn = get_robot_fkin_expr(robot_name, vars, sin_hat_list=[], cos_hat_list=[], cameras=None)
    print("real_p_true_fn:", real_p_true_fn)
    results = evaluate_joint_space(joint_ranges, sin_list=[sin]*10, cos_list=[cos]*10, q0=q0, vars=vars, robot_name=robot_name, p=p,p_true=p_true, chord_newton=True)
    print("RESULTS:\n",results)
    plot_convergence_results_not_vs(results)




def eval_joint_space_main():
    dof = 3
    planar=0
    structure_type = "planar" if planar else "alt"
    vars = sp.symbols(f"q0:{3}", real=True)
    robot_name = f"dof{dof}_{structure_type}"
    joint_ranges = [JOINT_RANGE] * dof

    planar_combinations = get_robot_possible_linear_model_combinations(name=robot_name)
    sin_list_hat,cos_list_hat = planar_combinations[1]
    sin_list_hat.append(sin) #this is just here to not break indexing but it seriously doesnt affect anything
    cos_list_hat.append(cos) #same as above comment!


    q0     = np.array([pi/4, pi/4, pi/3][:dof], dtype=float)
    q_star = np.array([pi/5, pi/2, pi/4][:dof], dtype=float)

    x = sp.symbols('x')
    expr = sin_list_hat[0](x)
    f = sp.lambdify(x, expr, "numpy")
    xs = np.linspace(LOWER_SINUSOID, UPPER_SINUSOID, 500)
    ys = f(xs)

    plt.plot(xs, ys)
    plt.ylim(-2, 2)
    plt.show()

    x = sp.symbols('x')
    expr = cos_list_hat[0](x)
    f = sp.lambdify(x, expr, "numpy")
    xs = np.linspace(LOWER_SINUSOID, UPPER_SINUSOID, 500)
    ys = f(xs)

    plt.plot(xs, ys)
    plt.ylim(-2, 2)
    plt.show()

    p_true=get_robot_fkin_expr(robot_name, vars=vars, cameras=[cam1,cam2])
    print(f"p_true: {p_true}")
    p = get_vs_fkin_expr(robot_name, vars, sin_hat_list=sin_list_hat, cos_hat_list=cos_list_hat)
    print(f"p: {p}")
    real_p_true_fn = get_robot_fkin_expr(robot_name, vars, sin_hat_list=[], cos_hat_list=[], cameras=None)
    print("real_p_true_fn:", real_p_true_fn)
    results = evaluate_joint_space(joint_ranges, sin_list_hat, cos_list_hat, q0, vars, robot_name, p=p,p_true=p_true)
    plot_convergence_results(results, real_p_true_fn=real_p_true_fn)

# --------------------------
# main (updated: run 4 cases + CLI flag for chord/newton)
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chord_newton",
        action="store_true",
        help="If set, runs chord Newton (fixed Jacobian) for the selected runs."
    )
    parser.add_argument(
        "--run_both",
        action="store_true",
        help="If set, run BOTH chord and full Newton and plot 4-way comparisons (recommended)."
    )
    args = parser.parse_args()

    dof = 2
    planar=1
    structure_type = "planar" if planar else "alt"
    vars = sp.symbols(f"q0:{dof}", real=True)
    robot_name = f"dof{dof}_{structure_type}"
    joint_ranges = [(pi/6, pi/2)] * dof

    q0     = np.array([pi/4, pi/3, pi/3][:dof], dtype=float)
    q_star = np.array([pi/5, pi/2, pi/4][:dof], dtype=float)

    # --- build piecewise trig lists once
    sin_list_hat = []
    cos_list_hat = []
    # for _ in range(dof): #### OUTDATED since now we have the handmade linear combos
    #     sin_knots = np.linspace(-2*pi, 2*pi, 15)
    #     cos_knots = np.linspace(-2*pi, 2*pi, 15)
    #     sin_list_hat.append(create_piecewise_sinusoid(sp.sin, sin_knots))
    #     cos_list_hat.append(create_piecewise_sinusoid(sp.cos, cos_knots))
    planar_combinations = get_robot_possible_linear_model_combinations(name=robot_name)
    sin_list_hat,cos_list_hat = planar_combinations[1]
    sin_list_hat.append(sin) #this is just here to not break indexing but it seriously doesnt affect anything
    cos_list_hat.append(cos) #same as above comment!


    sin_list_true = [sp.sin] * dof
    cos_list_true = [sp.cos] * dof

    # choose which solver modes to run
    modes = []
    if args.run_both:
        modes = [("chord", True), ("newton", False)]
    else:
        modes = [("chord" if args.chord_newton else "newton", args.chord_newton)]

    # run (solver mode) x (model/true)
    outs = {}  # outs[(mode_name, "true"/"model")] = script output

    for mode_name, chord_flag in modes:
        outs[(mode_name, "true")] = script(
            sin_list=sin_list_true,
            cos_list=cos_list_true,
            vars=vars,
            robot_name=robot_name,
            q0=q0,
            q_star=q_star,
            chord_newton=chord_flag
        )
        outs[(mode_name, "model")] = script(
            sin_list=sin_list_hat,
            cos_list=cos_list_hat,
            vars=vars,
            robot_name=robot_name,
            q0=q0,
            q_star=q_star,
            chord_newton=chord_flag
        )

    # For 4-way plots we need both modes
    if not args.run_both:
        mode_name, _ = modes[0]
        out_t = outs[(mode_name, "true")]
        out_m = outs[(mode_name, "model")]
        x_star = out_t["x_star"]

        e_t = np.linalg.norm(out_t["x_hist"] - x_star.reshape(1, -1), axis=1)
        e_m = np.linalg.norm(out_m["x_hist"] - x_star.reshape(1, -1), axis=1)

        # reuse your old 2-way plots (or keep these as-is)
        plot_cartesian_trajectory(out_t["x_hist"], out_m["x_hist"], out_t["x_hist"], out_m["x_hist"], x_star)
        plot_joint_trajectory(out_t["q_hist"], out_m["q_hist"], out_t["q_hist"], out_m["q_hist"])
        plot_error_history(e_t, e_m, e_t, e_m)
        plot_spectral_norm_history(out_t["sig_hist"], out_m["sig_hist"], out_t["sig_hist"], out_m["sig_hist"])
        return

    # 4-way plot data
    out_ct = outs[("chord", "true")]
    out_cm = outs[("chord", "model")]
    out_nt = outs[("newton", "true")]
    out_nm = outs[("newton", "model")]

    # use a single target for fair comparison (true target from true FK at q_star)
    x_star = out_ct["x_star"]

    e_ct = np.linalg.norm(out_ct["x_hist"] - x_star.reshape(1, -1), axis=1)
    e_cm = np.linalg.norm(out_cm["x_hist"] - x_star.reshape(1, -1), axis=1)
    e_nt = np.linalg.norm(out_nt["x_hist"] - x_star.reshape(1, -1), axis=1)
    e_nm = np.linalg.norm(out_nm["x_hist"] - x_star.reshape(1, -1), axis=1)

    plot_cartesian_trajectory(out_ct["x_hist"], out_cm["x_hist"], out_nt["x_hist"], out_nm["x_hist"], x_star)
    plot_joint_trajectory(out_ct["q_hist"], out_cm["q_hist"], out_nt["q_hist"], out_nm["q_hist"])
    plot_error_history(e_ct, e_cm, e_nt, e_nm)
    plot_spectral_norm_history(out_ct["sig_hist"], out_cm["sig_hist"], out_nt["sig_hist"], out_nm["sig_hist"])

    print("\n--- Summary (4-way) ---")
    print(f"Target x*: {x_star}")
    print(f"Chord  True:  iters={out_ct['iters']}, final ||e||={e_ct[-1]}")
    print(f"Chord  Model: iters={out_cm['iters']}, final ||e||={e_cm[-1]}")
    print(f"Newton True:  iters={out_nt['iters']}, final ||e||={e_nt[-1]}")
    print(f"Newton Model: iters={out_nm['iters']}, final ||e||={e_nm[-1]}")

def get_robot_possible_linear_model_combinations(name:str='dof2_planar', lower=LOWER_SINUSOID, upper=UPPER_SINUSOID):
    '''>>> (pi/6)-2*(pi/2)
    -2.6179938779914944
    >>> 3*pi/2
    4.71238898038469'''
        
    # sin_knots_6 = [-6.28318531, -4.71238898, -1.57079633, 1.57079633,  4.71238898,  6.28318531]
    # sin_knots_10 = [-6.28318531, -5.23598776, -4.1887902, -2.0943951 , -1.04719755, 1.04719755,  2.0943951,   4.1887902 ,  5.23598776,  6.28318531]
    # sin_knots_14 = [-6.28318531, -5.49778714, -4.71238898, -3.92699082, -2.35619449, -1.57079633, -0.78539816, 0.78539816, 1.57079633,  2.35619449,    3.92699082,  4.71238898, 5.49778714, 6.28318531]
    # sin_knots_18 =[-6.28318531, -5.65486678, -5.02654825, -4.39822972, -3.76991118, -2.51327412, -1.88495559, -1.25663706, -0.62831853, 
    #               0.62831853,  1.25663706,  1.88495559,  2.51327412,  3.76991118,  4.39822972,  5.02654825,  5.65486678, 6.28318531]

    # cos_knots_5 = [-6.28318531, -3.14159265, 0, 3.14159265, 6.28318531]
    # cos_knots_10 = [-6.28318531, -4.71238898, -3.14159265, -1.57079633, 0, 1.57079633, 3.14159265, 4.71238898, 6.28318531]
    # cos_knots_13 = [-6.28318531, -5.49778714, -3.92699082, -3.14159265, -2.35619449, -0.78539816,  0.        ,  0.78539816,  2.35619449,  3.14159265,  3.92699082,  5.49778714,  6.28318531]
  
  
    sin_knots_A = [lower, -1.57079633, 1.57079633,  upper]
    sin_knots_B = [lower, -2.0943951 , -1.04719755, 1.04719755,  2.0943951,   4.1887902 ,  upper]
    sin_knots_C = [lower, -2.35619449, -1.57079633, -0.78539816, 0.78539816, 1.57079633,  2.35619449,    3.92699082, upper]
    sin_knots_D=[lower, -2.51327412, -1.88495559, -1.25663706, -0.62831853, 0.62831853,  1.25663706,  1.88495559,  2.51327412,  3.76991118,  4.39822972, upper]
    sin_hat_A =create_piecewise_sinusoid(sp.sin, sin_knots_A)
    sin_hat_B= create_piecewise_sinusoid(sp.sin, sin_knots_B)
    sin_hat_C = create_piecewise_sinusoid(sp.sin, sin_knots_C)
    sin_hat_D = create_piecewise_sinusoid(sp.sin, sin_knots_D)

    cos_knots_A = [lower, 0, 3.14159265, upper]
    cos_knots_B = [lower,  -0.52359878,  0.52359878,  2.61799388,  3.66519143, upper]
    cos_knots_C = [lower, -2.35619449, -0.78539816,  0.        ,  0.78539816,  2.35619449,  3.14159265,  3.92699082,  upper]
    cos_knots_D =[lower, -2.31485774, -0.99208189, -0.33069396,  0.33069396,  0.99208189,  2.31485774,  pi-0.33, pi+0.33, 3.9683275671795863, upper]
    cos_hat_A=create_piecewise_sinusoid(sp.cos, cos_knots_A)
    cos_hat_B=create_piecewise_sinusoid(sp.cos, cos_knots_B)
    cos_hat_C=create_piecewise_sinusoid(sp.cos, cos_knots_C)
    cos_hat_D=create_piecewise_sinusoid(sp.cos, cos_knots_D)

    # # cos_hat=create_piecewise_sinusoid(sp.cos, np.linspace(lower, upper, 10))
    # x = sp.symbols('x')
    # expr = cos_hat(x)
    # f = sp.lambdify(x, expr, "numpy")
    # xs = np.linspace(lower+0.01, upper-0.01, 1000)
    # ys = f(xs)

    # plt.plot(xs, ys)
    # plt.ylim(-2, 2)
    # plt.show()


    models={"dof2_planar": [[[sin_hat_A, sin_hat_A], [cos_hat_A, cos_hat_A]], 
                            [[sin_hat_B, sin_hat_B], [cos_hat_B, cos_hat_B]],
                           [ [sin_hat_C, sin_hat_C], [cos_hat_C, cos_hat_C]],
                           [ [sin_hat_D, sin_hat_D], [cos_hat_D, cos_hat_D]]],
                        #    [ [sin_hat_A, sin_hat_B], [cos_hat_A, cos_hat_B]],
                        #   [  [sin_hat_A, sin_hat_C], [cos_hat_A, cos_hat_C]],
                        #   [  [sin_hat_A, sin_hat_D], [cos_hat_A, cos_hat_D]],
                        #     [[sin_hat_B, sin_hat_A], [cos_hat_B, cos_hat_A]],
                        #     [[sin_hat_B, sin_hat_C], [cos_hat_B, cos_hat_C]],
                        #   [  [sin_hat_B, sin_hat_D], [cos_hat_B, cos_hat_D]],
                        #    [ [sin_hat_C, sin_hat_A], [cos_hat_C, cos_hat_A]],
                        #    [ [sin_hat_C, sin_hat_B], [cos_hat_C, cos_hat_B]],
                        #    [ [sin_hat_C, sin_hat_D], [cos_hat_C, cos_hat_D]],
                        #    [ [sin_hat_D, sin_hat_A], [cos_hat_D, cos_hat_A]],
                        #     [[sin_hat_D, sin_hat_B], [cos_hat_D, cos_hat_B]],
                        #    [ [sin_hat_D, sin_hat_C], [cos_hat_D, cos_hat_C]]],

            "dof2_alt": [[[sin_hat_A]*3, [cos_hat_A]*3],
                         [[sin_hat_B]*3, [cos_hat_B]*3],
                         [[sin_hat_C]*3, [cos_hat_C]*3],
                         [[sin_hat_D]*3, [cos_hat_D]*3]],
    "dof3_planar":[[[sin_hat_A]*3, [cos_hat_A]*3],
                         [[sin_hat_B]*3, [cos_hat_B]*3],
                         [[sin_hat_C]*3, [cos_hat_C]*3],
                         [[sin_hat_D]*3, [cos_hat_D]*3]],
    "dof3_alt":[[[sin_hat_A]*3, [cos_hat_A]*3],
                         [[sin_hat_B]*3, [cos_hat_B]*3],
                         [[sin_hat_C]*3, [cos_hat_C]*3],
                         [[sin_hat_D]*3, [cos_hat_D]*3]]}


    return models[name]

def get_count_of_linear_pieces():
    ''' a hardcoded function just for storage and  my own convenience, not to be used in scripts'''    
    sA = 3
    cA = 3
    sB = 6
    cB = 5
    sC = 8
    cC = 8
    sD = 11
    cD = 10

    s0 = s1 = s2 = sD
    c0 = c1 = c2 = cD

    # how many parameters do we actually need to measure for each model ==> how many parameters for each line, then how many lines?
    one = 2 # mx+b
    two = 3 #mx +my + b
    three = 4 #mx + my  +mz + b

    dof3_alt = s1*two + s1*two + s2*three + s2*three + s0*one + s1*two + c1*two + c1*two + c2*three + c2*three
    dof3_planar = s0*one + s1*two + s2*three + c0*one + c1*two + c2*three
    dof2_alt = s0*one + s1*two + s1*two + c0*one + c1*two + c1*two + s1*one
    dof2_planar = s0*one + s1*two + c0*one + c1*two 

    #now this should be the proper count of how many 
    print(dof2_alt)
    print(dof2_planar)
    print(dof3_alt)
    print(dof3_planar)

    ''' jotnotes
    (NOT CAMERA PROJECTION. if camera projection then just multiply each count by TWO)
    ---      
    s0 = s1 = s2 = sA
    c0 = c1 = c2 = cA
    dof2_alt, dof2_planar, dof3_alt, dof3_planar = 21, 12, 30, 18

     ---      
    s0 = s1 = s2 = sB
    c0 = c1 = c2 = cB
    dof2_alt, dof2_planar, dof3_alt, dof3_planar = 39, 22, 56, 33

     ---      
    s0 = s1 = s2 = sC
    c0 = c1 = c2 = cC
    dof2_alt, dof2_planar, dof3_alt, dof3_planar = 56, 32, 80, 48

         ---      
    s0 = s1 = s2 = sD
    c0 = c1 = c2 = cD
    dof2_alt, dof2_planar, dof3_alt, dof3_planar = 74, 42, 106, 63

    '''


def get_vs_fkin_expr(name:str, vars, sin_hat_list=[sin]*4, cos_hat_list=[cos]*4, cameras=[cam1,cam2]):
    ''' to get vs model with constant depth '''
    try:
        t0,t1,t2 = vars[0],vars[1],vars[2]
        sin_hat_list[2]
    except:
        t0,t1,t2= vars[0],vars[1],0
        sin_hat_list.append(sin)
        cos_hat_list.append(cos)
        
    #later can add to choose a specific camera but for now lets just do both cam1 and cam2

    dof2_planar_cam1 = sp.Matrix([[0.978142303813579*sin_hat_list[0](t0) + 0.978142303813579*sin_hat_list[1](t0 + t1)], [0.195628460762716*(5*cos_hat_list[0](t0) + 5*cos_hat_list[1](t0 + t1))*cos(pi/16) - 4.89071151906789*sin(pi/16) + 0.978142303813579*cos(pi/16)]])
    # s0 + s1 + c0 + c1 

    dof2_planar_cam2 = sp.Matrix([[0.702528441024851*sin_hat_list[0](t0) + 0.702528441024851*sin_hat_list[1](t0 + t1)], [0.351264220512425*sqrt(2)*(cos_hat_list[0](t0) + cos_hat_list[1](t0 + t1))]])
    # s0 + s1 + c0 + c1

    dof3_planar_cam1 = sp.Matrix([[1.00224812250043*sin_hat_list[0](t0) + 1.00224812250043*sin_hat_list[1](t0 + t1) + 1.00224812250043*sin_hat_list[2](t0 + t1 + t2)], [0.200449624500085*(5*cos_hat_list[0](t0) + 5*cos_hat_list[1](t0 + t1) + 5*cos_hat_list[2](t0 + t1 + t2))*cos(pi/16) - 5.01124061250214*sin(pi/16) + 1.00224812250043*cos(pi/16)]])
    # s0 + s1 + s2 + c0 + c1 + c2

    dof3_planar_cam2 = sp.Matrix([[0.749453228499139*sin_hat_list[0](t0) + 0.749453228499139*sin_hat_list[1](t0 + t1) + 0.749453228499139*sin_hat_list[2](t0 + t1 + t2)], [0.374726614249569*sqrt(2)*(cos_hat_list[0](t0) + cos_hat_list[1](t0 + t1) + cos_hat_list[2](t0 + t1 + t2))]])
    # s0 + s1 + s2 + c0 + c1 + c2

    dof2_alt_cam1 = sp.Matrix([[1.12635211389101*sin_hat_list[0](t0) + 0.563176056945505*sin_hat_list[1](t0 - t1) + 0.563176056945505*sin_hat_list[1](t0 + t1)], [0.225270422778202*(5*cos_hat_list[0](t0) + 2.5*cos_hat_list[1](t0 - t1) + 2.5*cos_hat_list[1](t0 + t1))*cos(pi/16) + 1.12635211389101*sin(pi/16)*sin_hat_list[1](t1) - 5.63176056945505*sin(pi/16) + 1.12635211389101*cos(pi/16)]])
    # s0 + s1 + s1 + c0 + c1 + c1 + s1

    dof2_alt_cam2 = sp.Matrix([[0.711026177453064*sin_hat_list[0](t0) + 0.355513088726532*sin_hat_list[1](t0 - t1) + 0.355513088726532*sin_hat_list[1](t0 + t1)], [0.355513088726532*sqrt(2)*(cos_hat_list[0](t0) + 0.5*cos_hat_list[1](t0 - t1) + 0.5*cos_hat_list[1](t0 + t1)) + 0.355513088726532*sqrt(2)*sin_hat_list[1](t1)]])
    # s0 + s1 + s1 + c0 + c1 + c1 + s1

    dof3_alt_cam1 = sp.Matrix([[0.688425842080919*sin_hat_list[1](t0 - t1) + 0.688425842080919*sin_hat_list[1](t0 + t1) - 0.688425842080919*sin_hat_list[2](-t0 + t1 + t2) + 0.688425842080919*sin_hat_list[2](t0 + t1 + t2)], [0.275370336832368*(5*sin_hat_list[0](t1) + 5*sin_hat_list[1](t1 + t2))*sin(pi/16) + 0.275370336832368*(2.5*cos_hat_list[1](t0 - t1) + 2.5*cos_hat_list[1](t0 + t1) + 2.5*cos_hat_list[2](-t0 + t1 + t2) + 2.5*cos_hat_list[2](t0 + t1 + t2))*cos(pi/16) - 6.88425842080919*sin(pi/16) + 1.37685168416184*cos(pi/16)]])
    # s1 + s1 + s2 + s2 + s0 + s1 + c1 + c1 + c2 + c2

    dof3_alt_cam2 = sp.Matrix([[0.407416750844502*sin_hat_list[1](t0 - t1) + 0.407416750844502*sin_hat_list[1](t0 + t1) - 0.407416750844502*sin_hat_list[2](-t0 + t1 + t2) + 0.407416750844502*sin_hat_list[2](t0 + t1 + t2)], [0.407416750844502*sqrt(2)*(sin_hat_list[0](t1) + sin_hat_list[1](t1 + t2)) + 0.407416750844502*sqrt(2)*(0.5*cos_hat_list[1](t0 - t1) + 0.5*cos_hat_list[1](t0 + t1) + 0.5*cos_hat_list[2](-t0 + t1 + t2) + 0.5*cos_hat_list[2](t0 + t1 + t2))]])
    # s1 + s1 + s2 + s2 + s0 + s1 + c1 + c1 + c2 + c2


    expr_map = {
        "dof2_planar_cam1": dof2_planar_cam1,
        "dof2_planar_cam2": dof2_planar_cam2,
        "dof3_planar_cam1": dof3_planar_cam1,
        "dof3_planar_cam2": dof3_planar_cam2,
        "dof2_alt_cam1": dof2_alt_cam1,
        "dof2_alt_cam2": dof2_alt_cam2,
        "dof3_alt_cam1": dof3_alt_cam1,
        "dof3_alt_cam2": dof3_alt_cam2,

        "dof2_planar": sp.Matrix.vstack(dof2_planar_cam1, dof2_planar_cam2),
        "dof3_planar": sp.Matrix.vstack(dof3_planar_cam1, dof3_planar_cam2),
        "dof2_alt": sp.Matrix.vstack(dof2_alt_cam1, dof2_alt_cam2),
        "dof3_alt": sp.Matrix.vstack(dof3_alt_cam1, dof3_alt_cam2),
    }

    if name not in expr_map:
        valid = ", ".join(expr_map.keys())
        raise ValueError(f"Unknown name '{name}'. Valid options are: {valid}")
    
    # how many linear pieces created?
    '''
    for dof2,
    
    '''

    return expr_map[name]
        
    # dof2_planar_cam1 = sp.Matrix([[0.978142303813579*sin(t0) + 0.978142303813579*sin(t0 + t1)], [0.195628460762716*(5*cos(t0) + 5*cos(t0 + t1))*cos(pi/16) - 4.89071151906789*sin(pi/16) + 0.978142303813579*cos(pi/16)]])

    # dof2_planar_cam2 = sp.Matrix([[0.702528441024851*sin(t0) + 0.702528441024851*sin(t0 + t1)], [0.351264220512425*sqrt(2)*(cos(t0) + cos(t0 + t1))]])

    # dof3_planar_cam1 = sp.Matrix([[1.00224812250043*sin(t0) + 1.00224812250043*sin(t0 + t1) + 1.00224812250043*sin(t0 + t1 + t2)], [0.200449624500085*(5*cos(t0) + 5*cos(t0 + t1) + 5*cos(t0 + t1 + t2))*cos(pi/16) - 5.01124061250214*sin(pi/16) + 1.00224812250043*cos(pi/16)]])

    # dof3_planar_cam2 = sp.Matrix([[0.749453228499139*sin(t0) + 0.749453228499139*sin(t0 + t1) + 0.749453228499139*sin(t0 + t1 + t2)], [0.374726614249569*sqrt(2)*(cos(t0) + cos(t0 + t1) + cos(t0 + t1 + t2))]])

    # dof2_alt_cam1 = sp.Matrix([[1.12635211389101*sin(t0) + 0.563176056945505*sin(t0 - t1) + 0.563176056945505*sin(t0 + t1)], [0.225270422778202*(5*cos(t0) + 2.5*cos(t0 - t1) + 2.5*cos(t0 + t1))*cos(pi/16) + 1.12635211389101*sin(pi/16)*sin(t1) - 5.63176056945505*sin(pi/16) + 1.12635211389101*cos(pi/16)]])

    # dof2_alt_cam2 = sp.Matrix([[0.711026177453064*sin(t0) + 0.355513088726532*sin(t0 - t1) + 0.355513088726532*sin(t0 + t1)], [0.355513088726532*sqrt(2)*(cos(t0) + 0.5*cos(t0 - t1) + 0.5*cos(t0 + t1)) + 0.355513088726532*sqrt(2)*sin(t1)]])

    # dof3_alt_cam1 = sp.Matrix([[0.688425842080919*sin(t0 - t1) + 0.688425842080919*sin(t0 + t1) - 0.688425842080919*sin(-t0 + t1 + t2) + 0.688425842080919*sin(t0 + t1 + t2)], [0.275370336832368*(5*sin(t1) + 5*sin(t1 + t2))*sin(pi/16) + 0.275370336832368*(2.5*cos(t0 - t1) + 2.5*cos(t0 + t1) + 2.5*cos(-t0 + t1 + t2) + 2.5*cos(t0 + t1 + t2))*cos(pi/16) - 6.88425842080919*sin(pi/16) + 1.37685168416184*cos(pi/16)]])

    # dof3_alt_cam2 = sp.Matrix([[0.407416750844502*sin(t0 - t1) + 0.407416750844502*sin(t0 + t1) - 0.407416750844502*sin(-t0 + t1 + t2) + 0.407416750844502*sin(t0 + t1 + t2)], [0.407416750844502*sqrt(2)*(sin(t1) + sin(t1 + t2)) + 0.407416750844502*sqrt(2)*(0.5*cos(t0 - t1) + 0.5*cos(t0 + t1) + 0.5*cos(-t0 + t1 + t2) + 0.5*cos(t0 + t1 + t2))]])

    return sp.Matrix.vstack(*uvs)

def get_robot_fkin_expr(name: str, vars, sin_hat_list=[], cos_hat_list=[], cameras=[cam1,cam2]):
    ''' TRUE fkin, no approximations other than using sin and cos as approximations!
    
    return the fkin for dof2 planar, dof2 joint 1 rot about z axis and joint 2 rot about y axis, dof 3 planar, and dof 3 Matrix([[-x - 0.3*sin(t1)*sin(t2)*cos(t0) + 0.3*cos(t0)*cos(t1)*cos(t2) + 0.55*cos(t0)*cos(t1)], [-y - 0.3*sin(t0)*sin(t1)*sin(t2) + 0.3*sin(t0)*cos(t1)*cos(t2) + 0.55*sin(t0)*cos(t1)], [-z + 0.3*sin(t1)*cos(t2) + 0.55*sin(t1) + 0.3*sin(t2)*cos(t1)]])
     but all the fkin is expressed as linear combination of sin and cos. for instance instead of  cos(x)cos(y) - sin(x)sin(y) which has quadratic degree if sin and cos are repr by linear model, we can rewrite as cos(x+y) which is linear in sin and cos. This way, we can directly substitute the piecewise linear approximations of sin and cos into the fkin expression without increasing the degree of the approximation. 
     '''
    if sin_hat_list == []:
        sin_hat_list = [sp.sin] * (1+len(vars))
    if cos_hat_list == []:
        cos_hat_list = [sp.cos] * (1+len(vars))

    #max argument to sin or cos given joint boundaries u and l: max=u*3, min=u-2*l
    t0 = vars[0]
    t1 = vars[1] if len(vars) > 1 else 0
    t2 = vars[2] if len(vars) > 2 else 0

    while len(sin_hat_list) < 4:
        sin_hat_list.append(sp.sin)
        cos_hat_list.append(sp.cos)  

    L0=L1=L2=1

    models = {

    # -------------------------------------------------
    # 1) PLANAR 2 DOF  (z, z)
    # -------------------------------------------------
    "dof2_planar": sp.Matrix([
        L0*cos_hat_list[0](t0) + L1*cos_hat_list[1](t0 + t1),
        L0*sin_hat_list[0](t0) + L1*sin_hat_list[1](t0 + t1),
        0
    ]),
    # uses c0 + c1 + s0 + s1 linear pieces

    # -------------------------------------------------
    # 2) 2 DOF (z then local y)
    # fully expanded as linear sin/cos sums
    # -------------------------------------------------
    "dof2_alt": sp.Matrix([
        L0*cos_hat_list[0](t0)
        + 0.5*L1*(cos_hat_list[1](t0 + t1) + cos_hat_list[2](t0 - t1)),

        L0*sin_hat_list[0](t0)
        + 0.5*L1*(sin_hat_list[1](t0 + t1) + sin_hat_list[2](t0 - t1)),

        L1*sin_hat_list[0](t1)
    ]),
    # uses c0 + c1 + c2 + s0 + s1 + s2 + s0 linear pieces

    # -------------------------------------------------
    # 3) PLANAR 3 DOF  (z, z, z)
    # -------------------------------------------------
    "dof3_planar": sp.Matrix([
          L0*cos_hat_list[0](t0)
        + L1*cos_hat_list[1](t0 + t1)
        + L2*cos_hat_list[2](t0 + t1 + t2),

          L0*sin_hat_list[0](t0)
        + L1*sin_hat_list[1](t0 + t1)
        + L2*sin_hat_list[2](t0 + t1 + t2),

        0
    ]),
    # uses c0 + c1 + c2 + s0 + s1 + s2

    # -------------------------------------------------
    # 4) 3 DOF (dylan)
    # -------------------------------------------------
    "dof3_alt": sp.Matrix([
        0.5*L0*(cos_hat_list[0](t0 + t1) + cos_hat_list[1](t0 - t1))+0.5*L1*(cos_hat_list[2](t0 + t1 + t2) + cos_hat_list[2](t0 - t1 - t2)),

        0.5*L0*(sin_hat_list[0](t0 + t1) + sin_hat_list[1](t0 - t1))+ 0.5*L1*(sin_hat_list[2](t0 + t1 + t2) + sin_hat_list[2](t0 - t1 - t2)),

        L0*sin_hat_list[0](t1) + L1*sin_hat_list[1](t1 + t2) 
    ])

    # c0 + c1 + c2 + c2 + s0 + s1 + s2 + s2 + s0 + s1

    }

       # # -------------------------------------------------
    # # 1) PLANAR 2 DOF  (z, z)
    # # -------------------------------------------------
    # "dof2_planar": sp.Matrix([
    #     L0*sp.cos(t0) + L1*sp.cos(t0 + t1),
    #     L0*sp.sin(t0) + L1*sp.sin(t0 + t1),
    #     0
    # ]),


    # # -------------------------------------------------
    # # 2) 2 DOF (z then local y)
    # # fully expanded as linear sin/cos sums
    # # -------------------------------------------------
    # "dof2_alt": sp.Matrix([
    #     L0*sp.cos(t0)
    #     + 0.5*L1*(sp.cos(t0 + t1) + sp.cos(t0 - t1)),

    #     L0*sp.sin(t0)
    #     + 0.5*L1*(sp.sin(t0 + t1) + sp.sin(t0 - t1)),
    #     L1*sp.sin(t1)
    # ]),


    # # -------------------------------------------------
    # # 3) PLANAR 3 DOF  (z, z, z)
    # # -------------------------------------------------
    # "dof3_planar": sp.Matrix([
    #     L0*sp.cos(t0)
    #     + L1*sp.cos(t0 + t1)
    #     + L2*sp.cos(t0 + t1 + t2),

    #     L0*sp.sin(t0)
    #     + L1*sp.sin(t0 + t1)
    #     + L2*sp.sin(t0 + t1 + t2),

    #     0
    # ]),


    # # -------------------------------------------------
    # # 4) 3 DOF (dylan)
    # # -------------------------------------------------
    # "dof3_alt": sp.Matrix([
    #     0.5*L0*(sp.cos(t0 + t1) + sp.cos(t0 - t1))+0.5*L1*(sp.cos(t0 + t1 + t2) + sp.cos(t0 - t1 - t2)),

    #     0.5*L0*(sp.sin(t0 + t1) + sp.sin(t0 - t1))+ 0.5*L1*(sp.sin(t0 + t1 + t2) + sp.sin(t0 - t1 - t2)),

    #     L0*sp.sin(t1) + L1*sp.sin(t1 + t2) 
    # ])

    p_vec = models[name]

    # If no cameras passed, return world position p_vec (3x1)
    if cameras is None:
        return p_vec

    # If cameras passed, return stacked image coords: (2,) or (4,)
    uvs = []
    for cam in cameras:
        uv = cam.projectpoint(p_vec)   # 2x1 sympy
        uvs.append(uv)
    return sp.Matrix.vstack(*uvs)

 

def _additive_terms(expr):
    """Return (terms_list, expanded_expr) where expanded_expr is trig-expanded."""
    expr = sp.expand_trig(sp.simplify(expr))
    terms = list(sp.Add.make_args(expr))
    return terms, expr


def plot_basis_surfaces_xy(p_vec, t0, t1, t0_range, t1_range, fixed_subs=None, n=120,
                           wire_stride=6, title_prefix=""):
    """
    p_vec: sp.Matrix([px,py,pz]) or length-3 list of sympy exprs
    Plots 3D surfaces over (t0,t1) for outputs x and y:
      - each additive term as a wireframe
      - total sum as a surface
    fixed_subs: dict fixing any other symbols (e.g., {t2: 0.3, L0:1, L1:1})
    """
    if fixed_subs is None:
        fixed_subs = {}

    p_vec = sp.Matrix(p_vec)
    px = sp.simplify(p_vec[0].subs(fixed_subs))
    py = sp.simplify(p_vec[1].subs(fixed_subs))

    T0 = np.linspace(float(t0_range[0]), float(t0_range[1]), n)
    T1 = np.linspace(float(t1_range[0]), float(t1_range[1]), n)
    T0g, T1g = np.meshgrid(T0, T1, indexing="xy")

    for expr, out_name in [(px, "x"), (py, "y")]:
        terms, expr_full = _additive_terms(expr)

        # Lambdify: depends on (t0,t1) only (since fixed_subs removed other symbols)
        f_full = sp.lambdify((t0, t1), expr_full, "numpy")
        f_terms = [sp.lambdify((t0, t1), ti, "numpy") for ti in terms]

        Z_full = np.array(f_full(T0g, T1g), dtype=float)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot sum as a surface
        ax.plot_surface(T0g, T1g, Z_full, alpha=0.55)

        # Plot each term as wireframe (so they "overlap" visibly)
        for k, ft in enumerate(f_terms):
            Zk = np.array(ft(T0g, T1g), dtype=float)
            ax.plot_wireframe(
                T0g, T1g, Zk,
                rstride=wire_stride, cstride=wire_stride,
                linewidth=0.7
            )

        ax.set_xlabel(str(t0))
        ax.set_ylabel(str(t1))
        ax.set_zlabel(out_name)
        ax.set_title(f"{title_prefix}{out_name}(t0,t1): sum(surface) + terms(wireframes)")
        plt.show()


def evaluate_joint_space(joint_ranges, sin_list, cos_list, q0, vars, robot_name, damping=1e-3, tol=1e-1, p=None, p_true=None, chord_newton= False):
   
    # print("plotting basis surfaces for x and y...")
    # plot_basis_surfaces_xy(p, vars[0], vars[1], (np.pi/6, np.pi/2), (np.pi/6, np.pi/2))

    if robot_name == 'dof2_planar' or robot_name=='dof2_alt':
        dof=2
    if robot_name == 'dof3_planar' or robot_name=='dof3_alt':
        dof=3

    vars=vars[:dof]

    print("Evaluating Jacobian...")
    J = p.jacobian(vars)                               # 3 x dof
    # print(J)
    # #evaluate J at the initial configuration q0 to see how the piecewise approximation affects the Jacobian at the start of the trajectory
    # print("q0:",q0)
    # print("---")
    q0 = [float(v) for v in q0]

    # J_initial = np.array(J.subs({var: val for var, val in zip(vars, q0)}), dtype=float)
    # print(f"Initial Jacobian at q0={q0}:\n{J_initial}")
    #evaluate J at a cusp point of the piecewise function to see how the Jacobian behaves at a non-smooth point
    # cusp_q = [pi/6, pi/6, pi/6][:len(vars)]  # example cusp point where the piecewise function changes segments
    # J_cusp = np.array(J.subs({var: val for var, val in zip(vars, cusp_q)}), dtype=float)
    # print(f"Jacobian at cusp point q={cusp_q}:\n{J_cusp}")
    # --- numeric callables
    p_hat_fun = sp.lambdify(vars, p, "numpy")
    p_fun = sp.lambdify(vars, p_true, "numpy")
    J_fun = sp.lambdify(vars, J, "numpy")
        # knot range for numeric wrapping

    # --- visual servoing / IK error function: e(q) = x* - fkin(q)
    def f(q):
        x = np.array(p_fun(*q), dtype=float).reshape(-1)
        return x_star - x

    # --- damped pseudo-inverse Jacobian (stable)
    def J_inv(q):
        Jn = np.array(J_fun(*q), dtype=float)
        JJt = Jn @ Jn.T
        return Jn.T @ np.linalg.inv(JJt + damping * np.eye(JJt.shape[0]))


    ''' Evaluate the Newton-Raphson method across a grid of initial joint configurations. 
    This can help us understand how the piecewise approximation affects convergence across the workspace. '''
    # Create a grid of initial joint configurations
    grids = [np.linspace(r[0], r[1], num=10) for r in joint_ranges]
    # print("GRIDS:", grids)
    
    joints = np.array(np.meshgrid(*grids)).T.reshape(-1, len(joint_ranges))
    # print("joints", joints)
    
    results = []
    for q_star in joints:
         # --- define the target in Cartesian space from TRUE FK at q_star (passed in)
        x_star = np.array(p_fun(*q_star), dtype=float).reshape(-1)
        x_hat_star = np.array(p_hat_fun(*q_star), dtype=float).reshape(-1)
        q_sol, iters, q_hist, e_hist = newton_raphson(f, J_inv, q0, tol=tol, max_iter=60, chord_newton=chord_newton)
        results.append({
            "q0": q0,
            "q_star": q_star,
            "x_hat_star": x_hat_star,
            "x_star": x_star,
            "q_sol": q_sol,
            "iters": iters,
            "final_error": e_hist[-1],
            "converged": e_hist[-1] < tol
        })

    


    print("\n--- Summary of convergence across joint space ---")
    # for r in results:
    #     print(f"q0: {r['q0']}, q_star: {r['q_star']}, iters: {r['iters']}, final_error: {r['final_error']:.2e}, converged: {r['converged']}")


    return results

def split_uv(stacked_uv: np.ndarray):
    """
    stacked_uv: (T,2) for 1 cam OR (T,4) for 2 cams (u1,v1,u2,v2)
    Returns: list of arrays, each (T,2)
    """
    stacked_uv = np.asarray(stacked_uv)
    if stacked_uv.shape[1] == 2:
        return [stacked_uv]
    if stacked_uv.shape[1] == 4:
        return [stacked_uv[:, 0:2], stacked_uv[:, 2:4]]
    raise ValueError(f"Expected 2 or 4 columns, got {stacked_uv.shape[1]}")


def plot_convergence_results_not_vs(results):
    """Plot results from evaluate_joint_space."""
    dim = len(results[0]["q_star"])
    print("DIM:", dim)

    q0 = np.array([r["q0"] for r in results])
    q_stars = np.array([r["q_star"] for r in results])
    errors = np.array([r["final_error"] for r in results])
    converged = np.array([r["converged"] for r in results])

    if dim == 2:
        plt.figure()

        sc = plt.scatter(
            q_stars[:, 0],
            q_stars[:, 1],
            c=errors,
            cmap="viridis"
        )
        # mask = converged

        # plt.scatter(
        #     q_stars[~mask, 0],
        #     q_stars[~mask, 1],
        #     c="red",
        #     marker="x",
        #     label="Did not converge"
        # )

        # sc = plt.scatter(
        #     q_stars[mask, 0],
        #     q_stars[mask, 1],
        #     c=errors[mask],
        #     cmap="viridis",
        #     label="Converged"
        # )

        plt.scatter(
            q0[0][0],
            q0[0][1],
            c="black",
            s=200,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
            label="Initial q0",
            zorder=10
        )

        plt.colorbar(sc, label="Final error")
        plt.xlabel("q_star[0]")
        plt.ylabel("q_star[1]")
        plt.title("Convergence across joint space (color by final error)")
        plt.grid(True)
        plt.show()

    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # sc = ax.scatter(
        #     q_stars[:, 0],
        #     q_stars[:, 1],
        #     q_stars[:, 2],
        #     c=errors,
        #     cmap="viridis"
        # )

        q0 = np.array(results[0]["q0"])  # shape (3,)

        ax.scatter(
            q0[0],
            q0[1],
            q0[2],
            c="black",
            s=200,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
            label="Initial q0"
        )

        mask = converged

        ax.scatter(
            q_stars[~mask, 0],
            q_stars[~mask, 1],
            q_stars[~mask, 2],
            c="red",
            marker="x",
            label="Did not converge"
        )

        sc = ax.scatter(
            q_stars[mask, 0],
            q_stars[mask, 1],
            q_stars[mask, 2],

            c=errors[mask],
            cmap="viridis",
            label="Converged"
        )


        fig.colorbar(sc, ax=ax, label="Final error")
        ax.set_xlabel("q_star[0]")
        ax.set_ylabel("q_star[1]")
        ax.set_zlabel("q_star[2]")
        ax.set_title("Convergence across joint space (color by final error)")
        ax.grid(True)
        plt.show()

def plot_convergence_results(results, real_p_true_fn = None):
    """Plot results from evaluate_joint_space."""
    dim = len(results[0]["q_star"])
    x_hat_stars = np.array([r["x_hat_star"] for r in results])
    x_stars = np.array([r["x_star"] for r in results])
    q0 = np.array([r["q0"] for r in results])
    q_stars = np.array([r["q_star"] for r in results])
    errors = np.array([r["final_error"] for r in results])
    converged = np.array([r["converged"] for r in results])

    # ------------------------------------------------------------------
    # 1) Plot convergence in JOINT SPACE, colored by final error
    # ------------------------------------------------------------------
    if dim == 2:
        plt.figure()

        sc = plt.scatter(
            q_stars[:, 0],
            q_stars[:, 1],
            c=errors,
            cmap="viridis"
        )

        # Optional: mark non-converged points differently
        # mask = converged
        # plt.scatter(
        #     q_stars[~mask, 0],
        #     q_stars[~mask, 1],
        #     c="red",
        #     marker="x",
        #     label="Did not converge"
        # )

        plt.scatter(
            q0[0, 0],   # same q0 for all results
            q0[0, 1],
            c="black",
            s=200,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
            label="Initial q0",
            zorder=10
        )

        plt.colorbar(sc, label="Final error")
        plt.xlabel("q_star[0]")
        plt.ylabel("q_star[1]")
        plt.title("Convergence across joint space (color by final error)")
        plt.grid(True)
        plt.legend()
        plt.show()

    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            q_stars[:, 0],
            q_stars[:, 1],
            q_stars[:, 2],
            c=errors,
            cmap="viridis"
        )

        ax.scatter(
            q0[0, 0],   # same q0 for all results
            q0[0, 1],
            q0[0, 2],
            c="black",
            s=200,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
            label="Initial q0"
        )

        fig.colorbar(sc, ax=ax, label="Final error")
        ax.set_xlabel("q_star[0]")
        ax.set_ylabel("q_star[1]")
        ax.set_zlabel("q_star[2]")
        ax.set_title("Convergence across joint space (color by final error)")
        ax.legend()
        plt.show()

    # ------------------------------------------------------------------
    # 2) Plot the FK SHAPE in IMAGE SPACE (u, v)
    # ------------------------------------------------------------------
    x_stars_list = split_uv(x_stars)
    x_hat_list = split_uv(x_hat_stars)

    if len(x_stars_list) == 1:
        plt.figure()
        plt.scatter(
            x_stars_list[0][:, 0],
            x_stars_list[0][:, 1],
            c="black",
            label="s* true"
        )
        plt.scatter(
            x_hat_list[0][:, 0],
            x_hat_list[0][:, 1],
            c="orange",
            label="ŝ* model"
        )
        plt.xlabel("u (pixels)")
        plt.ylabel("v (pixels)")
        plt.title("Forward map samples in image space (1 camera)")
        plt.grid(True)
        plt.legend()
        plt.show()

    elif len(x_stars_list) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, ax in enumerate(axes):
            ax.scatter(
                x_stars_list[i][:, 0],
                x_stars_list[i][:, 1],
                c="black",
                label="s* true"
            )
            ax.scatter(
                x_hat_list[i][:, 0],
                x_hat_list[i][:, 1],
                c="orange",
                label="ŝ* model"
            )
            ax.set_xlabel("u (pixels)")
            ax.set_ylabel("v (pixels)")
            ax.set_title(f"Camera {i+1}")
            ax.grid(True)
            ax.legend()

        fig.suptitle("Forward map samples in image space (2 cameras)")
        plt.tight_layout()
        plt.show()

    # using the forward kinematics, plot q_star as fkim_star in the real world space to evaluate whether we can successfully converge in that area.
    # pass through each joint (q_star in q_stars) through the forward kineamtic function (real_p_true_fn)
    # then assign a color map or the errors (errors) to the real world points
    # if the real world point is indeed 3D, that is good but if it is in 2D, just add a third dimension so all of the plots are in 3D
        # ------------------------------------------------------------------
    # 3) Plot q_star mapped through real_p_true_fn in WORLD SPACE, colored by error
    # ------------------------------------------------------------------
    if real_p_true_fn is not None:
        import sympy as sp

        world_points = []

        # If real_p_true_fn is a SymPy expression/matrix, convert it to a callable
        if isinstance(real_p_true_fn, sp.MatrixBase):
            joint_syms = sorted(real_p_true_fn.free_symbols, key=lambda s: s.name)
            real_p_callable = sp.lambdify(joint_syms, real_p_true_fn, modules="numpy")

            for q_star in q_stars:
                p = np.asarray(real_p_callable(*q_star), dtype=float).reshape(-1)

                if p.size == 2:
                    p = np.array([p[0], p[1], 0.0])
                elif p.size > 3:
                    p = p[:3]
                elif p.size < 2:
                    raise ValueError("real_p_true_fn must return at least 2 coordinates.")

                world_points.append(p)

        else:
            # Assume it is already a normal Python callable
            for q_star in q_stars:
                p = np.asarray(real_p_true_fn(q_star), dtype=float).reshape(-1)

                if p.size == 2:
                    p = np.array([p[0], p[1], 0.0])
                elif p.size > 3:
                    p = p[:3]
                elif p.size < 2:
                    raise ValueError("real_p_true_fn must return at least 2 coordinates.")

                world_points.append(p)

        world_points = np.array(world_points)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            world_points[:, 0],
            world_points[:, 1],
            world_points[:, 2],
            c=errors,
            cmap="viridis"
        )

        fig.colorbar(sc, ax=ax, label="Final error")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Convergence in real world space (color by final error)")
        plt.show()
    

def jacobian_max_entry_error(model_J, true_J):

    model_J = np.array(model_J)
    true_J = np.array(true_J)

    scale = np.max(np.abs(true_J))
    if scale == 0:
        scale = 1.0

    flat_model_J = model_J.flatten()
    flat_true_J = true_J.flatten()

    upper_error_scaling = -np.inf
    lower_error_scaling = np.inf

    for m, t in zip(flat_model_J, flat_true_J):

        scaling = (m - t) / scale
        print(f"scaling = {scaling} = ({m} - {t})/{scale}")

        upper_error_scaling = max(upper_error_scaling, scaling)
        lower_error_scaling = min(lower_error_scaling, scaling)

    print(upper_error_scaling, lower_error_scaling)
    return upper_error_scaling, lower_error_scaling
        
def get_best_model_main(name):
    '''
    for the robot we have its forward kinematics expressed as p.

    p = sp Matrix ([x,y,z]) or later [u,v] or [u1,v1,u2,v2]
    
    for each dimension:
        for each additive term in that dimension: (ie each linear term sin(t1), then sin(t1+t2), then ... in [sin(t1)+sin(t1+t2)+cos(t1)+cos(t1+t2)])
            we will have a vector like [9,21,16,16] or something to rep how many linear pieces in that func.
            [9, 9, 9, 9, 9]
            []
    '''

    if name=='dof2_planar' or name=='dof2_alt':
        dof=2
    elif name=='dof3_planar' or name=='dof3_alt':
        dof=3
    vars=sp.symbols(f'q0:{dof}')
    q0     = np.array([pi/4, pi/4, pi/3][:dof], dtype=float)
    p_true = get_robot_fkin_expr(name=name, vars=vars, cameras=[cam1, cam2])
    J_true = p_true.jacobian(vars)  
    J_true_fun = sp.lambdify(vars, J_true, "numpy")
    planar_combinations = get_robot_possible_linear_model_combinations(name=name)


    print(planar_combinations)

    result_score=[]
    number_of_linear_pieces_in_the_model=[] #this does not indicate how many parmaters yet but wwe definitely need to do this
    number_of_linear_pieces_in_the_model = [21, 39, 56, 74]
    for i in range(len(planar_combinations)):
        sin_list, cos_list = planar_combinations[i]
        print(f"Model {i+1}:")
        print(f"  sin_list: {[s.__name__ for s in sin_list]}")
        print(f"  cos_list: {[c.__name__ for c in cos_list]}")
        p = get_vs_fkin_expr(name='dof2_planar', vars=sp.symbols("q0:2"), sin_hat_list=sin_list, cos_hat_list=cos_list, cameras=[cam1,cam2]) #approximated model
        J_model = p.jacobian(vars)  
        J_model_fun = sp.lambdify(vars, J_model, "numpy")

        #now we want to check two error metrics:
        # 1. check the convergence of newtons method over the joint space
        # 2. check how much error each entry had compared to the true

        # for each jointconfig in the joint space, evaluate the jacobian and count the ratio of jacobian scaling error is within bounds vs not in bounds
        joint_ranges = [JOINT_RANGE] * dof

        grids = [np.linspace(r[0], r[1], num=10) for r in joint_ranges]
    # print("GRIDS:", grids)
        
        joints = np.array(np.meshgrid(*grids)).T.reshape(-1, len(joint_ranges))
        number_of_joint_configurations = len(joints)
        # print("joints", joints)
        good_count = 0
        for j in joints:
            J_true_q = J_true_fun(*j)
            J_model_q = J_model_fun(*j)
            J_true_q = np.trunc(J_true_q * 1000) / 1000
            J_model_q = np.trunc(J_model_q * 1000) / 1000
            print("---")
            print(J_model_q)
            print(J_true_q)
            u, l = jacobian_max_entry_error(J_model_q, J_true_q )
            
            #hard code bounds for now, maybe lower = -0.5, upper = 1.5
            upper_acceptable_scaling = 1.5
            lower_acceptable_scaling = -0.5
            if lower_acceptable_scaling <= l <= u <= upper_acceptable_scaling:
                good_count+=1

        score = good_count /  number_of_joint_configurations       
        print(score)
        result_score.append(score)

    # get the best model based on convergence rate and error
    best_index = np.argmax(result_score)  # or use a weighted metric of convergence
    best_model = planar_combinations[best_index]
    print(f"Best model: {best_index+1} with perturbation score {result_score[best_index]:.2%}")

    # plot convergence rate vs error for each model
    plt.figure()
    plt.scatter(number_of_linear_pieces_in_the_model, result_score)   
    plt.xlabel("Average Convergence Rate")
    plt.ylabel("Average Final Error")
    plt.title(f"Model Comparison for {name}")
    plt.grid(True)
    plt.show()


# get_best_model_main(name='dof2_planar')
valid_jacobian_perturbation_bounds_main()
# main()
# eval_joint_space_main()
# PLOT_using_chord_for_convergence_true_real()

# def script(sin_list, cos_list, vars):
#     rotation_matrix = get_robot_rotation_matrix(sin_hat=sin_list, cos_hat=cos_list, vars)
#     fkin_function = rotation_matrix[translation column]
#     jacobian = fkin_function jacobian
#     jacobian_inv = inv of jacobian
#     total number of linear hyperplanes: # count the number of segments for each sin cos piece, or the number of linear pieces that combinatorially are created during the matrix multiplications? can we count both?
#     x, iterations = newton_raphson(fkin_function, jacobian_inv)
#     print(f"x:{x},iter:{iterations},fkin(x):{fkin_function(x)}")


# def main():
#     # initialize one of our robots via cmd line arg 
#     dof = int(cmd_line_arg)
#     vars = sp.symbols(f'q0:{dof}')
#     robot_name = f"dof{dof}"

#     print("True Sin and Cos:")
#     sin_list=[sp.sin]*dof
#     cos_list=[sp.cos]*dof
#     script(sin_list=sin_list,cos_list=cos_list, vars=vars)

#     print("Approximated Sin and Cos:")
#     sin_list=[]
#     cos_list=[]
#     knots = [[0,pi/2,pi,3*pi/2,2*pi]]
#     for i in range(dof):
#         sin_hat = create_piecewise(sp.sin, knots[i])
#         cos_hat = create_piecewise(sp.cos, knots[i])
#     script(sin_list=sin_list, cos_list=cos_list, vars=vars)



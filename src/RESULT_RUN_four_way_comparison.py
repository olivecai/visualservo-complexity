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
import matplotlib.pylab as plt
import argparse
pi=np.pi

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
            m*=1.0 #earlier i was experimenting with changing the slope but i dont think that is a good idea
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


pw = create_piecewise_sinusoid(sp.sin, np.linspace(-2*pi,2*pi,9))

x = sp.symbols('x')
expr = pw(x)
f = sp.lambdify(x, expr, "numpy")
xs = np.linspace(-2*np.pi, 2*np.pi, 1000)
ys = f(xs)

# plt.plot(xs, ys)
# plt.ylim(-2, 2)
# plt.show()


def get_robot_rotation_matrix(name: str, sin_hat: list , cos_hat: list, vars: list) -> sp.Matrix:
    ''' sin_hat: list of approximation sin functions. 
    for instance, 
    if use sin_hat_1 in the base matrix, 
    sin_hat_2 in second translation matrix, 
    then sin=[sin_hat_1, sin_hat_2]
    
    Note that sin_hat, cos_hat, vars must be the same length lists (for each link use a cos or sin approximation)'''
    Rxy_mats=[]
    Rxz_mats=[]

    dof = len(vars)
    def Rz(angle, sfun, cfun):
        c = cfun(angle)
        s = sfun(angle)
        return sp.Matrix([
            [c,  s, 0, 0],
            [-s, c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
    
    def Tx(dist):
        return sp.Matrix([
            [1, 0, 0, dist],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
    
    # 3 dof robot in terms of transformation matrices
    # 2 dof robot in terms of transofmraation matrices
    T = sp.eye(4)
    if name == 'dof2': #planar 2 dof, L1 = 1, L2 = 1
        if dof != 2:
            raise ValueError("name='dof2' expects 2 vars")
        T = T * Rz(vars[0], sin_hat[0], cos_hat[0]) * Tx(1)
        T = T * Rz(vars[1], sin_hat[1], cos_hat[1]) * Tx(1)

    elif name == "dof3": # planar 3 dof, L1=L2=L3=1
        if dof != 3:
            raise ValueError("name='dof3' expects 3 vars")
        T = T * Rz(vars[0], sin_hat[0], cos_hat[0]) * Tx(1)
        T = T * Rz(vars[1], sin_hat[1], cos_hat[1]) * Tx(1)
        T = T * Rz(vars[2], sin_hat[2], cos_hat[2]) * Tx(1)

    else:
        raise ValueError("Only 'dof2' and 'dof3' are implemented in this simple demo")

    print(f"Rotation matrix for {name} with vars {vars}:\n{T}")
    # exit()
    return T

    
def newton_raphson(f, J_inv, x0, tol=1e-3, max_iter=100, bounds=None, chord_newton=False):
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
def plot_cartesian_trajectory(series, x_star, title="End-effector trajectory (Cartesian)"):
    """
    series: list of (xs, label)
        xs shape: (N, 2) or (N, 3); plots x-y projection
    """
    plt.figure()
    for xs, label in series:
        plt.plot(xs[:, 0], xs[:, 1], marker="o", label=label)

    plt.scatter([x_star[0]], [x_star[1]], marker="x", s=90, label="Target")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_joint_trajectory(series, title="Joint-space trajectory"):
    """
    series: list of (q_hist, label_prefix)
        q_hist shape: (N, dof)
    """
    plt.figure()
    for q_hist, label_prefix in series:
        dof = q_hist.shape[1]
        it = np.arange(q_hist.shape[0])
        for j in range(dof):
            plt.plot(it, q_hist[:, j], label=f"{label_prefix} q{j}")

    plt.xlabel("iteration")
    plt.ylabel("joint angle (rad)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_error_history(series, title="Task error vs iteration"):
    """
    series: list of (e_hist, label)
    """
    plt.figure()
    for e_hist, label in series:
        plt.plot(e_hist, label=label)

    plt.xlabel("iteration")
    plt.ylabel("||x* - fkin(q)||")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_spectral_norm_history(series, title="Jacobian Residual Largest Entry"):
    """
    series: list of (sig_hist, label)
    """
    plt.figure()
    for sig_hist, label in series:
        plt.plot(sig_hist, label=label)

    plt.xlabel("iteration")
    plt.ylabel("Jacobian entry with largest residual")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()
    

def script(sin_list, cos_list, vars, robot_name, q0, q_star, damping=1e-3, chord_newton=False):
    # --- build symbolic FK (4x4) using your rotation-matrix chain
    true_sin = sp.sin
    true_cos = sp.cos

    T = get_robot_rotation_matrix(name=robot_name, sin_hat=sin_list, cos_hat=cos_list, vars=vars)
    T_true = get_robot_rotation_matrix(name=robot_name, sin_hat=[true_sin]*len(sin_list), cos_hat=[true_cos]*len(cos_list), vars=vars)
    # --- end-effector position = translation column
    p = sp.Matrix([T[0, 3], T[1, 3], T[2, 3]])         # 3x1 #note that the piecewise approximation is used for the jacobian, but the true function is used for the position
    p_true = sp.Matrix([T_true[0, 3], T_true[1, 3], T_true[2, 3]])   

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

        JJt = Jn @ Jn.T
        return Jn.T @ np.linalg.inv(JJt + damping * np.eye(JJt.shape[0]))

    # --- run Newton iterations in joint space
    q_sol, iters, q_hist, e_hist = newton_raphson(f, J_inv, q0, tol=1e-6, max_iter=60, chord_newton=chord_newton)

    # --- collect Cartesian trajectory and spectral norms
    x_hist = np.array([np.array(p_fun(*q), dtype=float).reshape(-1) for q in q_hist])

    sig_hist = []
    for q in q_hist:
        s = np.linalg.svd(np.array(J_fun(*q), dtype=float), compute_uv=False)
        sig_hist.append(float(s[0]))
    sig_hist = np.array(sig_hist)

    return {
        "T": T, "p": p, "J": J,
        "p_fun": p_fun, "J_fun": J_fun,
        "x_star": x_star,
        "q_sol": q_sol, "iters": iters,
        "q_hist": q_hist, "x_hist": x_hist,
        "e_hist": e_hist,
        "sig_hist": sig_hist,
    }


# --------------------------
# main (updated: run 4 cases + CLI flag for chord/newton)
# --------------------------
def eval_joint_space_main():
    dof = 3
    vars = sp.symbols(f"q0:{dof}", real=True)
    robot_name = f"dof{dof}"
    joint_ranges = [(pi/6, pi/2)] * dof

    q0     = np.array([pi/4, pi/4, pi/3][:dof], dtype=float)
    q_star = np.array([pi/5, pi/2, pi/4][:dof], dtype=float)

    # --- build piecewise trig lists once
    sin_list_hat = []
    cos_list_hat = []
    for _ in range(dof):
        sin_knots = np.linspace(0, pi, 10)
        cos_knots = np.linspace(0, pi, 10)
        sin_list_hat.append(create_piecewise_sinusoid(sp.sin, sin_knots))
        cos_list_hat.append(create_piecewise_sinusoid(sp.cos, cos_knots))

    results = evaluate_joint_space(joint_ranges, sin_list_hat, cos_list_hat, q0, vars, robot_name)
    plot_convergence_results(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chord_newton",
        action="store_true",
        help="If set, runs chord Newton only. Otherwise runs full Newton only."
    )
    parser.add_argument(
        "--run_both",
        action="store_true",
        help="Run BOTH chord and full Newton, for BOTH true and model trig (4-way comparison)."
    )
    parser.add_argument(
        "--true_only_compare",
        action="store_true",
        help="Run BOTH chord and full Newton, but using ONLY the true trig model."
    )
    args = parser.parse_args()

    dof = 3
    vars = sp.symbols(f"q0:{dof}", real=True)
    robot_name = f"dof{dof}"

    q0     = np.array([pi/4, pi/4, pi/3][:dof], dtype=float)
    q_star = np.array([pi/5, pi/2, pi/4][:dof], dtype=float)

    # piecewise model trig
    sin_list_hat = []
    cos_list_hat = []
    for _ in range(dof):
        sin_knots = np.linspace(0, pi, 5)
        cos_knots = np.linspace(0, pi, 5)
        sin_list_hat.append(create_piecewise_sinusoid(sp.sin, sin_knots))
        cos_list_hat.append(create_piecewise_sinusoid(sp.cos, cos_knots))

    # true trig
    sin_list_true = [sp.sin] * dof
    cos_list_true = [sp.cos] * dof

    # -------------------------------------------------
    # MODE 1: true-only comparison (Chord True vs Newton True)
    # -------------------------------------------------
    if args.true_only_compare:
        out_ct = script(
            sin_list=sin_list_true,
            cos_list=cos_list_true,
            vars=vars,
            robot_name=robot_name,
            q0=q0,
            q_star=q_star,
            chord_newton=True
        )
        out_nt = script(
            sin_list=sin_list_true,
            cos_list=cos_list_true,
            vars=vars,
            robot_name=robot_name,
            q0=q0,
            q_star=q_star,
            chord_newton=False
        )

        x_star = out_ct["x_star"]

        e_ct = np.linalg.norm(out_ct["x_hist"] - x_star.reshape(1, -1), axis=1)
        e_nt = np.linalg.norm(out_nt["x_hist"] - x_star.reshape(1, -1), axis=1)

        # duplicate true trajectories into the "model" slots so your existing plotters still work
        plot_cartesian_trajectory(
            [
                (out_ct["x_hist"], "Chord (true trig)"),
                (out_nt["x_hist"], "Newton (true trig)"),
            ],
            x_star,
            title="End-effector trajectory: true trig only"
        )

        plot_joint_trajectory(
            [
                (out_ct["q_hist"], "Chord (true trig)"),
                (out_nt["q_hist"], "Newton (true trig)"),
            ],
            title="Joint-space trajectory: true trig only"
        )

        plot_error_history(
            [
                (e_ct, "Chord (true trig)"),
                (e_nt, "Newton (true trig)"),
            ],
            title="Task error vs iteration: true trig only"
        )

        plot_spectral_norm_history(
            [
                (out_ct["sig_hist"], "Chord (true trig)"),
                (out_nt["sig_hist"], "Newton (true trig)"),
            ],
            title="Jacobian spectral norm: true trig only"
        )

        print("\n--- Summary (true-only comparison) ---")
        print(f"Target x*: {x_star}")
        print(f"Chord  True: iters={out_ct['iters']}, final ||e||={e_ct[-1]}")
        print(f"Newton True: iters={out_nt['iters']}, final ||e||={e_nt[-1]}")
        return

    # -------------------------------------------------
    # MODE 2: original full 4-way comparison
    # -------------------------------------------------
    modes = []
    if args.run_both:
        modes = [("chord", True), ("newton", False)]
    else:
        modes = [("chord" if args.chord_newton else "newton", args.chord_newton)]

    outs = {}

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

    # single solver, true vs model
    if not args.run_both:
        mode_name, _ = modes[0]
        out_t = outs[(mode_name, "true")]
        out_m = outs[(mode_name, "model")]
        x_star = out_t["x_star"]

        e_t = np.linalg.norm(out_t["x_hist"] - x_star.reshape(1, -1), axis=1)
        e_m = np.linalg.norm(out_m["x_hist"] - x_star.reshape(1, -1), axis=1)

        
        plot_cartesian_trajectory(
        [
            (out_t["x_hist"], f"{mode_name} (true trig)"),
            (out_m["x_hist"], f"{mode_name} (piecewise trig)"),
        ],
        x_star,
        title="End-effector trajectory"
        )

        plot_joint_trajectory(
            [
                (out_t["q_hist"], f"{mode_name} (true trig)"),
                (out_m["q_hist"], f"{mode_name} (piecewise trig)"),
            ],
            title="Joint-space trajectory"
        )

        plot_error_history(
            [
                (e_t, f"{mode_name} (true trig)"),
                (e_m, f"{mode_name} (piecewise trig)"),
            ],
            title="Task error vs iteration"
        )

        plot_spectral_norm_history(
            [
                (out_t["sig_hist"], f"{mode_name} (true trig)"),
                (out_m["sig_hist"], f"{mode_name} (piecewise trig)"),

            ],
            title="Largest Jacobian Error"
        )

        print(f"\n--- Summary ({mode_name}: true vs model) ---")
        print(f"Target x*: {x_star}")
        print(f"{mode_name.title()} True:  iters={out_t['iters']}, final ||e||={e_t[-1]}")
        print(f"{mode_name.title()} Model: iters={out_m['iters']}, final ||e||={e_m[-1]}")
        return

    # 4-way comparison
    out_ct = outs[("chord", "true")]
    out_cm = outs[("chord", "model")]
    out_nt = outs[("newton", "true")]
    out_nm = outs[("newton", "model")]

    x_star = out_ct["x_star"]

    e_ct = np.linalg.norm(out_ct["x_hist"] - x_star.reshape(1, -1), axis=1)
    e_cm = np.linalg.norm(out_cm["x_hist"] - x_star.reshape(1, -1), axis=1)
    e_nt = np.linalg.norm(out_nt["x_hist"] - x_star.reshape(1, -1), axis=1)
    e_nm = np.linalg.norm(out_nm["x_hist"] - x_star.reshape(1, -1), axis=1)

    plot_cartesian_trajectory(
    [
        (out_ct["x_hist"], "Chord (true trig)"),
        (out_cm["x_hist"], "Chord (piecewise trig)"),
        (out_nt["x_hist"], "Newton (true trig)"),
        (out_nm["x_hist"], "Newton (piecewise trig)"),
    ],
    x_star,
    title="End-effector trajectory"
    )

    plot_joint_trajectory(
        [
            (out_ct["q_hist"], "Chord (true trig)"),
            (out_cm["q_hist"], "Chord (piecewise trig)"),
            (out_nt["q_hist"], "Newton (true trig)"),
            (out_nm["q_hist"], "Newton (piecewise trig)"),
        ],
        title="Joint-space trajectory"
    )

    plot_error_history(
        [
            (e_ct, "Chord (true trig)"),
            (e_cm, "Chord (piecewise trig)"),
            (e_nt, "Newton (true trig)"),
            (e_nm, "Newton (piecewise trig)"),
        ],
        title="Task error vs iteration"
    )

    plot_spectral_norm_history(
        [
            (out_ct["sig_hist"], "Chord (true trig)"),
            (out_cm["sig_hist"], "Chord (piecewise trig)"),
            (out_nt["sig_hist"], "Newton (true trig)"),
            (out_nm["sig_hist"], "Newton (piecewise trig)"),
        ],
        title="Jacobian spectral norm along trajectory"
    )

    print("\n--- Summary (4-way) ---")
    print(f"Target x*: {x_star}")
    print(f"Chord  True:  iters={out_ct['iters']}, final ||e||={e_ct[-1]}")
    print(f"Chord  Model: iters={out_cm['iters']}, final ||e||={e_cm[-1]}")
    print(f"Newton True:  iters={out_nt['iters']}, final ||e||={e_nt[-1]}")
    print(f"Newton Model: iters={out_nm['iters']}, final ||e||={e_nm[-1]}")

def evaluate_joint_space(joint_ranges, sin_list, cos_list, q0, vars, robot_name, damping=1e-3, tol=1e-3):
    true_sin = sp.sin
    true_cos = sp.cos

    T_true = get_robot_rotation_matrix(name=robot_name, sin_hat=[true_sin]*len(sin_list), cos_hat=[true_cos]*len(cos_list), vars=vars)
    T = get_robot_rotation_matrix(name=robot_name, sin_hat=sin_list, cos_hat=cos_list, vars=vars)
    # --- end-effector position = translation column
    p = sp.Matrix([T[0, 3], T[1, 3], T[2, 3]])         # 3x1
    p_true = sp.Matrix([T_true[0, 3], T_true[1, 3], T_true[2, 3]])   
    J = p.jacobian(vars)                               # 3 x dof
    print(J)
    #evaluate J at the initial configuration q0 to see how the piecewise approximation affects the Jacobian at the start of the trajectory
    q0 = [float(v) for v in q0]

    J_initial = np.array(J.subs({var: val for var, val in zip(vars, q0)}), dtype=float)
    print(f"Initial Jacobian at q0={q0}:\n{J_initial}")
    #evaluate J at a cusp point of the piecewise function to see how the Jacobian behaves at a non-smooth point
    cusp_q = [pi/6, pi/6, pi/6][:len(vars)]  # example cusp point where the piecewise function changes segments
    J_cusp = np.array(J.subs({var: val for var, val in zip(vars, cusp_q)}), dtype=float)
    print(f"Jacobian at cusp point q={cusp_q}:\n{J_cusp}")
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
    print("GRIDS:", grids)
    
    joints = np.array(np.meshgrid(*grids)).T.reshape(-1, len(joint_ranges))
    print("joints", joints)

    results = []
    for q_star in joints:
         # --- define the target in Cartesian space from TRUE FK at q_star (passed in)
        x_star = np.array(p_fun(*q_star), dtype=float).reshape(-1)
        x_hat_star = np.array(p_hat_fun(*q_star), dtype=float).reshape(-1)
        q_sol, iters, q_hist, e_hist = newton_raphson(f, J_inv, q0, tol=tol, max_iter=60)
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

def plot_convergence_results(results):
    """Plot results from evaluate_joint_space."""
    dim = len(results[0]["q_star"])
    x_hat_stars = np.array([r["x_hat_star"] for r in results])
    x_stars = np.array([r["x_star"] for r in results])
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

        #then plot the forward kinematics shape: plot x_star in the world coordinate space x and y
        plt.figure()
        plt.scatter(x_stars[:, 0], x_stars[:, 1], c="black", label="x_star true (TRUE)")
        plt.scatter(x_hat_stars[:, 0], x_hat_stars[:, 1], c="orange", label="x_hat_star (MODEL)")
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        plt.title("Forward Kinematics Shape in World Space")
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

        #then plot the forward kinematics shape: plot x_star in the world coordinate space x y z
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x_stars[:, 0], x_stars[:, 1], x_stars[:, 2], c="black", label="x_star true")
        ax.scatter(x_hat_stars[:, 0], x_hat_stars[:, 1], x_hat_stars[:, 2], c="orange", label="x_star")
        ax.set_xlabel("x coordinate")
        ax.set_ylabel("y coordinate")
        ax.set_zlabel("z coordinate")
        ax.set_title("Forward Kinematics Shape in World Space")
        ax.grid(True)
        plt.show()  



# main()
eval_joint_space_main()

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



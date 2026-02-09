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
pi=np.pi

def create_piecewise_sinusoid(sympy_function, knots):
    """
    Returns a callable pw(arg) that produces a SymPy Piecewise linear interpolation
    of sympy_function(arg) over the knot interval [knots[0], knots[-1]].

    note: the caller is responsible for passing arugments in acceptable domain
    """
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

    return T

    
def newton_raphson(f, J_inv, x0, tol=1e-6, max_iter=100):
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

    for n in range(max_iter):
        e = np.array(f(q), dtype=float).reshape(-1)
        e_norm = float(np.linalg.norm(e))
        e_hist.append(e_norm)

        if e_norm < tol:
            return q, n, np.array(q_hist), np.array(e_hist)

        Jpinv = np.array(J_inv(q), dtype=float)
        dq = Jpinv @ e
        q = q + dq

        q_hist.append(q.copy())

    print(f"Warning: Maximum iterations ({max_iter}) exceeded. Not converged to tol={tol}.")
    return q, max_iter, np.array(q_hist), np.array(e_hist)


# --------------------------
# plotting (simple)
# --------------------------
def plot_cartesian_trajectory(xs_true, xs_hat, x_star):
    plt.figure()
    plt.plot(xs_true[:, 0], xs_true[:, 1], marker="o", label="True trig")
    plt.plot(xs_hat[:, 0],  xs_hat[:, 1],  marker="o", label="Piecewise trig")
    plt.scatter([x_star[0]], [x_star[1]], marker="x", s=90, label="Target")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("End-effector trajectory (Cartesian)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_joint_trajectory(qs_true, qs_hat):
    plt.figure()
    it_true = np.arange(qs_true.shape[0])
    it_hat  = np.arange(qs_hat.shape[0])
    dof = qs_true.shape[1]

    for j in range(dof):
        plt.plot(it_true, qs_true[:, j], label=f"q{j} true")
        plt.plot(it_hat,  qs_hat[:, j],  linestyle="--", label=f"q{j} piecewise")

    plt.xlabel("iteration")
    plt.ylabel("joint angle (rad)")
    plt.title("Joint-space trajectory")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_error_history(e_true, e_hat):
    plt.figure()
    plt.plot(e_true, label="||e|| true")
    plt.plot(e_hat,  label="||e|| piecewise")
    plt.xlabel("iteration")
    plt.ylabel("||x* - fkin(q)||")
    plt.title("Task error vs iteration")
    plt.grid(True)
    plt.legend()
    plt.show()



def plot_spectral_norm_history(sig_true, sig_hat):
    plt.figure()
    plt.plot(sig_true, label="||J||2 true")
    plt.plot(sig_hat,  label="||J||2 piecewise")
    plt.xlabel("iteration")
    plt.ylabel("spectral norm")
    plt.title("Jacobian spectral norm along trajectory")
    plt.grid(True)
    plt.legend()
    plt.show()

def script(sin_list, cos_list, vars, robot_name, q0, q_star, damping=1e-3):
    # --- build symbolic FK (4x4) using your rotation-matrix chain
    true_sin = sp.sin
    true_cos = sp.cos

    T = get_robot_rotation_matrix(name=robot_name, sin_hat=sin_list, cos_hat=cos_list, vars=vars)
    T_true = get_robot_rotation_matrix(name=robot_name, sin_hat=[true_sin]*len(sin_list), cos_hat=[true_cos]*len(cos_list), vars=vars)
    # --- end-effector position = translation column
    p = sp.Matrix([T[0, 3], T[1, 3], T[2, 3]])         # 3x1
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
    x_star = np.array(p_fun_wrapped(*q_star), dtype=float).reshape(-1)

    # --- visual servoing / IK error function: e(q) = x* - fkin(q)
    def f(q):
        x = np.array(p_fun_wrapped(*q), dtype=float).reshape(-1)
        return x_star - x

    # --- damped pseudo-inverse Jacobian (stable)
    def J_inv(q):
        Jn = np.array(J_fun_wrapped(*q), dtype=float)
        JJt = Jn @ Jn.T
        return Jn.T @ np.linalg.inv(JJt + damping * np.eye(JJt.shape[0]))

    # --- run Newton iterations in joint space
    q_sol, iters, q_hist, e_hist = newton_raphson(f, J_inv, q0, tol=1e-6, max_iter=60)

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


def main():
    # initialize one of our robots via cmd line arg
    # (fallback to 3 if cmd_line_arg not defined)
    dof= 3
    vars = sp.symbols(f"q0:{dof}", real=True)
    robot_name = f"dof{dof}"

    # fixed test setup so both models are compared fairly
    q0     = np.array([0.3, np.pi/4, 0.1][:dof], dtype=float)
    q_star = np.array([0.1, np.pi/2, 0.4][:dof], dtype=float)

    # ---------------- True trig ----------------
    print("True Sin and Cos:")
    sin_list_true = [sp.sin] * dof
    cos_list_true = [sp.cos] * dof
    out_true = script(
        sin_list=sin_list_true,
        cos_list=cos_list_true,
        vars=vars,
        robot_name=robot_name,
        q0=q0,
        q_star=q_star
    )

    # ---------------- Piecewise trig ----------------
    print("Approximated Sin and Cos:")
    knots = np.linspace(-2*pi,2*pi,9)  # 4 segments
    

    sin_list_hat = []
    cos_list_hat = []
    for _ in range(dof):
        sin_list_hat.append(create_piecewise_sinusoid(sp.sin, knots))
        cos_list_hat.append(create_piecewise_sinusoid(sp.cos, knots))

    x = sp.symbols('x')
    expr = sin_list_hat[0](x)
    f = sp.lambdify(x, expr, "numpy")
    xs = np.linspace(-2*np.pi, 2*np.pi, 1000)
    ys = f(xs)
    plt.plot(xs, ys)
    plt.ylim(-2, 2)
    plt.show(block=True)


    x = sp.symbols('x')
    expr = cos_list_hat[0](x)
    f = sp.lambdify(x, expr, "numpy")
    xs = np.linspace(-2*np.pi, 2*np.pi, 1000)
    ys = f(xs)
    plt.plot(xs, ys)
    plt.ylim(-2, 2)
    plt.show(block=True)



    out_hat = script(
        sin_list=sin_list_hat,
        cos_list=cos_list_hat,
        vars=vars,
        robot_name=robot_name,
        q0=q0,
        q_star=q_star
    )

    # ---------------- Compare & plot ----------------
    # Use the SAME target (true x*) for both plots (fair comparison)
    x_star = out_true["x_star"]

    # Recompute error vs the same target for both trajectories
    e_true = np.linalg.norm(out_true["x_hist"] - x_star.reshape(1, -1), axis=1)
    e_hat  = np.linalg.norm(out_hat["x_hist"]  - x_star.reshape(1, -1), axis=1)

    plot_cartesian_trajectory(out_true["x_hist"], out_hat["x_hist"], x_star)
    plot_joint_trajectory(out_true["q_hist"], out_hat["q_hist"])
    plot_error_history(e_true, e_hat)
    plot_spectral_norm_history(out_true["sig_hist"], out_hat["sig_hist"])

    print("\n--- Summary ---")
    print(f"Target x*: {x_star}")
    print(f"True:     iters={out_true['iters']}, final ||e||={e_true[-1]}")
    print(f"Piecewise iters={out_hat['iters']}, final ||e||={e_hat[-1]}")

main()

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



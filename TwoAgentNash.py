import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
#from cyipopt import minimize_ipopt
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time



#reference line info
def exp_function(x):
    return -np.exp(-0.4*x) + 1

def line(x):
    return 1.0*np.ones(x.shape[0])

def cur_accum_s(x, y):
    s = 0
    accum_s = []
    accum_s.append(s)
    for i in range(len(x)):
        if i >= 1:
            s += np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
            accum_s.append(s)
    return np.array(accum_s)

def poly_derivative(poly_params):
    poly_order = poly_params.shape[0]
    max_order = poly_order - 1
    if max_order == 0:
        return 0.0
    grad_vec = np.zeros(max_order)
    for i in range(max_order):
        grad_vec[i] = (max_order - i) * poly_params[i]
    return grad_vec


#optimization setup
def collision_avoidance_objective(s1, s2, fitted_lane_funcs):
    x1 = fitted_lane_funcs["s2x_exp"](s1)
    y1 = fitted_lane_funcs["s2y_exp"](s1)
    x2 = fitted_lane_funcs["s2x_line"](s2)
    y2 = fitted_lane_funcs["s2y_line"](s2)
    dist = (x1 - x2)**2 + (y1 - y2)**2
    collision_avoidance_cost = np.exp(-dist + 15)
    return collision_avoidance_cost
    

def track_speed_objective(s1_dot, s2_dot, track_vel_param):
    return (track_vel_param["v1_weight"]*(s1_dot - track_vel_param["v1_ref"])**2 + 
            track_vel_param["v2_weight"]*(s2_dot - track_vel_param["v2_ref"])**2)


def comfort_objective(s1_ddot, s2_ddot):
    number_points = s1_ddot.shape[0]
    num_jerk1 = s1_ddot[0:number_points-2] - s1_ddot[1:number_points-1]
    num_jerk2 = s2_ddot[0:number_points-2] - s2_ddot[1:number_points-1]
    return np.linalg.norm(num_jerk1) + np.linalg.norm(num_jerk2)
   

def gradient(s, fitted_lane_funcs, fitted_lane_prime, track_vel_param, objective_weight):
    var_len = s.shape[0]
    single_agent_var_len = int(var_len / 2)
    single_order_var_len = int(single_agent_var_len / 3)
    s1_var = s[0:single_agent_var_len]
    s2_var = s[single_agent_var_len:var_len]
    s1 = s1_var[0:single_order_var_len]
    s2 = s2_var[0:single_order_var_len]
    s1_dot = s1_var[single_order_var_len:single_order_var_len*2]
    s2_dot = s2_var[single_order_var_len:single_order_var_len*2]
    s1_ddot = s1_var[single_order_var_len*2:single_order_var_len*3]
    s2_ddot = s2_var[single_order_var_len*2:single_order_var_len*3]
    x1 = fitted_lane_funcs["s2x_exp"](s1)
    y1 = fitted_lane_funcs["s2y_exp"](s1)
    x2 = fitted_lane_funcs["s2x_line"](s2)
    y2 = fitted_lane_funcs["s2y_line"](s2)
    x1_x2 = x1 - x2
    y1_y2 = y1 - y2
    x1_prime = fitted_lane_prime["s2x_exp_prime"](s1)
    y1_prime = fitted_lane_prime["s2y_exp_prime"](s1)
    x2_prime = fitted_lane_prime["s2x_line_prime"](s2)
    y2_prime = fitted_lane_prime["s2y_line_prime"](s2)
    dist = x1_x2**2 + y1_y2**2
    collision_avoidance_cost = np.exp(-dist + 15)
    dv_ds1 = objective_weight["collision_weight"] * collision_avoidance_cost * (-2 * x1_x2 * x1_prime - 2 * y1_y2 * y1_prime)
    dv_ds2 = objective_weight["collision_weight"] * collision_avoidance_cost * (2 * x1_x2 * x2_prime + 2 * y1_y2 * y2_prime)
    dv_dds1 = 2 * objective_weight["track_speed_weight"] * track_vel_param["v1_weight"] * (s1_dot - track_vel_param["v1_ref"])
    dv_dds2 = 2 * objective_weight["track_speed_weight"] * track_vel_param["v2_weight"] * (s2_dot - track_vel_param["v2_ref"])
    dv_ddds1 = np.zeros(single_order_var_len)
    dv_ddds2 = np.zeros(single_order_var_len)
    
    for i in range(single_order_var_len - 2):
        if i == 0:
            dv_ddds1[i] = objective_weight["comfort_weight"] * 2 * (s1_ddot[i] - s1_ddot[i+1])
            dv_ddds2[i] = objective_weight["comfort_weight"] * 2 * (s2_ddot[i] - s2_ddot[i+1])
        else:
            dv_ddds1[i] = objective_weight["comfort_weight"] * 4 * s1_ddot[i] - 2 * s1_ddot[i-1] - 2 * s1_ddot[i+1]
            dv_ddds2[i] = objective_weight["comfort_weight"] * 4 * s2_ddot[i] - 2 * s2_ddot[i-1] - 2 * s2_ddot[i+1]
    return np.hstack((dv_ds1,dv_dds1, dv_ddds1, dv_ds2, dv_dds2, dv_ddds2))


#---------------Check correctness of this hessian-------------------------#
def hessian(s, fitted_lane_funcs, fitted_lane_prime, track_vel_param, objective_weight):
    var_len = s.shape[0]
    single_agent_var_len = int(var_len / 2)
    single_order_var_len = int(single_agent_var_len / 3)
    s1_var = s[0:single_agent_var_len]
    s2_var = s[single_agent_var_len:var_len]
    s1 = s1_var[0:single_order_var_len]
    s2 = s2_var[0:single_order_var_len]
    s1_dot = s1_var[single_order_var_len:single_order_var_len*2]
    s2_dot = s2_var[single_order_var_len:single_order_var_len*2]
    x1 = fitted_lane_funcs["s2x_exp"](s1)
    y1 = fitted_lane_funcs["s2y_exp"](s1)
    x2 = fitted_lane_funcs["s2x_line"](s2)
    y2 = fitted_lane_funcs["s2y_line"](s2)
    x1_x2 = x1 - x2
    y1_y2 = y1 - y2
    x1_prime = fitted_lane_prime["s2x_exp_prime"](s1)
    y1_prime = fitted_lane_prime["s2y_exp_prime"](s1)
    x2_prime = fitted_lane_prime["s2x_line_prime"](s2)
    y2_prime = fitted_lane_prime["s2y_line_prime"](s2)

    x1_dprime = fitted_lane_prime["s2x_exp_dprime"](s1)
    y1_dprime = fitted_lane_prime["s2y_exp_dprime"](s1)
    x2_dprime = fitted_lane_prime["s2x_line_dprime"](s2)
    y2_dprime = fitted_lane_prime["s2y_line_dprime"](s2)

    dist = x1_x2**2 + y1_y2**2
    collision_avoidance_cost = np.exp(-dist + 15)
    #first order
    #dv_ds1 = objective_weight["collision_weight"] * collision_avoidance_cost * (-2 * x1_x2 * x1_prime - 2 * y1_y2 * y1_prime)
    #dv_ds2 = objective_weight["collision_weight"] * collision_avoidance_cost * (2 * x1_x2 * x2_prime + 2 * y1_y2 * y2_prime)
    #second order
    dv_2_ds1_2 = -2 * objective_weight["collision_weight"] * collision_avoidance_cost * ((-2 * x1_x2**2 * x1_prime**2 - 
                  2 * y1_y2**2 * y1_prime**2 - 2 * x1_x2 * y1_y2 * x1_prime * y1_prime) + x1_x2 * x1_dprime + y1_y2 * y1_dprime)
    dv_2_ds2_2 =  2 * objective_weight["collision_weight"] * collision_avoidance_cost * ((-2 * x1_x2**2 * x2_prime**2 - 
                  2 * y1_y2**2 * y2_prime**2 - 2 * x1_x2 * y1_y2 * x2_prime * y2_prime) + x1_x2 * x2_dprime + y1_y2 * y2_dprime)
    dv_2_ds1s2 = 2 * collision_avoidance_cost * (-2 * x1_x2 * x1_prime * x2_prime - 2 * y1_y2 * y1_prime * y2_prime)
    dv_dds1 = 2 * objective_weight["track_speed_weight"] * track_vel_param["v1_weight"]
    dv_dds2 = 2 * objective_weight["track_speed_weight"] * track_vel_param["v2_weight"]

    hessian_mat = np.zeros((var_len,var_len))
    #fill in six blocks
    for ii in range(single_order_var_len):
        hessian_mat[ii,ii] = dv_2_ds1_2[ii]
        hessian_mat[single_order_var_len + ii, single_order_var_len + ii] = dv_dds1
        hessian_mat[single_agent_var_len + ii, single_agent_var_len + ii] = dv_2_ds2_2[ii]
        hessian_mat[single_agent_var_len + single_order_var_len + ii, single_agent_var_len + single_order_var_len + ii] = dv_dds2
        hessian_mat[ii, single_agent_var_len + ii] = dv_2_ds1s2[ii]
        hessian_mat[single_agent_var_len + ii, ii] = dv_2_ds1s2[ii]
    return hessian_mat


def objective(s, fitted_lane_funcs, fitted_lane_prime, track_vel_param, objective_weight):
    #segment decision variables 
    var_len = s.shape[0]
    single_agent_var_len = int(var_len / 2)
    single_order_var_len = int(single_agent_var_len / 3)
    s1_var = s[0:single_agent_var_len]
    s2_var = s[single_agent_var_len:var_len]
    s1 = s1_var[0:single_order_var_len]
    s2 = s2_var[0:single_order_var_len]
    s1_dot = s1_var[single_order_var_len:single_order_var_len*2]
    s2_dot = s2_var[single_order_var_len:single_order_var_len*2]
    s1_ddot = s1_var[single_order_var_len*2:single_order_var_len*3]
    s2_ddot = s2_var[single_order_var_len*2:single_order_var_len*3]

    return (objective_weight["collision_weight"]*np.sum(collision_avoidance_objective(s1, s2, fitted_lane_funcs)) +
            objective_weight["track_speed_weight"]*np.sum(track_speed_objective(s1_dot, s2_dot, track_vel_param)) +
            objective_weight["comfort_weight"]*comfort_objective(s1_ddot, s2_ddot))

def construct_linear_constraints(s, delta_t):
    #segment decision variables
    var_len = s.shape[0]
    single_agent_var_len = int(var_len / 2)
    single_order_var_len = int(single_agent_var_len / 3)
    num_constraints = 4 * (single_order_var_len - 1)
    #num_constraints = 6 * (single_order_var_len - 1)
    A = np.zeros((num_constraints, var_len))
    ub = np.zeros(num_constraints)
    lb = np.zeros(num_constraints)

    #dynamic constraints
    # s(i+1) - s(i) - delta_t * s(i)'
    # - 1/3 * delta_t^2 * s(i)'' - 1/6 * delta_t^2 * s(i+1)''
    #player 1
    constraint_index = 0
    for i in range(single_order_var_len - 1):
        A[constraint_index][i] = -1.0
        A[constraint_index][i + 1] = 1.0
        A[constraint_index][i + single_order_var_len] = -delta_t
        A[constraint_index][i + 2 * single_order_var_len] = -delta_t**2 / 3.0
        A[constraint_index][i + 2 * single_order_var_len + 1] = -delta_t**2 / 6.0
        ub[constraint_index] = 0.0
        lb[constraint_index] = 0.0
        constraint_index  = constraint_index + 1
    #player 2
    for i in range(single_order_var_len - 1):
        A[constraint_index][i + single_agent_var_len] = -1.0
        A[constraint_index][i + 1 + single_agent_var_len] = 1.0
        A[constraint_index][i + single_order_var_len + single_agent_var_len] = -delta_t
        A[constraint_index][i + 2 * single_order_var_len + single_agent_var_len] = -delta_t**2 / 3
        A[constraint_index][i + 2 * single_order_var_len + 1 + single_agent_var_len] = -delta_t**2 / 6
        ub[constraint_index] = 0.0
        lb[constraint_index] = 0.0
        constraint_index = constraint_index + 1

    # s(i+1)' - s(i)' - 0.5 * delta_t * s(i)'' - 0.5 * delta_t * s(i+1)'' = 0
    # player 1
    for i in range(single_order_var_len - 1):
        A[constraint_index][i + single_order_var_len] = -1.0
        A[constraint_index][i + 1 + single_order_var_len] = 1.0
        A[constraint_index][i + 2 * single_order_var_len] = -delta_t / 2.0
        A[constraint_index][i + 2 * single_order_var_len + 1] = -delta_t / 2.0
        ub[constraint_index] = 0.0
        lb[constraint_index] = 0.0
        constraint_index  = constraint_index + 1
    # player 2
    for i in range(single_order_var_len - 1):
        A[constraint_index][i + single_order_var_len + single_agent_var_len] = -1.0
        A[constraint_index][i + 1 + single_order_var_len + single_agent_var_len] = 1.0
        A[constraint_index][i + 2 * single_order_var_len + single_agent_var_len] = -delta_t / 2.0
        A[constraint_index][i + 2 * single_order_var_len + 1 + single_agent_var_len] = -delta_t / 2.0
        ub[constraint_index] = 0.0
        lb[constraint_index] = 0.0
        constraint_index  = constraint_index + 1

    """
    #(s(i+1)'‘ - s(i)'')/delta_t bounded
    #player 1
    for i in range(single_order_var_len - 1):
        A[constraint_index][i + 2 * single_order_var_len] = -1.0 / delta_t
        A[constraint_index][i + 1 + 2 * single_order_var_len] = 1.0 / delta_t
        ub[constraint_index] = 2
        lb[constraint_index] = -2
        constraint_index  = constraint_index + 1
    #player 2
    for i in range(single_order_var_len - 1):
        A[constraint_index][i + 2 * single_order_var_len + single_agent_var_len] = -1.0 / delta_t
        A[constraint_index][i + 1 + 2 * single_order_var_len + single_agent_var_len] = 1.0 / delta_t
        ub[constraint_index] = 4
        lb[constraint_index] = -4
        constraint_index  = constraint_index + 1
    
    """
    return A, ub, lb

"""
def constraints_ipopt(s, A):
    return A.dot(s)

def constraints_ipopt_jac(s, A):
    return A
def constraints_ipopt_hess(s, A):
    var_len = s.shape[0]
    return np.zeros((var_len,var_len))
"""

def construct_bounds(s, track_vel_param, s1_max, s2_max):
    var_len = s.shape[0]
    ub = np.zeros(var_len)
    lb = np.zeros(var_len)
    single_agent_var_len = int(var_len / 2)
    single_order_var_len = int(single_agent_var_len / 3)
    #s,s',s''bounds
    for i in range(single_order_var_len):
        #s1,s2 bounds
        lb[i] = 0.0
        ub[i] = s1_max
        lb[i + single_agent_var_len] = 0.0
        ub[i + single_agent_var_len] = s2_max
        #s1',s2' bounds
        lb[i + single_order_var_len] = 0.0
        ub[i + single_order_var_len] = track_vel_param["v1_ref"]
        lb[i + single_order_var_len + single_agent_var_len] = 0.0
        ub[i + single_order_var_len + single_agent_var_len] = track_vel_param["v2_ref"]
        #s1'',s2'' bounds
        lb[i + 2 * single_order_var_len] = -track_vel_param["max_acc"]
        ub[i + 2 * single_order_var_len] = track_vel_param["max_acc"]
        lb[i + 2 * single_order_var_len + single_agent_var_len] = -track_vel_param["max_acc"]
        ub[i + 2 * single_order_var_len + single_agent_var_len] = track_vel_param["max_acc"]
    #initial condition
    #s
    lb[0] = 0.0
    ub[0] = 0.0
    lb[single_agent_var_len] = 0.0
    ub[single_agent_var_len] = 0.0
    #s'
    lb[single_order_var_len] = 0.0
    ub[single_order_var_len] = 0.0
    lb[single_agent_var_len + single_order_var_len] = 0.0
    ub[single_agent_var_len + single_order_var_len] = 0.0

    return ub, lb

def construct_init_guess(t_max, delta_t, track_vel_param, s1_max, s2_max):
    #initial guess matters which mode you converge to
    #if you init s1_max, s2_max / 2, you implicitly say that say that s2 should 
    #yield to s1
    single_order_var_len = int(t_max / delta_t)
    single_agent_var_len = 3 * single_order_var_len
    var_len = single_agent_var_len * 2
    s0 = np.zeros(var_len)
    for i in range(single_order_var_len):
        #player 1
        s0[i] = s1_max
        s0[i + single_order_var_len] = track_vel_param["v1_ref"]
        s0[i + 2 * single_order_var_len] = track_vel_param["max_acc"]
        #player 2
        s0[i + single_agent_var_len] = s2_max / 2
        s0[i + single_order_var_len + single_agent_var_len] = track_vel_param["v2_ref"]
        s0[i + 2 * single_order_var_len + single_agent_var_len] = track_vel_param["max_acc"]
     #s
    s0[0] = 0.0
    s0[single_agent_var_len] = 0.0
    #s'
    s0[single_order_var_len] = 0.0
    s0[single_agent_var_len + single_order_var_len] = 0.0
    #s''
    s0[2 * single_order_var_len] = 0.0
    s0[2 * single_order_var_len + single_agent_var_len] = 0.0

    return s0

def callback(xk, fitted_lane_funcs, track_vel_param, objective_weight):
    print(f"Current solution: x = {xk}")
    print(f"Current objective value: f(x) = {objective(xk, fitted_lane_funcs, track_vel_param, objective_weight)}")
    print("-" * 80)

#animation display
def init():
    car1.set_data([], [])
    car2.set_data([], [])
    return car1, car2

def update(t, s1, s2, s2x_exp, s2y_exp, s2x_line, s2y_line, delta_t, t_max):
    t_total = np.arange(0, t_max, delta_t)
    t_index = np.where(t_total == t)
    s1_t = s1[t_index]
    s2_t = s2[t_index]

    x1 = s2x_exp(s1_t)
    y1 = s2y_exp(s1_t)

    x2 = s2x_line(s2_t)
    y2 = s2y_line(s2_t)

    car1.set_data(x1, y1)
    car2.set_data(x2, y2)
    return car1, car2

if __name__=="__main__":

    #Factors: s1_max, s2_max, delta_t, v1_weight, v2_weight
    #how do you compenstate the non-convexity of collision function?

    # Generate x values for plotting
    x_exp = np.linspace(-5, 18, 400)
    y_exp = exp_function(x_exp)
    x_line = np.linspace(-3, 18, 400)
    y_line = line(x_line)


    exp_accum_s = cur_accum_s(x_exp, y_exp)
    line_accum_s = cur_accum_s(x_line, y_line)

    exp_s2x_param = np.polyfit(exp_accum_s, x_exp, 5)
    exp_s2y_param = np.polyfit(exp_accum_s, y_exp, 5)

    line_s2x_param = np.polyfit(line_accum_s, x_line, 1)
    line_s2y_param = np.polyfit(line_accum_s, y_line, 1)


    """
    decision variables:
    s1,s1',s1'' 0:T
    s2,s2',s2'' 0:T
    """
    s2x_exp =  np.poly1d(exp_s2x_param)
    #eval prime s2x_exp
    s2x_exp_prime = np.poly1d(poly_derivative(exp_s2x_param))
    s2x_exp_dprime = np.poly1d(poly_derivative(poly_derivative(exp_s2x_param)))

    s2y_exp =  np.poly1d(exp_s2y_param)
    s2y_exp_prime = np.poly1d(poly_derivative(exp_s2y_param))
    s2y_exp_dprime = np.poly1d(poly_derivative(poly_derivative(exp_s2y_param)))

    s2x_line =  np.poly1d(line_s2x_param)
    s2x_line_prime = np.poly1d(poly_derivative(line_s2x_param))
    s2x_line_dprime = np.poly1d(poly_derivative(poly_derivative(line_s2x_param)))

    s2y_line =  np.poly1d(line_s2y_param)
    s2y_line_prime = np.poly1d(poly_derivative(line_s2y_param))
    s2y_line_dprime = np.poly1d(poly_derivative(poly_derivative(line_s2y_param)))

    fitted_lane_funcs = {
        "s2x_exp": s2x_exp,
        "s2y_exp": s2y_exp,
        "s2x_line": s2x_line,
        "s2y_line": s2y_line,
    }

    fitted_lane_prime = {
        "s2x_exp_prime": s2x_exp_prime,
        "s2y_exp_prime": s2y_exp_prime,
        "s2x_line_prime": s2x_line_prime,
        "s2y_line_prime": s2y_line_prime,
        "s2x_exp_dprime": s2x_exp_dprime,
        "s2y_exp_dprime": s2y_exp_dprime,
        "s2x_line_dprime": s2x_line_dprime,
        "s2y_line_dprime": s2y_line_dprime,
    }

    v1_ref = 2.0
    v2_ref = 2.0

    track_vel_param = {
        "v1_ref": v1_ref,
        "v1_weight": 1.0,
        "v2_ref": v2_ref,
        "v2_weight": 1.0,
        "max_acc": 1.0
    }

    objective_weight = {
        "collision_weight":1.0,
        "track_speed_weight":1.0,
        "comfort_weight":10.0
    }

    #First estimate a long enough time
    t_max = 20 #sec
    delta_t = 1.0 #sec

    s0 = construct_init_guess(t_max, delta_t, track_vel_param, exp_accum_s[-1], line_accum_s[-1])
    A, ug, lg = construct_linear_constraints(s0, delta_t)
    linear_constraints = LinearConstraint(A, lg, ug)
    ub, lb = construct_bounds(s0, track_vel_param, exp_accum_s[-1], line_accum_s[-1])
    bounds = Bounds(lb, ub)

    callback_with_params = partial(callback, fitted_lane_funcs=fitted_lane_funcs, track_vel_param=track_vel_param, objective_weight=objective_weight)

   
    """
    check shape of the collision avoidance objective
    collision_avoidance_objective_with_params = partial(collision_avoidance_objective, fitted_lane_funcs=fitted_lane_funcs)


    s1_arr = np.linspace(0, exp_accum_s[-1], 200)
    s2_arr = np.linspace(0, line_accum_s[-1], 200)

    S1, S2 = np.meshgrid(s1_arr, s2_arr)

    Z = collision_avoidance_objective_with_params(S1, S2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S1, S2, Z, cmap='viridis')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title('Surface plot of collision objective')

    plt.show()
    """
    start_time = time.time()   
    result = minimize(objective, s0, args=(fitted_lane_funcs, fitted_lane_prime, track_vel_param, objective_weight), 
                      method='SLSQP', 
                      jac=gradient,
                      constraints=linear_constraints, bounds=bounds, 
                      #callback=callback_with_params,
                      options={'disp': True,'maxiter': 2000,'ftol': 0.01}
                      )

    if result.success:
        print("Optimization was successful!")
        print("Optimal Solution:", result.x)
        print("Function Value at Optimal Solution:", result.fun)
    else:
        print("Optimization failed.")
        print("Reason:", result.message)
    end_time = time.time()
    print("Optimization time: ", end_time - start_time)

    
    """
    objective_params = partial(objective, fitted_lane_funcs = fitted_lane_funcs, fitted_lane_prime=fitted_lane_prime, track_vel_param=track_vel_param, objective_weight=objective_weight)
    jac_params = partial(gradient, fitted_lane_funcs = fitted_lane_funcs, fitted_lane_prime=fitted_lane_prime, track_vel_param=track_vel_param, objective_weight=objective_weight)
    hess_params = partial(hessian, fitted_lane_funcs = fitted_lane_funcs, fitted_lane_prime=fitted_lane_prime, track_vel_param=track_vel_param, objective_weight=objective_weight)

    constraints_ipopt_params = partial(constraints_ipopt, A = A)
    constraints_ipopt_jac_params = partial(constraints_ipopt_jac, A = A)
    constraints_ipopt_hess_params = partial(constraints_ipopt_hess, A = A)

    constr = {'type': 'eq', 'fun':  constraints_ipopt_params, 'jac': constraints_ipopt_jac_params}#, 'hess': constraints_ipopt_hess_params}

    start_time_ipopt = time.time()  
    result = minimize_ipopt(
                fun=objective_params,
                x0=s0,
                jac=jac_params,
                #hess=hess_params,
                bounds=bounds,
                constraints=constr,
                options={'disp': 0}
                )
    end_time_ipopt = time.time()
    print("Optimal solution:", result.x)
    print("Optimal value:", result.fun)
    print("IPOPT processing time: ", end_time_ipopt - start_time_ipopt)
    """
    


    #segment optimal solution
    var_len = result.x.shape[0]
    single_agent_var_len = int(var_len / 2)
    single_order_var_len = int(single_agent_var_len / 3)
    s1_var = result.x[0:single_agent_var_len]
    s2_var = result.x[single_agent_var_len:var_len]
    s1 = s1_var[0:single_order_var_len]
    s2 = s2_var[0:single_order_var_len]
    s1_dot = s1_var[single_order_var_len:single_order_var_len*2]
    s2_dot = s2_var[single_order_var_len:single_order_var_len*2]

    t = np.arange(0, t_max, delta_t)

    
    #plot animation
    fig, ax = plt.subplots()

    xdata1, ydata1 = [], []
    xdata2, ydata2 = [], []

    car1, = plt.plot([], [], 'ro', animated=True)
    car2, = plt.plot([], [], 'bo', animated=True)

    ax.set_xlim(-6, 20)
    ax.set_ylim(-8, 5)

    # Static background: road and buildings
    road1 = plt.plot(x_exp, y_exp, 'r-', lw=1)
    road2 = plt.plot(x_line, y_line, 'g-', lw=1)

    
    update_with_params = partial(update, s1 = s1, s2 = s2, s2x_exp = s2x_exp, s2y_exp = s2y_exp, 
                                 s2x_line = s2x_line, s2y_line = s2y_line, delta_t = delta_t, t_max = t_max)

    ani = animation.FuncAnimation(fig, update_with_params, frames=t,
                                  init_func=init, blit=True)

    plt.show()


    #display results graph
    plt.plot(t, s1, 'r-', label='Player 1 s')
    plt.plot(t, s2, 'g-', label='Player 2 s')

    plt.plot(t, s1_dot, 'r:', label='Player 1 s_dot')
    plt.plot(t, s2_dot, 'g:', label='Player 2 s_dot')

    plt.xlabel('t')
    plt.ylabel('s/s_dot')
    plt.title('Logitudinal info')
    plt.legend()
    plt.axis('equal')
    # Show the plot
    plt.grid(True)
    plt.show()
    




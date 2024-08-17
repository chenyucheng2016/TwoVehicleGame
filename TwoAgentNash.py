import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
from functools import partial

# Define the exp function
def exp_function(x):
    return -np.exp(-0.4*x) + 1

# Derivative of the exp function
def exp_derivative_function(x):
    return -0.4*np.exp(-0.4*x)  

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


def objective(s, fitted_lane_funcs, track_vel_param, objective_weight):
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
            objective_weight["track_speed_weight"]*np.sum(track_speed_objective(s1_dot, s2_dot, track_vel_param)))

def construct_linear_constraints(s, delta_t):
    #segment decision variables
    var_len = s.shape[0]
    single_agent_var_len = int(var_len / 2)
    single_order_var_len = int(single_agent_var_len / 3)
    num_constraints = 4 * (single_order_var_len - 1)
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

    return A, ub, lb


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
        s0[i + single_agent_var_len] = s2_max
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
    print("-" * 50)



# Generate x values for plotting
x_exp = np.linspace(-3, 12, 400)
y_exp = exp_function(x_exp)

x_line = np.linspace(-5, 12, 400)
y_line = line(x_line)


exp_accum_s = cur_accum_s(x_exp, y_exp)
line_accum_s = cur_accum_s(x_line, y_line)

exp_s2x_param = np.polyfit(exp_accum_s, x_exp, 3)
exp_s2y_param = np.polyfit(exp_accum_s, y_exp, 3)

line_s2x_param = np.polyfit(line_accum_s, x_line, 1)
line_s2y_param = np.polyfit(line_accum_s, y_line, 1)

"""
decision variables:
s1,s1',s1'' 0:T
s2,s2',s2'' 0:T
"""
s2x_exp =  np.poly1d(exp_s2x_param)
s2y_exp =  np.poly1d(exp_s2y_param)

s2x_line =  np.poly1d(line_s2x_param)
s2y_line =  np.poly1d(line_s2y_param)

fitted_lane_funcs = {
    "s2x_exp": s2x_exp,
    "s2y_exp": s2y_exp,
    "s2x_line": s2x_line,
    "s2y_line": s2y_line,
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

result = minimize(objective, s0, args=(fitted_lane_funcs, track_vel_param, objective_weight), 
                  method='SLSQP', constraints=linear_constraints, bounds=bounds, 
                  callback=callback_with_params,
                  options={
                            'disp': True,        # Display output during optimization
                            'maxiter': 2000,      # Maximum number of iterations
                            'ftol': 1e-1          # General tolerance for convergence
                        })

if result.success:
    print("Optimization was successful!")
    print("Optimal Solution:", result.x)
    print("Function Value at Optimal Solution:", result.fun)
else:
    print("Optimization failed.")
    print("Reason:", result.message)

#display results graph
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


plt.plot(t, s1, 'r-', label='Player 1 s')
plt.plot(t, s2, 'g-', label='Player 2 s')

plt.plot(t, s1_dot, 'r:', label='Player 1 s_dot')
plt.plot(t, s2_dot, 'g:', label='Player 2 s_dot')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Quadratic Curve with Horizontal Tangent Line')
plt.legend()
plt.axis('equal')
# Show the plot
plt.grid(True)
plt.show()

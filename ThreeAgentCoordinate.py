import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
#from cyipopt import minimize_ipopt
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import math
import random
import copy
from scipy.interpolate import interp1d

##########################################################################
#MCTS
##########################################################################
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 1e-6
        self.value = 1e-6
        self.average_value = -1
        self.ave_val_change_rate = 1.0

    def is_fully_expanded(self):
        actions = self.state.get_legal_actions()
        return len(self.children) == (len(actions["p1"]) * len(actions["p2"] * len(actions["p3"])))

    def state_equal(self,state1,state2):
        if  (abs(state1.s1 - state2.s1) < 1e-3 and 
            abs(state1.s2 - state2.s2) < 1e-3 and 
            abs(state1.s3 - state2.s3) < 1e-3 and
            abs(state1.s1_dot - state2.s1_dot) < 1e-3 and 
            abs(state1.s2_dot - state2.s2_dot) < 1e-3 and
            abs(state1.s3_dot - state2.s3_dot) < 1e-3 and
            abs(state1.s1_ddot - state2.s1_ddot) < 1e-3 and 
            abs(state1.s2_ddot - state2.s2_ddot) < 1e-3 and
            abs(state1.s3_ddot - state2.s3_ddot) < 1e-3):
            return True
        else:
            return False



    def best_child(self, c_param=1.4):
        # for c in self.children:
        #     print("exploration: ", math.sqrt((2 * math.log(self.visits) / c.visits)))
        c_param = 1.0
        choices_weights = [
            (-child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        legal_actions = self.state.get_legal_actions()
        actions4p1 = legal_actions["p1"]
        actions4p2 = legal_actions["p2"]
        actions4p3 = legal_actions["p3"]
        p = False
        for actionp1 in actions4p1:
            for actionp2 in actions4p2:
                for actionp3 in actions4p3:
                    actions = {"p1": actionp1,
                               "p2": actionp2,
                               "p3": actionp3}
                    # if not p:
                    #     print("actions",actions)
                    #     p = True
                    state_copy = copy.deepcopy(self.state)
                    new_state = state_copy.move(actions)
                    new_state_exist = False
                    for child in self.children:
                        if self.state_equal(child.state, new_state):
                            new_state_exist = True
                            break
                    if new_state_exist == False:
                        child_node = Node(new_state, parent=self)
                        self.children.append(child_node)
                        return child_node

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        new_ave_val = self.value / self.visits
        self.ave_val_change_rate = abs(new_ave_val - self.average_value) / abs(self.average_value)
        self.average_value = new_ave_val
        if self.parent:
            self.parent.backpropagate(reward)

    def best_action(self):
        return self.best_child(c_param=1.4)

class MCTS:
    def __init__(self, n_simulations=100):
        self.n_simulations = n_simulations

    def search(self, initial_state):
        root = Node(state=initial_state)
        ret = {"delta_t": root.state.delta_t,
               "s1": [root.state.s1],
               "s1_dot": [root.state.s1_dot],
               "s1_ddot": [root.state.s1_ddot],
               "s2": [root.state.s2],
               "s2_dot": [root.state.s2_dot],
               "s2_ddot": [root.state.s2_ddot],
               "s3": [root.state.s3],
               "s3_dot": [root.state.s3_dot],
               "s3_ddot": [root.state.s3_ddot],}
        
        while not root.state.is_terminal():
            step = 0
            while step < 1000:
                node = self._select(root)
                reward = self._simulate(copy.deepcopy(node.state))
                print("sim reward: ", reward)
                step = step + 1
                node.backpropagate(reward)
            

            # print("child len", len(root.children))
            # print("visits: ", root.visits)
            # for c in root.children:
            #     print("child visits: ", c.visits)
            # first_child = root.children[0]
            # print("first s1",first_child.state.s1)
            # print("first s2",first_child.state.s2)
            # print("first s3",first_child.state.s3)
            # print("first s1_dot",first_child.state.s1_dot)
            # print("first s2_dot",first_child.state.s2_dot)
            # print("first s3_dot",first_child.state.s3_dot)
            root = root.best_child()
            # print("converged root val", root.average_value)
            # print("\n")
            ret["s1"].append(root.state.s1)
            ret["s2"].append(root.state.s2)
            ret["s3"].append(root.state.s3)
            ret["s1_dot"].append(root.state.s1_dot)
            ret["s2_dot"].append(root.state.s2_dot)
            ret["s3_dot"].append(root.state.s3_dot)
            ret["s1_ddot"].append(root.state.s1_ddot)
            ret["s2_ddot"].append(root.state.s2_ddot)
            ret["s3_ddot"].append(root.state.s3_ddot)

        return ret

    def _select(self, node):
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                #print("in expand")
                return node.expand()
            else:
                #print("best child")
                node = node.best_child()
        return node

    def _simulate(self, state):#specfiy
        current_state = state


        actions  = {"p1": 0.0,
                    "p2": 0.0,
                    "p3": 0.0,
                    }

        if current_state.has_acclerated1 == False:
            waiting_steps_p1 =  np.random.randint(current_state.allowed_waiting_time_p1)
        else:
            waiting_steps_p1 = -1

        if current_state.has_acclerated2 == False:
            waiting_steps_p2 =  np.random.randint(current_state.allowed_waiting_time_p2)
        else:
            waiting_steps_p2 = -1

        if current_state.has_acclerated1 == False:
            waiting_steps_p3 =  np.random.randint(current_state.allowed_waiting_time_p3)
        else:
            waiting_steps_p3 = -1

        print("waiting_steps_p1: ", waiting_steps_p1)
        print("waiting_steps_p2: ", waiting_steps_p2)
        print("waiting_steps_p3: ", waiting_steps_p3)


        step = 0
        while not current_state.is_terminal():
            if waiting_steps_p1 >= 0:
                if step >= waiting_steps_p1:
                    actions["p1"] = 0.5
            else:
                actions["p1"] = 0.5

            if waiting_steps_p2 >= 0:
                if step >= waiting_steps_p2:
                    actions["p2"] = 0.5
            else:
                actions["p2"] = 0.5

            if waiting_steps_p3 >= 0:
                if step >= waiting_steps_p3:
                    actions["p3"] = 0.5
            else:
                actions["p3"] = 0.5
            
            current_state = current_state.move(actions)
            step += 1
        return current_state.get_accumulated_cost()


# Two Vehicle lon Game
class GameState:
    def __init__(self, lon_max, lon_info_init, cost, fitted_lane_funcs, delta_t = 1.0):
        self.s1_max = lon_max["p1"][0]#max s1
        self.s2_max = lon_max["p2"][0]#max s2
        self.s3_max = lon_max["p3"][0] #max_s3

        self.s1_dot_max = lon_max["p1"][1] #max s1_dot
        self.s2_dot_max = lon_max["p2"][1] #max s2_dot
        self.s3_dot_max = lon_max["p3"][1] #max s3_dot



        self.s1 = lon_info_init["p1"][0] #cur s1
        self.s1_dot = lon_info_init["p1"][1] #cur s1_dot
        self.s1_ddot = lon_info_init["p1"][2]
        self.has_acclerated1 =  lon_info_init["p1"][3] #has p1 accelerated

        self.s2 = lon_info_init["p2"][0] #cur s2
        self.s2_dot = lon_info_init["p2"][1] #cur s2_dot
        self.s2_ddot = lon_info_init["p2"][2]
        self.has_acclerated2 =  lon_info_init["p2"][3]  #has p2 accelerated


        self.s3 = lon_info_init["p3"][0] #cur s2
        self.s3_dot = lon_info_init["p3"][1] #cur s2_dot
        self.s3_ddot = lon_info_init["p3"][2]
        self.has_acclerated3 =  lon_info_init["p3"][3]  #has p2 accelerated


        self.delta_t = delta_t
        self.accumulated_cost = cost
        self.fitted_lane_funcs = fitted_lane_funcs
        self.lon_max = lon_max
        self.t_max = 2.0 * (np.max([self.s1_max - self.s1, self.s2_max - self.s2, self.s3_max - self.s3]) / np.max([self.s1_dot_max,self.s2_dot_max,self.s3_dot_max]))
        self.min_reach_time_p1 = (self.s1_max - self.s1) / self.s1_dot_max;
        self.allowed_waiting_time_p1 = int((self.t_max - self.min_reach_time_p1) / self.delta_t)

        self.min_reach_time_p2 = (self.s2_max - self.s2) / self.s2_dot_max;
        self.allowed_waiting_time_p2 = int((self.t_max - self.min_reach_time_p2) / self.delta_t)

        self.min_reach_time_p3 = (self.s3_max - self.s3) / self.s3_dot_max;
        self.allowed_waiting_time_p3 = int((self.t_max - self.min_reach_time_p3) / self.delta_t)


    def get_legal_actions(self):
        # Return actions of two players, for each player 
        # either apply 0 acc or a constant acc at each 
        # time instant
        # each action should be a dictionary
        actions = {"p1": [],
                   "p2": [],
                   "p3": [],}
        action_set1 = [0.0,0.5]
        action_set2 = [0.5]
        if self.has_acclerated1:
            actions["p1"] = action_set2
        else:
            actions["p1"] = action_set1

        if self.has_acclerated2:
            actions["p2"] = action_set2
        else:
            actions["p2"] = action_set1

        if self.has_acclerated3:
            actions["p3"] = action_set2
        else:
            actions["p3"] = action_set1
       
        return actions
    
    
    def move(self, action):
        # Return the new state after applying the action
        if self.has_acclerated1 == False and action["p1"] > 1e-3:
            self.has_acclerated1 = True
        if self.has_acclerated2 == False and action["p2"] > 1e-3:
            self.has_acclerated2 = True
        if self.has_acclerated3 == False and action["p3"] > 1e-3:
            self.has_acclerated3 = True

        new_s1_dot = self.s1_dot + self.delta_t *  0.5 * (self.s1_ddot + action["p1"])
        self.s1_ddot = action["p1"]
        if (new_s1_dot > self.s1_dot_max):
            new_s1_dot = self.s1_dot_max
        self.s1 = self.s1 + 0.5 * (new_s1_dot + self.s1_dot) * self.delta_t
        if (self.s1 > self.s1_max):
            self.s1 = self.s1_max
        self.s1_dot = new_s1_dot

        new_s2_dot = self.s2_dot + self.delta_t *  0.5 * (self.s2_ddot + action["p2"])
        self.s2_ddot = action["p2"]
        if (new_s2_dot > self.s2_dot_max):
            new_s2_dot = self.s2_dot_max
        self.s2 = self.s2 + 0.5 * (new_s2_dot + self.s2_dot) * self.delta_t
        if (self.s2 > self.s2_max):
            self.s2 = self.s2_max
        self.s2_dot = new_s2_dot

        new_s3_dot = self.s3_dot + self.delta_t *  0.5 * (self.s3_ddot + action["p3"])
        self.s3_ddot = action["p3"]
        if (new_s3_dot > self.s3_dot_max):
            new_s3_dot = self.s3_dot_max
        self.s3 = self.s3 + 0.5 * (new_s3_dot + self.s3_dot) * self.delta_t
        if (self.s3 > self.s3_max):
            self.s3 = self.s3_max
        self.s3_dot = new_s3_dot


        lon_info = {"p1": [self.s1, self.s1_dot, self.s1_ddot, self.has_acclerated1],
                    "p2": [self.s2, self.s2_dot, self.s2_ddot, self.has_acclerated2],
                    "p3": [self.s3, self.s3_dot, self.s3_ddot, self.has_acclerated3],
                    }

        ref_lines_12 = ["s2x_exp", "s2y_exp", "s2x_line", "s2y_line"]

        ref_lines_23 = ["s2x_line", "s2y_line", "s2x_exp2", "s2y_exp2"]

        ref_lines_13 = ["s2x_exp", "s2y_exp", "s2x_exp2", "s2y_exp2"]


        collision_cost_p1_p2 = self.collision_avoidance_objective_specified_players(self.s1, self.s2, ref_lines_12)
        collision_cost_p2_p3 = self.collision_avoidance_objective_specified_players(self.s2, self.s3, ref_lines_23)

 
        self.accumulated_cost = self.accumulated_cost + collision_cost_p1_p2 +  collision_cost_p2_p3
                                                 

        #print("collision p1 p2: ", collision_cost_p1_p2)
        #print("collision p2 p3: ", collision_cost_p2_p3)


         
        return GameState(self.lon_max, lon_info, self.accumulated_cost, self.fitted_lane_funcs)
    
    def is_terminal(self):
        # Return True if the state is terminal (end of game)
        #------------------TODO: Add more conditions for termination check------------------------#
        if self.s1 >= self.s1_max or (self.s2 >= self.s2_max and self.s3 >= self.s3_max):
            return True
        else:
            return False
    #need to import lateral information in here, the polynomial functions
    def get_accumulated_cost(self):
        return self.accumulated_cost# + self.progress_objective_bounded_specified_players()

    def collision_avoidance_objective_specified_players(self, s1, s2, ref_lines):
        x1 = self.fitted_lane_funcs[ref_lines[0]](s1)
        y1 = self.fitted_lane_funcs[ref_lines[1]](s1)
        x2 = self.fitted_lane_funcs[ref_lines[2]](s2)
        y2 = self.fitted_lane_funcs[ref_lines[3]](s2)
        dist = (x1 - x2)**2 + (y1 - y2)**2
        #print("dist: ", np.sqrt(dist))
        if np.sqrt(dist) >= 10.0:
            return 0.0
        elif np.sqrt(dist) < 1.0:
            return np.exp(15)*20
        else:
            return np.exp(-np.sqrt(dist)+15)
  

    def progress_objective_bounded_specified_players(self):
        return (self.s1 - self.lon_max["p1"][0])**2 + (self.s2 - self.lon_max["p2"][0])**2 + (self.s3 - self.lon_max["p3"][0])**2
        

###########################################
#Env setup
##########################################
#reference line info
def exp_function(x):
    return -np.exp(-0.4*x) + 1

def exp_function2(x):
    return -np.exp(-0.4*(x-7)) + 1

def qudratic_function(x):
    return -0.5*x**2 + 8

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

##########################################################################
#Numerical Optimization
##########################################################################
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

def progress_objective(s1, s2):
    return -(s1[-1]**2 + s2[-1]**2)


def proregss_objective_bounded(s1, s2, lon_max):
    return (s1 - lon_max["p1"][0])**2 + (s2 - lon_max["p2"][0])**2
   

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
    dv_ds1[-1] = dv_ds1[-1] - 2 * objective_weight["progress_weight"] * s1[-1]

    dv_ds2 = objective_weight["collision_weight"] * collision_avoidance_cost * (2 * x1_x2 * x2_prime + 2 * y1_y2 * y2_prime)
    dv_ds2[-1] = dv_ds2[-1] - 2 * objective_weight["progress_weight"] * s2[-1]

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
            objective_weight["comfort_weight"]*comfort_objective(s1_ddot, s2_ddot) +
            objective_weight["progress_weight"]*progress_objective(s1, s2))

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
        s0[i] = s1_max / 2
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


def construct_init_guess_via_search(t_max, delta_t, lon_max, lon_info_init, init_ret):
    coarse_delta_t = init_ret["delta_t"]
    coarse_t_max = len(init_ret["s1"]) * coarse_delta_t
    t = np.arange(0, coarse_t_max, coarse_delta_t)

    t2s1 = interp1d(t, init_ret["s1"], kind='linear', fill_value="extrapolate")
    t2s1_dot = interp1d(t, init_ret["s1_dot"], kind='linear', fill_value="extrapolate")
    t2s1_ddot = interp1d(t, init_ret["s1_ddot"], kind='linear', fill_value="extrapolate")

    t2s2 = interp1d(t, init_ret["s2"], kind='linear', fill_value="extrapolate")
    t2s2_dot = interp1d(t, init_ret["s2_dot"], kind='linear', fill_value="extrapolate")
    t2s2_ddot = interp1d(t, init_ret["s2_ddot"], kind='linear', fill_value="extrapolate")

    single_order_var_len = int(t_max / delta_t)
    single_agent_var_len = 3 * single_order_var_len
    var_len = single_agent_var_len * 2

    s0 = np.zeros(var_len)
    t_fill = np.linspace(0, t_max, single_order_var_len)

    for i in range(single_order_var_len):
        s0[i] = min(t2s1(t_fill[i]),lon_max["p1"][0])
        s0[i + single_order_var_len] = t2s1_dot(t_fill[i])

        s0[i + single_order_var_len * 2] =  0.5#track_vel_param["max_acc"]#t2s1_ddot(t_fill[i])

        s0[i + single_agent_var_len] = min(t2s2(t_fill[i]),lon_max["p2"][0])
        s0[i + single_order_var_len + single_agent_var_len] = t2s2_dot(t_fill[i])
        s0[i + 2*single_order_var_len + single_agent_var_len] =  0.5#track_vel_param["max_acc"]#t2s2_ddot(t_fill[i])

    return s0

def callback(xk, fitted_lane_funcs, track_vel_param, objective_weight):
    print(f"Current solution: x = {xk}")
    print(f"Current objective value: f(x) = {objective(xk, fitted_lane_funcs, track_vel_param, objective_weight)}")
    print("-" * 80)

#animation display
def init():
    car1.set_data([], [])
    car2.set_data([], [])
    car3.set_data([], [])
    return car1, car2, car3

def update(t, s1, s2, s3, s2x_exp, s2y_exp, s2x_line, s2y_line, s2x_exp2, s2y_exp2, delta_t, t_max):

    t_total = np.arange(0, t_max, delta_t)
    t_index = np.where(t_total == t)
    s1_t = s1[t_index]
    s2_t = s2[t_index]
    s3_t = s3[t_index]

    x1 = s2x_exp(s1_t)
    y1 = s2y_exp(s1_t)

    x2 = s2x_line(s2_t)
    y2 = s2y_line(s2_t)

    x3 = s2x_exp2(s3_t)
    y3 = s2y_exp2(s3_t)

    car1.set_data(x1, y1)
    car2.set_data(x2, y2)
    car3.set_data(x3, y3)
    return car1, car2, car3

if __name__=="__main__":
    #Factors: s1_max, s2_max, delta_t, v1_weight, v2_weight
    #how do you compenstate the non-convexity of collision function?

    # Generate x values for plotting
    x_exp = np.linspace(-4, 24, 400)
    y_exp = exp_function(x_exp)

    x_exp2 = np.linspace(18, 3, 400)
    y_exp2 = exp_function2(x_exp2) 


    x_line = np.linspace(-6, 24, 400)
    y_line = line(x_line)


    exp_accum_s = cur_accum_s(x_exp, y_exp)
    line_accum_s = cur_accum_s(x_line, y_line)
    exp_accum_s2 = cur_accum_s(x_exp2, y_exp2)

    exp_s2x_param = np.polyfit(exp_accum_s, x_exp, 6)
    exp_s2y_param = np.polyfit(exp_accum_s, y_exp, 6)

    line_s2x_param = np.polyfit(line_accum_s, x_line, 1)
    line_s2y_param = np.polyfit(line_accum_s, y_line, 1)

    exp2_s2x_param = np.polyfit(exp_accum_s2, x_exp2, 6)
    exp2_s2y_param = np.polyfit(exp_accum_s2, y_exp2, 6)


    
    # decision variables:
    # s1,s1',s1'' 0:T
    # s2,s2',s2'' 0:T
    
    s2x_exp =  np.poly1d(exp_s2x_param)
    s2x_exp_prime = np.poly1d(poly_derivative(exp_s2x_param))
    s2x_exp_dprime = np.poly1d(poly_derivative(poly_derivative(exp_s2x_param)))

    s2y_exp =  np.poly1d(exp_s2y_param)
    s2y_exp_prime = np.poly1d(poly_derivative(exp_s2y_param))
    s2y_exp_dprime = np.poly1d(poly_derivative(poly_derivative(exp_s2y_param)))#double prime

    s2x_line =  np.poly1d(line_s2x_param)
    s2x_line_prime = np.poly1d(poly_derivative(line_s2x_param))
    s2x_line_dprime = np.poly1d(poly_derivative(poly_derivative(line_s2x_param)))

    s2y_line =  np.poly1d(line_s2y_param)
    s2y_line_prime = np.poly1d(poly_derivative(line_s2y_param))
    s2y_line_dprime = np.poly1d(poly_derivative(poly_derivative(line_s2y_param)))


    s2x_exp2 =  np.poly1d(exp2_s2x_param)
    s2x_exp2_prime = np.poly1d(poly_derivative(exp2_s2x_param))
    s2x_exp2_dprime = np.poly1d(poly_derivative(poly_derivative(exp2_s2x_param)))

    s2y_exp2 =  np.poly1d(exp2_s2y_param)
    s2y_exp2_prime = np.poly1d(poly_derivative(exp2_s2y_param))
    s2y_exp2_dprime = np.poly1d(poly_derivative(poly_derivative(exp2_s2y_param)))#double prime



    fitted_lane_funcs = {
        "s2x_exp": s2x_exp,
        "s2y_exp": s2y_exp,
        "s2x_exp2": s2x_exp2,
        "s2y_exp2": s2y_exp2,
        "s2x_line": s2x_line,
        "s2y_line": s2y_line,
    }

    lon_max = {"p1": [exp_accum_s[-1], 2.0],
               "p2": [line_accum_s[-1], 2.0],
               "p3": [exp_accum_s2[-1], 2.0],
               }
    print(lon_max)
    lon_info_init = {"p1": [0.0, 0.0, 0.0, False],
                     "p2": [10.0, 0.0, 0.0, False],
                     "p3": [0.0, 0.0, 0.0, False],
                     }

    initial_state = GameState(lon_max, lon_info_init, 0.0, fitted_lane_funcs)
    mcts = MCTS(n_simulations=100)
    start_time = time.time()
    ret_traj = mcts.search(initial_state)
    end_time = time.time()
    print("Searching time: ", end_time - start_time)
    

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
        "comfort_weight":10.0,
        "progress_weight":0.1,
    }

    coarse_delta_t = ret_traj["delta_t"]
    coarse_t_max = len(ret_traj["s1"]) * coarse_delta_t
    #First estimate a long enough time
    t_max = coarse_t_max #sec
    delta_t = 1.0

     #sec

    s0_search = construct_init_guess(t_max, delta_t, track_vel_param, exp_accum_s[-1], line_accum_s[-1])
    #s0_search = construct_init_guess_via_search(t_max, delta_t, lon_max, lon_info_init, ret_traj)
    A, ug, lg = construct_linear_constraints(s0_search, delta_t)
    linear_constraints = LinearConstraint(A, lg, ug)
    ub, lb = construct_bounds(s0_search, track_vel_param, exp_accum_s[-1], line_accum_s[-1])
    bounds = Bounds(lb, ub)

    callback_with_params = partial(callback, fitted_lane_funcs=fitted_lane_funcs, track_vel_param=track_vel_param, objective_weight=objective_weight)

   
    
    # #check shape of the collision avoidance objective
    # collision_avoidance_objective_with_params = partial(collision_avoidance_objective, fitted_lane_funcs=fitted_lane_funcs)


    # s1_arr = np.linspace(0, exp_accum_s[-1], 200)
    # s2_arr = np.linspace(0, line_accum_s[-1], 200)

    # S1, S2 = np.meshgrid(s1_arr, s2_arr)

    # Z = collision_avoidance_objective_with_params(S1, S2)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(S1, S2, Z, cmap='viridis')

    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # plt.title('Surface plot of collision objective')

    # plt.show()

    
    # start_time = time.time()   
    # result = minimize(objective, s0_search, args=(fitted_lane_funcs, fitted_lane_prime, track_vel_param, objective_weight), 
    #                   method='SLSQP', 
    #                   jac=gradient,
    #                   constraints=linear_constraints, bounds=bounds, 
    #                   #callback=callback_with_params,
    #                   options={'disp': True,'maxiter': 2000,'ftol': 0.01}
    #                   )

    # if result.success:
    #     print("Optimization was successful!")
    #     print("Optimal Solution:", result.x)
    #     print("Function Value at Optimal Solution:", result.fun)
    # else:
    #     print("Optimization failed.")
    #     print("Reason:", result.message)
    # end_time = time.time()
    # print("Optimization time: ", end_time - start_time)

    
    
    # objective_params = partial(objective, fitted_lane_funcs = fitted_lane_funcs, fitted_lane_prime=fitted_lane_prime, track_vel_param=track_vel_param, objective_weight=objective_weight)
    # jac_params = partial(gradient, fitted_lane_funcs = fitted_lane_funcs, fitted_lane_prime=fitted_lane_prime, track_vel_param=track_vel_param, objective_weight=objective_weight)
    # hess_params = partial(hessian, fitted_lane_funcs = fitted_lane_funcs, fitted_lane_prime=fitted_lane_prime, track_vel_param=track_vel_param, objective_weight=objective_weight)

    # constraints_ipopt_params = partial(constraints_ipopt, A = A)
    # constraints_ipopt_jac_params = partial(constraints_ipopt_jac, A = A)
    # constraints_ipopt_hess_params = partial(constraints_ipopt_hess, A = A)

    # constr = {'type': 'eq', 'fun':  constraints_ipopt_params, 'jac': constraints_ipopt_jac_params}#, 'hess': constraints_ipopt_hess_params}

    # start_time_ipopt = time.time()  
    # result = minimize_ipopt(
    #             fun=objective_params,
    #             x0=s0,
    #             jac=jac_params,
    #             #hess=hess_params,
    #             bounds=bounds,
    #             constraints=constr,
    #             options={'disp': 0}
    #             )
    # end_time_ipopt = time.time()
    # print("Optimal solution:", result.x)
    # print("Optimal value:", result.fun)
    # print("IPOPT processing time: ", end_time_ipopt - start_time_ipopt)
    
    


    #segment optimal solution
    # var_len = result.x.shape[0]
    # single_agent_var_len = int(var_len / 2)
    # single_order_var_len = int(single_agent_var_len / 3)
    # s1_var = result.x[0:single_agent_var_len]
    # s2_var = result.x[single_agent_var_len:var_len]
    # s1 = s1_var[0:single_order_var_len]
    # s2 = s2_var[0:single_order_var_len]
    # s1_dot = s1_var[single_order_var_len:single_order_var_len*2]
    # s2_dot = s2_var[single_order_var_len:single_order_var_len*2]

    s1 = np.array(ret_traj["s1"])
    s2 = np.array(ret_traj["s2"])
    s3 = np.array(ret_traj["s3"])



    t = np.arange(0, t_max, delta_t)

    
    #plot animation
    fig, ax = plt.subplots()

    car1, = plt.plot([], [], 'ro', animated=True)
    car2, = plt.plot([], [], 'go', animated=True)
    car3, = plt.plot([], [], 'bo', animated=True)

    ax.set_xlim(-7, 25)
    ax.set_ylim(-5, 3)

    # Static background: road and buildings
    road1 = plt.plot(x_exp, y_exp, 'r-', lw=1)
    road2 = plt.plot(x_line, y_line, 'g-', lw=1)
    road3 = plt.plot(x_exp2, y_exp2, 'b-', lw=1)

    
    update_with_params = partial(update, s1 = s1, s2 = s2, s3 = s3, s2x_exp = s2x_exp, s2y_exp = s2y_exp, 
                                 s2x_line = s2x_line, s2y_line = s2y_line, s2x_exp2 = s2x_exp2, s2y_exp2 = s2y_exp2, delta_t = delta_t, t_max = t_max)

    ani = animation.FuncAnimation(fig, update_with_params, frames=t,
                                  init_func=init, blit=True)

    #ani.save(filename="Documents/GitHub/TwoVehicleGame/unload_long.mp4", writer="ffmpeg")

    plt.show()


    # #display results graph
    # plt.plot(t, s1, 'r-', label='Player 1 s')
    # plt.plot(t, s2, 'g-', label='Player 2 s')

    # plt.plot(t, s1_dot, 'r:', label='Player 1 s_dot')
    # plt.plot(t, s2_dot, 'g:', label='Player 2 s_dot')

    # plt.xlabel('t')
    # plt.ylabel('s/s_dot')
    # plt.title('Logitudinal info')
    # plt.legend()
    # plt.axis('equal')
    # # Show the plot
    # plt.grid(True)
    # plt.show()
    




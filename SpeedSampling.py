import numpy as np

from RefLine import RefLine
from RefLineGraph import RefLineGraph

import math
import cmath

import matplotlib.pyplot as plt


# def exp_function(x):
#     return -np.exp(-0.4*x) + 1

# def exp_function2(x):
#     return -np.exp(-0.4*(x-7)) + 1

# def qudratic_function(x):
#     return -0.5*x**2 + 8

# def line(x):
#     return 1.0*np.ones(x.shape[0])


def solve_quadratic(a, b, c):
    # Check for degenerate cases where the equation is not quadratic
    if a == 0:
        if b == 0:
            if c == 0:
                return [-1]  # All coefficients are 0
            else:
                return [-1]  # No x satisfies 0 = c (c != 0)
        else:
            return [-c / b]  # Linear equation case (bx + c = 0)
    
    # Calculate the discriminant
    discriminant = b**2 - 4*a*c
    
    # Use complex math to handle both real and complex roots
    if discriminant >= 0:
        # Real roots
        root1 = (-b + math.sqrt(discriminant)) / (2*a)
        root2 = (-b - math.sqrt(discriminant)) / (2*a)
    else:
        # Complex roots
        root1 = (-b + cmath.sqrt(discriminant)) / (2*a)
        root2 = (-b - cmath.sqrt(discriminant)) / (2*a)
    
    return [root1, root2]

#output: success, speed_profile
def SpeedProfileSampling(decision_combination, init_lon_info, horizon, rl_graph, delta_t = 0.5):
    ref_line_graph = rl_graph.ref_line_graph
    #define different maneuver parameters
    v_cruise = 2.00
    v_max = 4.00
    acc_normal = 0.5
    acc_overtake = 1.0
    acc_brake = -1.0
    d_safe = 2.0
    t_buffer = 2.0 #sec

    num_agents = len(decision_combination)

    num_steps = int(horizon / delta_t)

    #first assume all s init values are smaller than s_min
    #init speed profile
    speed_profile = {}
    for i in range(num_agents):
        speed_profile[str(i)] = np.zeros((3, num_steps))
        speed_profile[str(i)][0,0] = init_lon_info[str(i)][0]
        speed_profile[str(i)][1,0] = init_lon_info[str(i)][0]
        speed_profile[str(i)][2,0] = init_lon_info[str(i)][0]


    agents = list(decision_combination.keys())
    #make sure the state of each agent is updated in each iteration

    for i in range(num_steps - 1):
        profile_updated = np.zeros(num_agents)
        for j in range(num_agents):
            self_agent = str(j)
            agent_decisions = decision_combination[self_agent]
            other_agents = list(agent_decisions.keys())
            for other_agent in other_agents:
                cur_self_s = speed_profile[self_agent][0,i]# 0 for s, 1 for s_dot, 2 for s_ddot
                cur_self_v = speed_profile[self_agent][1,i]
                cur_neighbor_s = speed_profile[other_agent][0,i]
                cur_neighbor_v = speed_profile[other_agent][1,i]  
                cur_self_s_max = rl_graph.reflines[int(self_agent)].s[-1]  
                cur_neighbor_s_max = rl_graph.reflines[int(other_agent)].s[-1]     
                ref_line_info = {}
                #retrieve ref_line_info 
                for rlinfo in ref_line_graph[self_agent]:
                    if rlinfo['child_id'] == other_agent:
                        ref_line_info = rlinfo
                s_min_self = ref_line_info['s_min_self']
                s_max_self = ref_line_info['s_max_self']
                s_min_neighbor = ref_line_info['s_min_neighbor']
                s_max_neighbor = ref_line_info['s_max_neighbor']

                neighbor_acc = 0.0
                neighbor_vel = 0.0
                neighbor_s = 0.0
                self_acc = 0.0
                self_vel = 0.0
                self_s = 0.0

                manuver_decision = agent_decisions[other_agent]

                if (manuver_decision == 'overtake'):
                    #self overtake other
                    if cur_self_s < s_min_self and cur_neighbor_s < s_min_neighbor:
                        #self accelerate with high acc
                        ret_self = solve_quadratic(0.5*acc_normal, cur_self_v, -(s_min_self - cur_self_s))
                        ret_neighbor = solve_quadratic(0.5*acc_normal, cur_neighbor_v, -(s_min_neighbor - cur_neighbor_s))
                        if len(ret_self) == 2 and ret_self[0] >= 0 and len(ret_neighbor) == 2 and ret_neighbor[0] >= 0:
                            #print('has solutions')
                            t_self = ret_self[0]
                            t_neighbor = ret_neighbor[0]
                            if t_self + t_buffer > t_neighbor:
                                neighbor_acc = acc_normal * (t_neighbor / (t_self + t_buffer)) * 0.5
                            else:
                                neighbor_acc = acc_normal
                        else:#dumb: make neighbor static
                            neighbor_acc = 0.0
                        neighbor_vel = (neighbor_acc + speed_profile[other_agent][2,i]) * 0.5 * delta_t + speed_profile[other_agent][1,i]
                        if neighbor_vel >= v_cruise:
                            neighbor_vel = v_cruise
                        neighbor_s = (neighbor_vel + speed_profile[other_agent][1,i]) * 0.5 * delta_t + speed_profile[other_agent][0,i]
                        self_acc = acc_normal
                        self_vel = (self_acc + speed_profile[self_agent][2,i]) * 0.5 * delta_t + speed_profile[self_agent][1,i]
                        if self_vel >= v_cruise:
                            self_vel = v_cruise
                        self_s = (self_vel + speed_profile[self_agent][1,i]) * 0.5 * delta_t + speed_profile[self_agent][0,i]
                    elif ((cur_self_s >= s_min_self and cur_neighbor_s < s_min_neighbor) or
                         (cur_self_s >= s_min_self and cur_neighbor_s >= s_min_neighbor)):
                        #both accelerate with normal acc, cruise if v = v_cruise
                        if cur_self_v >= v_cruise:
                            self_acc = 0.0
                        else:
                            self_acc = acc_normal
                        self_vel = (self_acc + speed_profile[self_agent][2,i]) * 0.5 * delta_t + speed_profile[self_agent][1,i]
                        if self_vel >= v_cruise:
                            self_vel = v_cruise
                        self_s = (self_vel + speed_profile[self_agent][1,i]) * 0.5 *delta_t + speed_profile[self_agent][0,i]

                        if cur_neighbor_v >= v_cruise:
                            neighbor_acc = 0.0
                        else:
                            neighbor_acc = acc_normal
                        neighbor_vel = (neighbor_acc + speed_profile[other_agent][2,i]) * 0.5 * delta_t + speed_profile[other_agent][1,i]
                        if neighbor_vel >= v_cruise:
                            neighbor_vel = v_cruise
                        neighbor_s = (neighbor_vel + speed_profile[other_agent][1,i]) * 0.5 * delta_t + speed_profile[other_agent][0,i]

                    elif cur_self_s < s_min_self and cur_neighbor_s >= s_min_neighbor:
                        return False, speed_profile

                    if (self_s >= cur_self_s_max):
                        self_s = cur_self_s_max
                        self_vel = 0.0
                        self_acc = 0.0
                    if (neighbor_s >= cur_neighbor_s_max - d_safe):
                        neighbor_s = cur_neighbor_s_max - d_safe
                        neighbor_vel = 0.0
                        neighbor_acc = 0.0

                    if profile_updated[int(self_agent)] < 0.5:
                        speed_profile[self_agent][0,i+1] = self_s
                        speed_profile[self_agent][1,i+1] = self_vel
                        speed_profile[self_agent][2,i+1] = self_acc
                        profile_updated[int(self_agent)] = 1.0
                    elif speed_profile[self_agent][0,i+1] < self_s:
                        speed_profile[self_agent][0,i+1] = self_s
                        speed_profile[self_agent][1,i+1] = self_vel
                        speed_profile[self_agent][2,i+1] = self_acc
                        profile_updated[int(self_agent)] = 1.0

                    if profile_updated[int(other_agent)] < 0.5:
                        speed_profile[other_agent][0,i+1] = neighbor_s
                        speed_profile[other_agent][1,i+1] = neighbor_vel
                        speed_profile[other_agent][2,i+1] = neighbor_acc
                        profile_updated[int(other_agent)] = 1.0
                    elif speed_profile[other_agent][0,i+1] > neighbor_s:
                        speed_profile[other_agent][0,i+1] = neighbor_s
                        speed_profile[other_agent][1,i+1] = neighbor_vel
                        speed_profile[other_agent][2,i+1] = neighbor_acc
                        profile_updated[int(other_agent)] = 1.0


    return True, speed_profile


# x_exp = np.linspace(-4, 24, 400)
# y_exp = exp_function(x_exp)
# x_exp2 = np.linspace(18, 3, 400)
# y_exp2 = exp_function2(x_exp2) 


# x_line = np.linspace(-6, 24, 400)
# y_line = line(x_line)

# refline1 = RefLine(0, x_exp, y_exp)
# refline2 = RefLine(1, x_exp2, y_exp2)
# refline3 = RefLine(2, x_line, y_line)

# ref_line_arr = [refline1, refline2, refline3]

# rl_graph = RefLineGraph(ref_line_arr)
# rl_graph.buildRefLineGraph()
# ref_line_graph = rl_graph.ref_line_graph

# consistent_combinations = rl_graph.generate_consistent_combinations()

# init_lon_info = {}

# for rl in ref_line_arr:
#     init_lon_info[rl.get_id()] = [0.1,0.0,0.0]

# #init_lon_info['2'][0] = 6



# horizon = 20

# for decision_combination in consistent_combinations:

#     ret, speed_profile = SpeedProfileSampling(decision_combination, init_lon_info, horizon, ref_line_graph)

#     if ret:
#         print(decision_combination)
#         plt.plot(np.linspace(0,horizon, int(horizon/0.5)), speed_profile['0'][0], 'r-',lw = 1.0, label='Player 0 s')
#         # plt.plot(np.linspace(0,horizon, int(horizon/0.1)), speed_profile['0'][1], 'r-', label='Player 0 s_dot')
#         # plt.plot(np.linspace(0,horizon, int(horizon/0.1)), speed_profile['0'][2], 'r-', label='Player 0 s_ddot')
#         plt.plot(np.linspace(0,horizon, int(horizon/0.5)), speed_profile['1'][0], 'g--', lw = 1.0, label='Player 1 s')
#         # plt.plot(np.linspace(0,horizon, int(horizon/0.1)), speed_profile['1'][1], 'g--')
#         # plt.plot(np.linspace(0,horizon, int(horizon/0.1)), speed_profile['1'][2], 'g.-')
#         plt.plot(np.linspace(0,horizon, int(horizon/0.5)), speed_profile['2'][0], 'm:', lw = 2.0, label='Player 2 s')
#         # plt.plot(np.linspace(0,horizon, int(horizon/0.1)), speed_profile['2'][1], 'm--', label='Player 2 s_dot')
#         # plt.plot(np.linspace(0,horizon, int(horizon/0.1)), speed_profile['2'][2], 'm--', label='Player 2 s_ddot')
#         plt.legend()
#         plt.show()
#         #print(speed_profile['1'][0:])
#         #print(speed_profile['2'][0:])





# # for idx, combination in enumerate(consistent_combinations):
# #     print(f"Combination {idx+1}:")
# #     for vehicle, decisions in combination.items():
# #         print(f"  {vehicle}: {decisions}")
# #     print()

# # # Count total combinations
# # print(f"Total consistent combinations: {len(consistent_combinations)}")




# # # Static background: road and buildings
# # road1 = plt.plot(x_exp, y_exp, 'r-', lw=1)
# # road2 = plt.plot(x_line, y_line, 'g-', lw=1)
# # road3 = plt.plot(x_exp2, y_exp2, 'b-', lw=1)
# # print(rl_graph.ref_line_graph["0"])
# # print(rl_graph.ref_line_graph["1"])
# # print(rl_graph.ref_line_graph["2"])

# # plt.axis('equal')
# # plt.show()

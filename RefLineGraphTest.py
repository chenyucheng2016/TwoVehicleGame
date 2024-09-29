import numpy as np

from RefLine import RefLine
from RefLineGraph import RefLineGraph

import matplotlib.pyplot as plt


def exp_function(x):
    return -np.exp(-0.4*x) + 1

def exp_function2(x):
    return -np.exp(-0.4*(x-7)) + 1

def qudratic_function(x):
    return -0.5*x**2 + 8

def line(x):
    return 1.0*np.ones(x.shape[0])

x_exp = np.linspace(-4, 24, 400)
y_exp = exp_function(x_exp)

x_exp2 = np.linspace(18, 3, 400)
y_exp2 = exp_function2(x_exp2) 


x_line = np.linspace(-6, 24, 400)
y_line = line(x_line)

refline1 = RefLine(0, x_exp, y_exp)
refline2 = RefLine(1, x_exp2, y_exp2)
refline3 = RefLine(2, x_line, y_line)

ref_line_arr = [refline1, refline2, refline3]

rl_graph = RefLineGraph(ref_line_arr)
rl_graph.buildRefLineGraph()

# Static background: road and buildings
road1 = plt.plot(x_exp, y_exp, 'r-', lw=1)
road2 = plt.plot(x_line, y_line, 'g-', lw=1)
road3 = plt.plot(x_exp2, y_exp2, 'b-', lw=1)
print(rl_graph.ref_line_graph["0"])
print(rl_graph.ref_line_graph["1"])
print(rl_graph.ref_line_graph["2"])

plt.axis('equal')
plt.show()

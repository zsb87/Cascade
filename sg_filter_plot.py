# Example Usage:
# python sg.py position.dat 7 2
# 
# NOTICE:  THE CODE IS PROBLEMATIC!!
#  ERROR MESSAGE AS FOLLOWS:
# 
# Users/shibozhang/Documents/Cascade/sg_filter_plot.py:24: RuntimeWarning: invalid value encountered in power
#   print(list(map(lambda i: i**x, a)))
# [array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan]), array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan]), array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan]), array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan]), array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan]), array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]), array([ 0.11524449,  0.09491014,  0.07994289,  0.06752567,  0.05592488,
#         0.04737137,  0.04001336,  0.03295319,  0.02822902,  0.02324814,
#         0.01958193]), array([ 0.20745929,  0.18012068,  0.15896794,  0.14058693,  0.12256161,
#         0.10861236,  0.09605382,  0.08339602,  0.07451226,  0.06469317,
#         0.05709583]), array([ 0.27250852,  0.24246693,  0.21867866,  0.19755839,  0.1763755 ,
#         0.15961089,  0.14419546,  0.12829922,  0.11689313,  0.10400673,
#         0.0938027 ]), array([ 0.34801388,  0.31652781,  0.29107321,  0.26803374,  0.24445478,
#         0.22541491,  0.20757253,  0.1887927 ,  0.17504787,  0.15921066,
#         0.14640722]), array([ 0.41081091,  0.37925395,  0.3533807 ,  0.32965419,  0.30503726,
#         0.28488641,  0.26575872,  0.24534413,  0.23020078,  0.21251762,
#         0.19801936])]
# Traceback (most recent call last):
#   File "/Users/shibozhang/Documents/Cascade/sg_filter_plot.py", line 110, in <module>
#     plot_results(data, size, order)
#   File "/Users/shibozhang/Documents/Cascade/sg_filter_plot.py", line 98, in plot_results
#     ["Velocity",     smooth(*params, deriv=1)],
#   File "/Users/shibozhang/Documents/Cascade/sg_filter_plot.py", line 43, in smooth
#     f = sg_filter(x[start:end], order, deriv)
#   File "/Users/shibozhang/Documents/Cascade/sg_filter_plot.py", line 25, in sg_filter
#     A = np.r_[map(expa, range(0,m+1))].transpose()
#   File "/Users/shibozhang/anaconda/lib/python3.6/site-packages/numpy/lib/index_tricks.py", line 338, in __getitem__
#     res = _nx.concatenate(tuple(objs), axis=self.axis)
# ValueError: zero-dimensional arrays cannot be concatenated
# [Finished in 0.9s with exit code 1]
# [shell_cmd: python -u "/Users/shibozhang/Documents/Cascade/sg_filter_plot.py"]
# [dir: /Users/shibozhang/Documents/Cascade]
# [path: /usr/bin:/bin:/usr/sbin:/sbin]
# 


import math
import sys

import numpy as np
import numpy.linalg
import pylab as py

def sg_filter(x, m, k=0):
    """
    x = Vector of sample times
    m = Order of the smoothing polynomial
    k = Which derivative
    """
    mid = len(x) / 2        
    a = x - x[mid]
    expa = lambda x: map(lambda i: i**x, a)
    print("expa")
    print(expa)
    print("m")
    print(m)
    print(list(map(lambda i: i**x, a)))
    A = np.r_[map(expa, range(0,m+1))].transpose()
    Ai = np.linalg.pinv(A)

    return Ai[k]

def smooth(x, y, size=5, order=2, deriv=0):

    if deriv > order:
        print("deriv must be <= order") 

    n = len(x)
    m = size

    result = np.zeros(n)

    for i in range(m, n-m):
        start, end = i - m, i + m + 1
        print(x[start:end])
        f = sg_filter(x[start:end], order, deriv)
        result[i] = np.dot(f, y[start:end])

    if deriv > 1:
        result *= math.factorial(deriv)

    return result

def plot(t, plots):
    n = len(plots)

    for i in range(0,n):
        label, data = plots[i]

        plt = py.subplot(n, 1, i+1)
        plt.tick_params(labelsize=8)
        py.grid()
        py.xlim([t[0], t[-1]])
        py.ylabel(label)

        py.plot(t, data, 'k-')

    py.xlabel("Time")

def create_figure(size, order):
    fig = py.figure(figsize=(8,6))
    nth = 'th'
    if order < 4:
        nth = ['st','nd','rd','th'][order-1]

    title = "%s point smoothing" % size
    title += ", %d%s degree polynomial" % (order, nth)

    fig.text(.5, .92, title,
             horizontalalignment='center')

def load(name):
    # f = open(name)    
    # dat = [map(float, x.split(' ')) for x in f]
    # f.close()
    # xs = [x[0] for x in dat]
    # ys = [x[1] for x in dat]

    import pandas as pd
    dat = pd.read_csv(name, names = ['x', 'y'])
    print(dat)

    return dat['x'].as_matrix(), dat['y'].as_matrix()

def plot_results(data, size, order):
    t, pos = load(data)
    params = (t, pos, size, order)

    plots = [
        ["Position",     pos],
        ["Velocity",     smooth(*params, deriv=1)],
        ["Acceleration", smooth(*params, deriv=2)]
    ]

    create_figure(size, order)
    plot(t, plots)

if __name__ == '__main__':
    data = '_sg_data.txt'
    size = 5
    order = 1

    plot_results(data, size, order)
    py.show()
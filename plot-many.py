#!/usr/bin/ipython

import numpy as np
import sys

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def func(x, a, b, c):
    """ function to fit the data to. """
    return a * np.exp(-b * x) + c

    
data = np.loadtxt(sys.argv[1])

num_sims = 10

MI_per_sim = data.size/num_sims

data_mat = np.zeros((MI_per_sim, num_sims))

err_bars = np.zeros(MI_per_sim)

avg_MI = np.zeros(MI_per_sim)



for i in range(num_sims):
    data_mat[:, i] = data[i*MI_per_sim:(i+1)*MI_per_sim]


for i in range(MI_per_sim):
    err_bars[i] = np.std(data_mat[i, :])
    avg_MI[i] = np.mean(data_mat[i, :])


    
x = np.array(range(avg_MI.size))
y = func(x, 0.5, 0.5, 0.5) # Some guesses for a, b, c

x_t = np.transpose(x)
xx = np.array([x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t])


p, pcov = curve_fit(func, xx, data_mat)




fit = func (x, p[0], p[1], p[2])

plt.errorbar(x, avg_MI, err_bars, linestyle='None', marker='*')

plt.plot(x, fit, "k-")

for i in range(num_sims):
    plt.plot(x, data_mat[:, i], "ko")

plt.ylabel('Mitotic Index')
plt.xlabel('Division Step/10000')
plt.savefig('MitoticIndex.svg')

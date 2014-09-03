#!/usr/bin/ipython



import numpy as np
import sys

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


if (len(sys.argv) < 2):
    print "Usage: plot-many.py mit-inddex.dat num_sims"
    sys.exit()


def func(x, a, b, c):
    """ function to fit the data to. """
    return a * np.exp(-b * x) + c

    
data = np.loadtxt(sys.argv[1])

num_sims = int(sys.argv[2])

MI_per_sim = data.size/num_sims

data_mat = np.zeros((MI_per_sim, num_sims))

err_bars = np.zeros(MI_per_sim)

avg_MI = np.zeros(MI_per_sim)



for i in range(num_sims):
    data_mat[:, i] = data[i*MI_per_sim:(i+1)*MI_per_sim]


for i in range(MI_per_sim):
    avg_MI[i] = np.mean(data_mat[i, :])
    err_bars[i] = np.std(data_mat[i, :])

mask = avg_MI != 0
mask = mask * (avg_MI < 0.2) # Basically remove all unwanted data points
offset = mask.sum() # Mask will be 1D since avg_MI is 1D
avg_MI = avg_MI[mask]
err_bars = err_bars[mask] 
    
x = np.array(range(avg_MI.size))
y = func(x, 0.5, 0.5, 0.5) # Some guesses for a, b, c

p, pcov = curve_fit(func, x, avg_MI)


fit = func (x, p[0], p[1], p[2])

for i in range(num_sims):
    plt.plot(x, data_mat[:, i][mask], '.', color='#B0B0B0')

plt.errorbar(x, avg_MI, err_bars, linestyle='line', marker='o', color='black')

plt.plot(x, fit, "k-")



plt.ylabel('Mitotic Index')
plt.xlabel('Time Step/1000')
plt.savefig('MitoticIndex.png')

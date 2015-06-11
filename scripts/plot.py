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

allAvgMI=100*avg_MI.copy()
allErrBars=100*err_bars.copy()

mask = avg_MI > 0
mask = mask * (avg_MI < 0.2) # Basically remove all unwanted data points
simLength = avg_MI.size
offset = simLength - mask.sum() # Mask will be 1D since avg_MI is 1D
avg_MI = avg_MI[mask]
err_bars = err_bars[mask]
avg_MI = avg_MI[15:]
err_bars = err_bars[15:]

x = np.arange(0, avg_MI.size)
p, pcov = curve_fit(func, x, avg_MI)
fit = 100*func (x, p[0], p[1], p[2])

print "MI = %f exp(-%f x) + %f" % (p[0], p[1], p[2])

x+= offset
avg_MI*=100
err_bars*=100
x+= 15
for i in range(num_sims):
    plt.plot(x, 100*data_mat[:, i][mask][15:], '.', color='#B0B0B0')

plt.errorbar(x, avg_MI, err_bars, linestyle='line', ecolor='#B0B0B0')
plt.plot(x, avg_MI, 'k.-', label="Average of 10 simulations")

plt.plot(x, fit, "-", lw=2.0, color="blue", label="Fit to average")
plt.xlim(0, simLength)
plt.ylim(0, 10)

plt.ylabel('Mitotic Index (%)')
plt.xlabel('Time')
a = plt.axes([0.55, 0.55, 0.3, 0.3])
plt.plot(xrange(allAvgMI.size-1), allAvgMI[:-1], 'k-')
plt.tick_params(axis='both', which='major', labelsize=6)
plt.xlabel('Time')
plt.ylabel('MI (%)')
plt.xlim(0, allAvgMI.size)

plt.savefig('MitoticIndex.eps')

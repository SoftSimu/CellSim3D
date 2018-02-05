#!/usr/bin/env ipython3



import numpy as np
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.style
matplotlib.style.use('seaborn')

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning

if (len(sys.argv) < 2):
    print("Usage: plot-many.py mit-inddex.dat num_sims")
    sys.exit()


def func(x, a, b, c):
    """ function to fit the data to. """
    return a * np.exp(-b * x) + c


data = np.loadtxt(sys.argv[1])
num_sims = int(sys.argv[2])
MI_per_sim = int(data.size/num_sims)
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



n = 0
with warnings.catch_warnings():
    warnings.simplefilter("error", OptimizeWarning)
    warnings.simplefilter("error", RuntimeWarning)

    done = False

    while not done:
        print("Trying optimization after skipping ", n, " datapoints.")
        avg_MI_t = avg_MI[n:]
        err_bars_t = err_bars[n:]
        x = np.arange(0, avg_MI_t.size)

        try:
            p, pcov = curve_fit(func, x, avg_MI_t, sigma=err_bars_t)
            if p[2] > 0.03 or np.abs(p[1]) > 1 or p[0] < 0:
                raise OptimizeWarning
        except OptimizeWarning:
            n+=1
        except RuntimeWarning:
            n+=1
        else:
            done = True

fit = 100*func (x, p[0], p[1], p[2])
avg_MI = avg_MI[n:]
err_bars = err_bars[n:]

print("MI = %f exp(-%f x) + %f" % (p[0], p[1], p[2]))

x+= offset
avg_MI*=100
err_bars*=100
x+= n
for i in range(num_sims):
    plt.plot(x, 100*data_mat[:, i][mask][n:], '.', color='#B0B0B0', alpha=0.5)

plt.errorbar(x, avg_MI, err_bars, ecolor='#B0B0B0', alpha=0.5)
plt.plot(x, avg_MI, 'k-', label="Average of 102 simulations")

plt.plot(x, fit, "-", lw=2.0, color="blue",
         label="MI = %f exp(-%f x) + %f" % (p[0], p[1], p[2]))
plt.legend()

plt.xlim(0, simLength)
plt.ylim(0, 10)

plt.ylabel('Mitotic Index (%)')
plt.xlabel('Time')
plt.title('%s' % sys.argv[1])
# a = plt.axes([0.55, 0.55, 0.3, 0.3])
# plt.plot(range(allAvgMI.size-1), allAvgMI[:-1], 'k-')
# plt.tick_params(axis='both', which='major', labelsize=6)
# plt.xlabel('Time')
# plt.ylabel('MI (%)')
# plt.xlim(0, allAvgMI.size)
plt.savefig(sys.argv[1][:-3])

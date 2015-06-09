#!/usr/bin/env python



import numpy as np
import sys, argparse, os

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


desc="Calculates and plots average mitotic index for a number of input data files"

parser=argparse.ArgumentParser(description=desc)

parser.add_argument("numSims",
                    help="Number of simulations, \
                    must be same for all data files")

parser.add_argument("--datPaths", nargs="+",
                    help="paths to data files")


args = parser.parse_args()


datPaths = [os.path.abspath(p) for p in args.datPaths]

labels = ['$\gamma_{m} = 5$', '$\gamma_{m} = 10$', '$\gamma_{m} = 15$', '$\gamma_{m} = 20$', '$\gamma_{m} = 25$']

offsets = [20, 25, 32, 40, 55]

fig, ax = plt.subplots()

ax_inset = plt.axes([0.2, 0.585, 0.3, 0.3])

cm = plt.get_cmap('rainbow')

n = []
yMax = 0

def func(x, a, b, c):
    """ function to fit the data to. """
    return a * np.exp(-b * x) + c

def FitAndPlot(datPath, plotName, c, offset):
    data = np.loadtxt(datPath)
    num_sims = int(args.numSims)
    MI_per_sim = data.size/num_sims
    data_mat = np.zeros((MI_per_sim, num_sims))
    err_bars = np.zeros(MI_per_sim)
    avg_MI = np.zeros(MI_per_sim)

    for i in range(num_sims):
        data_mat[:, i] = data[i*MI_per_sim:(i+1)*MI_per_sim]

    for i in range(MI_per_sim):
        avg_MI[i] = np.mean(data_mat[i, :])
        err_bars[i] = np.std(data_mat[i, :])

    allAvgMI = 100*avg_MI.copy()
    allErrBars = err_bars.copy()
    allX = np.arange(0, allAvgMI.size)

    #mask = avg_MI != 0
    #mask = mask * (avg_MI < 0.1) # Basically remove all unwanted data points
    #simLength = avg_MI.size
    #offset = simLength - mask.sum() # Mask will be 1D since avg_MI is 1D
    #avg_MI = avg_MI[mask]
    #err_bars = err_bars[mask]
    avg_MI = avg_MI[offset:]
    err_bars = err_bars[offset:]

    x = np.arange(0, avg_MI.size)
    p, pcov = curve_fit(func, x, avg_MI)
    fit = func (x, p[0], p[1], p[2])
    x+= offset
    avg_MI*=100
    err_bars*=100
    fit *= 100


    #ax.plot(x, avg_MI, '-',  color=cm(c))
    ax.plot(x, fit, "-", lw=2.0, label=plotName, color=cm(c))

    ax_inset.plot(allX, allAvgMI, '-', lw=0.5, color=cm(c))

for i in xrange(len(datPaths)):
    FitAndPlot(datPaths[i], labels[i], i*(1.0/len(datPaths)), offsets[i])

print n
ax.set_xlabel('Time')
ax.set_ylabel('Mitotic Index (%)')
ax.set_xlim([0, 100])
ax.set_ylim([0, 10])
ax.legend()

ax_inset.set_xticklabels([])
ax_inset.set_yticklabels([])
ax_inset.set_xlim([20, 100])
ax_inset.set_ylim([0, 10])


plt.savefig('MitoticIndex_many.eps')

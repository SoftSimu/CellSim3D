#!/usr/bin/env python

import numpy as np
import sys

if len(sys.argv) < 3:
    print "Usage: plot.py {Path to mit-index.dat} {Path to output}"
    print "Don't put the extension in the output"
    print "e.g"
    print "/path/to/plot.py /path/to/mit-index.dat /path/to/image"
    sys.exit(0)

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def func(x, a, b, c):
    """ function to fit the data to. """
    return a * np.exp(-b * x) + c


data = np.loadtxt(sys.argv[1])
#data = data[data!=0.0]
x = np.array(range(data.size))
y = func(x, 0.5, 0.5, 0.5) # Some guesses for a, b, c

p, pcov = curve_fit(func, x, data)

fit = func (x, p[0], p[1], p[2])

plt.plot(x, data,".")
#plt.plot(x, fit, "k-")
plt.ylabel('Mitotic Index')
plt.xlabel('Time Step/1000')
plt.savefig('MitoticIndex.png')

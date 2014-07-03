#!/usr/bin/python

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
x = np.array(range(data.size))
y = func(x, 0.5, 0.5, 0.5) # Some guesses for a, b, c

p, pcov = curve_fit(func, x, data)

fit = func (x, p[0], p[1], p[2])

plt.plot(x, data, '.')
plt.plot(x, fit, "k-")
plt.ylabel('Mitotic Index')
plt.xlabel('Division Step/10000')
plt.savefig('MitoticIndex.svg')

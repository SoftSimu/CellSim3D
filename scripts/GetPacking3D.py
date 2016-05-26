#!/usr/bin/env python3

#This script measures 3D packing. It is unclear how useful this measurement will
#be. But I think it is worth seeing what we get.


import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
import os, sys, argparse
import csv
from scipy.optimize import curve_fit
from scipy.stats import lognorm

sys.path.append("/home/pranav/dev/celldiv/scripts")
import celldiv

from scipy.spatial.qhull import QhullError

desc="""
This script will measure the polygonal packing of a 3D system of cells with
Voronoi Tesselation.
"""

parser = argparse.ArgumentParser(description=desc)

parser.add_argument("--traj", nargs="+",
                    help="One or more absolute or relative paths to trajectories")
parser.add_argument("--endat", type=int,
                    help="time step to stop processing at")
parser.add_argument("-k", "--skip", type=int, required=False,
                    help="Frame skip rate")
args = parser.parse_args()

trajPaths = [os.path.abspath(p) for p in args.traj]
storePaths = [os.path.split(p)[0] for p in trajPaths]

if args.endat is None:
    endat = 1e100
else:
    endat = args.endat

aRate = args.skip

if args.skip is None:
    aRate = 1

def measurePacking(filePath, storePath, name=None):
    fileName = os.path.splitext(os.path.split(filePath)[1])[0]
    print("Processing %s" % filePath)
    if name == None:
        storePath  += '/' + fileName + '/' + '3DPacking/'
    else:
        storePath +=  '/' + fileName + '/' + name + '/'

    try:
        os.makedirs(storePath)
    except:
        pass

    dataPath = storePath + "data.dat"
    # Now read trajectory file and get the centre of masses of
    # all cells at every time step
    if os.path.isfile(dataPath):
        if os.path.getmtime(dataPath) > os.path.getmtime(filePath):
            print("{0} already processed.".format(filePath))
            print("Skipping")
            return dataPath

    print("Generating cell CoMs...")

    with celldiv.TrajHandle(filePath) as s,\
         open(storePath + "data.dat", 'w') as f:

        while True:
            try:
                frame = s.ReadFrame(inc = aRate)
            except:
                pass
            step = s.step
            if s.fatalError:
                break

            nCells = len(frame)

            if nCells > 3:
                CoMs = np.array([np.mean(cell, axis=0) for cell in frame])
                sysCoM = np.mean(CoMs, axis=0)
                CoMs -= sysCoM

                print("Doing Del of", CoMs.shape[0], "cells at step", step)
                try:
                    d = Delaunay(CoMs)
                    indices, indptr = d.vertex_neighbor_vertices
                    count = []
                    for p in range(nCells):
                        n = len(indptr[indices[p]:indices[p+1]])
                        if n > len(count):
                            for i in range(len(count), n):
                                count.append(0)
                        count[n-1] += 1/nCells

                    f.write(", ".join([str(c) for c in count]))
                    f.write("\n")

                except QhullError:
                    print("delaunay failed, skipping")
                    pass

    return dataPath

fig, ax = plt.subplots()
figll, axll = plt.subplots()

def logNormal(x, s):
    return 1/(s*x*np.sqrt(2*np.pi) * np.exp(-0.5*(np.log(x)/s)**2))

def fitfunc(x, m, f, mi, c, d):
    return m*np.exp(-1 * ( (np.log(x + f*mi) - c)**2)/d)



def fitPacking(dataPath, interval=10):
    """Function to fit the resulting distribution of the packing. Assumes
    that interval of 10 timesteps is enough.
    """
    if not os.path.isfile(dataPath):
        print("Something went wrong with {0}, rerun".format(filePath))
        return

    # code for fitting the resulting distribution

    # First get the average it st devs
    data = []
    with open(dataPath, "r") as dFile:
        data = list(csv.reader(dFile))[(-1*interval):]

    data = [[float(v) for v in l] for l in data]
    maxLen = max([len(l) for l in data])

    for l in data:
        if len(l) < maxLen:
            for i in range(maxLen - len(l)):
                l.append(0)

    data = np.vstack(data)
    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    y = mean
    x = np.arange(y.shape[0])
    m = y.max()
    mi = np.argmax(m)
    f = 0.3
    c = 2.7
    d = 0.04

    guesses = np.array([m, f, mi, c, d])

    p, pcov = curve_fit(fitfunc, xdata = x,
                        ydata = y, p0=guesses)

    xx = np.linspace(0, x.max(), 1000)

    m, f, mi, c, d = p
    print(p)
    fit = fitfunc(xx, m, f, mi, c, d)

    ax.errorbar(x, y, yerr=stddev, fmt='.')
    l = "${max:.2f} \\times \\exp{{ \\left[ -\\frac{{1}}{{{den:.2f}}} \\left[ \ln\\left(x + {fmaxi:0.2f}\\right) - {cunt:0.2f}\\right]^2 \\right] }}$".format(max=m, fmaxi=f*mi, cunt=c, den=d)

    #sys.exit()
    ax.plot(xx, fit, '-', label=l)

    axll.semilogy(x, y, '.')
    axll.semilogy(xx, fit, '-')





for i in range(len(trajPaths)):
    fitPacking(dataPath = measurePacking(trajPaths[i], storePaths[i]))


ax.set_xlabel("n")
ax.set_ylabel("fraction")

axll.set_xlabel("n")
axll.set_ylabel("fraction")
h, l = ax.get_legend_handles_labels()
fig.legend(h, l)

fig.savefig("fit.png")
figll.savefig("llfit.png")

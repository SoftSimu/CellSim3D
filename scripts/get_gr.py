#!/usr/bin/env python2

# This script will output g(r) for aribitrary no. of timestamps.

import sys
import matplotlib
import os
matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from scipy.spatial import Voronoi, voronoi_plot_2d


minNumArgs = 2

erorrMessage = """
Usage: shaper input.xyz t1 t2 t3 ...
input.xyz: input trajectory file
t1 t2 t3 ...: timestamps, use any number of them you like
"""

if len(sys.argv) < minNumArgs:
    print errorMessage
    sys.exit(1)


trajPath = os.path.abspath(sys.argv[1])
storPath, fileName = os.path.split(trajPath)
trajName = os.path.splitext(fileName)[0]

storPath += "/" + fileName

timeStamps = sys.argv[2:]

# Make timeStamps ints

timeStamps = [int(ts) for ts in timeStamps]
timeStamps.sort()



# seek and compute g(r) for each time stamp requested

X = []
Y = []
Z = []

CMx = []
CMy = []
CMz = []

CM_x = 0
CM_y = 0
CM_z = 0

gr = plt.figure(1)
vor = plt.figure(2)


def getNonNans(a):
    nans = np.isnan(a)
    notnans = np.invert(nans)
    return a[notnans]


def ComputeGr(X, Y, Z):
    print "okay :("
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    X = getNonNans(X)
    Y = getNonNans(Y)
    Z = getNonNans(Z)

    vol = (X.max() - X.min()) * (Y.max() - Y.min()) * (Z.max() - Z.min())
    # num density
    N = X.size
    rho = N/vol
    print rho

    dx = X - X[:, np.newaxis]
    dy = Y - Y[:, np.newaxis]
    dz = Z - Z[:, np.newaxis]

    r_cells = np.sqrt(dx*dx + dy*dy + dz*dz)
    dr = 0.01
    r_max = ceil(r_cells.max())

    # looping through np.array is really slow
    # so converting to list
    rr = np.arange(2*dr, r_max, dr).tolist()
    g_r = []
    for r in rr:
        n_over = r_cells > (r - dr/2.)
        n_under = r_cells < (r + dr/2.)
        n_mask = n_over*n_under
        n = n_mask.sum()
        s = 4*np.pi * r*r * dr
        n = n/s/rho
        n = n/N/2
        g_r.append(n)

    plt.plot(rr,g_r)
    plt.legend("a")

    g_r = []




def Voro (X, Y, ts, xstr, ystr):
    """Compute the voronoi diagram for a given set of points"""
    X = np.array(X)
    Y = np.array(Y)
    X = getNonNans(X)
    Y = getNonNans(Y)
    #print X.size==Y.size
    points = np.zeros([X.size, 2])
    for i in xrange(X.size):
        points[i,0] = X[i]
        points[i,1] = Y[i]

    voronoi = Voronoi(points)
    #voronoi_plot_2d(voronoi)
    plt.title("CoM Voronoi\ntime stamp = %s" % ts)
    for region in voronoi.regions:
        if not -1 in region:
            polygon = [voronoi.vertices[i] for i in region]
            plt.fill(*zip(*polygon))

    plt.xlim([np.floor(X.min()), np.ceil(X.max())])
    plt.ylim([np.floor(Y.min()), np.ceil(Y.max())])
    plt.xlabel(xstr)
    plt.ylabel(ystr)
    plt.savefig("vorlater.svg")


with open(trajPath, "r") as trajFile:
    for timeStamp in timeStamps:
        print "Moving to time step %d ..." % timeStamp
        # move to time stamp, ignore everything
        while True:
            line = trajFile.readline().strip()
            nAtoms = int(line)
            line = trajFile.readline().strip()
            step = int(line[6:])
            if (step >= timeStamp):
                break;
            print "skiping time step %d" % step

            for n in xrange(nAtoms):
                line = trajFile.readline()

        # we are at the timeStamp now
        nCells = nAtoms/192
        print "We have %d cells in the system" % (nCells)
        for n in xrange(nCells):
            for m in xrange(192):
                line = trajFile.readline().strip()
                line = line.split(", ")

                x = float(line[0])
                y = float(line[1])
                z = float(line[2])

                X.append(x)
                Y.append(y)
                Z.append(Z)

                CM_x += x
                CM_y += y
                CM_z += z

            CMx.append(CM_x/192)
            CMy.append(CM_y/192)
            CMz.append(CM_z/192)

            CM_x = 0
            CM_y = 0
            CM_z = 0

        #Now process for g(r)

        Voro(CMx, CMy, timeStamp, "X", "Y")
        #Voro(CMx, CMz, timeStamp, "X", "Z")
        #Voro(CMy, CMy, timeStamp, "Y", "Z")
        X = []
        Y = []
        Z = []
        CMx = []
        CMy = []
        CMz = []

gr.savefig("gr.png")

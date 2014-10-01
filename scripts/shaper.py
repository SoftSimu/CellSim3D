#!/usr/bin/env python2


# This Script will do shape analysis on the trajectory
# and produce the polygons of the different cells.

import sys
import matplotlib
import os
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


numArgs = 3

errorMessage = """
Usage: shaper input output startStep

input:      Path to the trajectory output by the simulation.
output:     Path to save a xyz file of the time step requested for viewing in vmd,
            etc.
startStep:  The step to process; unknown behaviour if this is > total sim steps.
"""

if len(sys.argv) < numArgs:
    print errorMessage
    sys.exit(1)

os.system('rm pics/*.png')


trajName = sys.argv[1]
outputName = sys.argv[2]
startTime = int(sys.argv[3])
doneSeek = False
step = 0

X = []
Y = []
Z = []

CMx = []
CMy = []
CMz = []

CM_x = 0
CM_y = 0
CM_z = 0
outFile = open(outputName, 'w')

with open(trajName) as trajFile:
    # First seek the time step to begin processing
    print "Moving to time step %d..." % startTime
    while step < startTime:
        line = trajFile.readline().strip()
        # First line is always no. of atoms
        nAtoms = int(line)
        # Second line is always a comment with the time step no.
        line = trajFile.readline().strip()
        step = int(line[6:])
        print "Skipping time step %d" % step

        # Start skipping lines with coordinates
        for n in xrange(nAtoms):
            line = trajFile.readline()

    # Should be at the correct time step now
    # Begin processipng
    # For now only do one time step
    line = trajFile.readline().strip()
    nAtoms = int(line)
    nCells = nAtoms/192
    line = trajFile.readline().strip()
    step = int(line[6:])
    #outFile.write("%d\n" % nAtoms)
    #outFile.write("Step: %d\n" % step)
    print "We have %d cells in the system" % (nAtoms/192)
    for n in xrange(nCells):
        for m in xrange(192):
            line = trajFile.readline().strip()
            line = line.split(",  ")

            x = float(line[0])
            y = float(line[1])
            z = float(line[2])

            X.append(x)
            Y.append(y)
            Z.append(z)

            CM_x += x
            CM_y += y
            CM_z += z

            #outFile.write("C    %f    %f    %f\n" % (x, y, z))



        CMx.append(CM_x/192)
        CMy.append(CM_y/192)
        CMz.append(CM_z/192)

        CM_x = 0
        CM_y = 0
        CM_z = 0


outFile.close()

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

CMx = np.array(CMx)
CMy = np.array(CMy)
CMz = np.array(CMz)


# Get rid of zeros
X = X[X != 0.0]
Y = Y[Y != 0.0]
Z = Z[Z != 0.0]


# Normalize



CMX = CMx.sum() / CMx.size
CMY = CMy.sum() / CMy.size
CMZ = CMz.sum() / CMz.size

X = X - CMX
Y = Y - CMY
Z = Z - CMZ


CMx = CMx - CMX
CMy = CMy - CMY
CMz = CMz - CMZ

inc = 0.3
thresh = 0.1

def MakeImages(X, Y, Z, Xstr, Ystr, Zstr):
    c = 0
    for dz in np.arange(Z.min(), Z.max(), inc):
        # Get the points within a threshold distance of the plane
        nearMask = np.abs(Z - dz) < thresh
        nearX = X[nearMask]
        nearY = Y[nearMask]
        if nearX.size > 20:
            c += 1
            print "Making frame %s%s %d" % (Xstr, Ystr, c)
            # Start plotting
            plt.plot(nearX, nearY, 'k.', lw=0.5)
            plt.title("%s=%f" % (Zstr, dz))
            plt.xlabel("%s" % Xstr)
            plt.ylabel("%s" % Ystr)
            plt.xlim([X.min(), X.max()])
            plt.ylim([Y.min(), Y.max()])
            plt.savefig("pics/shapes_%s%s_%d_%d.jpg"
                        % (Xstr, Ystr, startTime, c))
            plt.close()


# do YZ plane, vary X
MakeImages(Y, Z, X, "Y", "Z", "X")

# do XZ plane, vary Y
MakeImages(X, Z, Y, "X", "Z", "Y")

# do XY plane, vary Z
MakeImages(X, Y, Z, "X", "Y", "Z")

#!/usr/bin/ipython

import matplotlib

matplotlib.use('Agg')


import sys, os, argparse

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


desc="""
Creates snapshots of the movement  of the center of mass of the system of cells.
"""
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("trajPath", nargs=1, help="Path to the trajectory file.")

args = parser.parse_args()

nAtoms = 0
msg = 0
CoMx = 0.0
CoMy = 0.0
CoMz = 0.0

CoM = []
line = 0

CoMxt = []
CoMyt = []
CoMzt = []
c = 0

trajPath = os.path.abspath(args.trajPath[0])

storPath, trajFileName = os.path.split(trajPath)
trajFileName = os.path.splitext(trajFileName)[0]
storPath += "/" + trajFileName + '/motion/'

print "Saving to %s" % storPath

try:
    os.makedirs(storPath)
except:
    pass

with open(trajPath, "r") as trajFile:
    line = trajFile.readline()
    while(line != ""):
        line = line.strip()
        nAtoms = int(line)
        step = trajFile.readline().strip()[6:]

        print "Processing %s ..." % step

        for atom in xrange(nAtoms):
            line = trajFile.readline()
            line = line.strip()
            line = line.split(',  ');
            CoMx += float(line[0])
            CoMy += float(line[1])
            CoMz += float(line[2])

        CoMx = CoMx / nAtoms
        CoMy = CoMy / nAtoms
        CoMz = CoMz / nAtoms
        CoMxt.append(CoMx)
        CoMyt.append(CoMy)
        CoMzt.append(CoMz)

        CoMx = 0.0
        CoMy = 0.0
        CoMz = 0.0

        line = trajFile.readline()


plt.subplot(2, 2, 1)
plt.plot(CoMxt, CoMyt, '.')
plt.xlabel("X")
plt.ylabel("Y")

plt.subplot(2, 2, 2)
plt.plot(CoMxt, CoMzt, '.')
plt.xlabel("X")
plt.ylabel("Z")

plt.subplot(2, 2, 3)
plt.plot(CoMyt, CoMzt, '.')
plt.xlabel("Y")
plt.ylabel("Z")

plt.tight_layout()
plt.savefig(storPath + "COM.png")
plt.clf()

CoMxt = np.array(CoMxt)
CoMyt = np.array(CoMyt)
CoMzt = np.array(CoMzt)


fig = plt.figure()
print "Now generating per step data..."

for i in range(CoMxt.size):

    print "done %d of %d" %(i+1, CoMxt.size)

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.scatter(CoMxt[i], CoMyt[i], CoMzt[i], '.')

    ax.set_xlim3d([np.floor(CoMxt.min()), np.ceil(CoMxt.max())])
    ax.set_ylim3d([np.floor(CoMyt.min()), np.ceil(CoMyt.max())])
    ax.set_zlim([CoMzt.min(), CoMzt.max()])
    plt.savefig(storPath + "3d%d.png" % i)
    plt.clf()

    plt.plot(CoMxt[i], CoMyt[i], '.')
    plt.ylim([CoMyt.min(), CoMyt.max()])
    plt.xlim([CoMxt.min(), CoMxt.max()])
    plt.savefig(storPath + "XY%d.png" % i)
    plt.clf()

    plt.plot(CoMxt[i], CoMzt[i], '.')
    plt.xlim([CoMxt.min(), CoMxt.max()])
    plt.ylim([CoMzt.min(), CoMzt.max()])
    plt.savefig(storPath + "XZ%d.png" % i)
    plt.clf()

    plt.plot(CoMyt[i], CoMzt[i], '.')
    plt.xlim([CoMyt.min(), CoMyt.max()])
    plt.ylim([CoMzt.min(), CoMzt.max()])
    plt.savefig(storPath + "YZ%d.png" % i)
    plt.clf()

#plt.plot(CoMx, CoMz, '.')
#plt.savefig("testing/ZX%d.png" % c)
#plt.clf()
#plt.plot(CoMy, CoMz, '.')
#plt.savefig("testing/ZY%d.png" % c)
#plt.clf()

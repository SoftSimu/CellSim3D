#!/usr/bin/ipython

import matplotlib
matplotlib.use('Agg')


import sys

import matplotlib.pyplot as plt
import numpy as np

trajFileName = sys.argv[1]

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

with open(trajFileName, "r") as trajFile:
    line = trajFile.readline()
    while(line != ""):
        line = line.strip()
        nAtoms = int(line)
        msg = trajFile.readline().strip()

        print "Processing %s ..." % msg

        for atom in xrange(nAtoms):
            line = trajFile.readline()
            line = line.strip()
            line = line.split(',  ');
            CoMx += float(line[0])
            CoMy += float(line[1])
            CoMz += float(line[2])

        CoMxt.append(CoMx/nAtoms)
        CoMyt.append(CoMy/nAtoms)
        CoMzt.append(CoMz/nAtoms)
        
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
plt.savefig("COM.png")



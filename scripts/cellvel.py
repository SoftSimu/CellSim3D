#!/usr/bin/ipython

import sys 
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
import numpy as np

trajFileName = sys.argv[1]
cellNo = int(sys.argv[2])

nAtoms = 0
nCells = 0
cmx = 0.0
cmy = 0.0
cmz = 0.0
step = 0
foundStep = 0
X = []
Y = []
Z = []
isFirst = True
temp = 0
dt = 0

with open(trajFileName, 'r') as trajFile:
    line = trajFile.readline()
    while (line != ""):
        line = line.strip()
        nAtoms = int(line)
        temp = step 
        step = int(trajFile.readline().strip()[6:])
        nCells = nAtoms/192
        if (nCells < cellNo):
            for i in xrange(nAtoms):
                trajFile.readline()
        else:
            if (isFirst):
                isFirst = False
                print "Cell was born in step %d" % step
                foundStep = step
                
            for i in xrange( (cellNo - 1) * 192): # skip all preceding cells
                trajFile.readline()
                
            for i in xrange(192): # read the cell of interest
                line = trajFile.readline().strip()
                line = line.split(",  ")
                cmx += float(line[0])
                cmy += float(line[1])
                cmz += float(line[2])
                
            cmx /= 192.0
            cmy /= 192.0
            cmz /= 192.0

            #print cmx, ", ", cmy, ", ", cmz

            X.append(cmx)
            Y.append(cmy)
            Z.append(cmz)

            for i in xrange( (nCells - cellNo)*192 ): # skip all the remaining cells
                trajFile.readline()
                
        line = trajFile.readline()


if (isFirst):
    print "The system is not that big! Only have %d cells." % nCells
    sys.exit()



plt.subplot(2, 2, 1)
plt.plot(X, Y, "k.")
plt.xlabel("X")
plt.ylabel("Y")

plt.subplot(2, 2, 2)
plt.plot(X, Z, "k.")
plt.xlabel("X")
plt.ylabel("Z")

plt.subplot(2, 2, 3)
plt.plot(Y, Z, "k.")
plt.xlabel("Y")
plt.ylabel("Z")


plt.suptitle("Movement of cell no. %d" % cellNo)
plt.tight_layout()
plt.savefig("cell_%d.png" % cellNo)
plt.close()

# Get avg cell velocity
v = []
dt = step - temp
for i in xrange(len(X) - 1):
    dx = X[i+1] - X[i]
    dy = Y[i+1] - Y[i]
    dz = Z[i+1] - Z[i]
    v.append(np.sqrt(dx*dx + dy*dy + dz*dz)/dt)

plt.plot(range(foundStep, step, dt), v, 'k.')
plt.xlabel("t")
plt.ylabel("V")
plt.savefig("cell_%d_vel.png" % cellNo)
plt.close()

    

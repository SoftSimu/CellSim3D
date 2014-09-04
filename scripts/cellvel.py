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
cellX = []
cellY = []
cellZ = []
isFirst = True
temp = 0
dt = 0
sysCMx = 0.0
sysCMy = 0.0
sysCMz = 0.0

sysX = []
sysY = []
sysZ = []

with open(trajFileName, 'r') as trajFile:
    line = trajFile.readline()
    while (line != ""):
        line = line.strip()
        nAtoms = int(line)
        temp = step 
        step = int(trajFile.readline().strip()[6:])
        nCells = nAtoms/192
        if (nCells < cellNo):
            # skip the current time step if it does not contain
            # the cell of interest 
            for i in xrange(nAtoms):
                trajFile.readline()
        else:
            if (isFirst):
                isFirst = False
                print "Cell was near step %d" % step
                foundStep = step

            # only contribute to the system center of mass for all the
            # preceeding cells
            for i in xrange( (cellNo - 1) * 192): 
                line = trajFile.readline().strip()
                line = line.split(",  ")
                sysCMx += float(line[0])
                sysCMy += float(line[1])
                sysCMz += float(line[2])
                
            for i in xrange(192): # read the cell of interest
                line = trajFile.readline().strip()
                line = line.split(",  ")
                cmx += float(line[0])
                cmy += float(line[1])
                cmz += float(line[2])
                sysCMx += float(line[0])
                sysCMy += float(line[1])
                sysCMz += float(line[2])
                
            cmx /= 180.0
            cmy /= 180.0
            cmz /= 180.0

            

            cellX.append(cmx)
            cellY.append(cmy)
            cellZ.append(cmz)
                        
            cmx = 0.0
            cmy = 0.0
            cmz = 0.0
            
            # Now get the contribution to system center of mass of the remaining
            # cells
            for i in xrange( (nCells - cellNo)*192 ):
                line = trajFile.readline().strip()
                line = line.split(",  ")
                sysCMx += float(line[0])
                sysCMy += float(line[1])
                sysCMz += float(line[2])

            sysCMx /= nCells * 180
            sysCMy /= nCells * 180 
            sysCMz /= nCells * 180

#            print sysCMx, sysCMy, sysCMz

            sysX.append(sysCMx)
            sysY.append(sysCMy)
            sysZ.append(sysCMz)

            sysCMx = 0.0
            sysCMy = 0.0
            sysCMz = 0.0 
            
        line = trajFile.readline()


if (isFirst):
    print "The system is not that big! Only have %d cells." % nCells
    sys.exit()


cellX = np.array(cellX)
cellY = np.array(cellY)
cellZ = np.array(cellZ)

sysX = np.array(sysX)
sysY = np.array(sysY)
sysZ = np.array(sysZ)


cellX = cellX - sysX
cellY = cellY - sysY
cellZ = cellZ - sysZ
    
plt.subplot(2, 2, 1)
plt.plot(cellX, cellY, "k.")
plt.xlabel("cellX")
plt.ylabel("cellY")

plt.subplot(2, 2, 2)
plt.plot(cellX, cellZ, "k.")
plt.xlabel("cellX")
plt.ylabel("cellZ")

plt.subplot(2, 2, 3)
plt.plot(cellY, cellZ, "k.")
plt.xlabel("cellY")
plt.ylabel("cellZ")


plt.suptitle("Movement of cell no. %d" % cellNo)
plt.tight_layout()
plt.savefig("cell_%d.png" % cellNo)
plt.close()

# Get avg cell velocity
v = []
dt = step - temp
for i in xrange(len(cellX) - 1):
    dx = cellX[i+1] - cellX[i]
    dy = cellY[i+1] - cellY[i]
    dz = cellZ[i+1] - cellZ[i]
    v.append(np.sqrt(dx*dx + dy*dy + dz*dz)/dt)

plt.plot(range(foundStep, step, dt), v, 'k.')
plt.xlabel("t")
plt.ylabel("V")
plt.savefig("cell_%d_vel.png" % cellNo)
plt.close()

    

#!/usr/bin/ipython
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 3:
    print "Usage: pop.py {Path to trajectory.xyz} {Path to output}"
    print "Don't put extension in output path."
    print "e.g."
    print "pop.py /path/to/trajec.xyz /path/to/image"
    sys.exit(0)


pop=[]

with open(sys.argv[1], 'r') as file:
    line = file.readline()
    while ( line != ""):
        #print line
        nAtoms = int(line.strip())
        nCells = nAtoms/192
        step = file.readline().strip()[6:]
        print "nCells = %d Step = %s" % (nCells, step)
        pop.append(nCells)
        for n in xrange(nAtoms): # skip coords + Step line
            line = file.readline()
        line = file.readline()


x=np.array(range(len(pop)))
y=np.array(pop);

plt.plot(x, y, 'k.')
plt.ylabel('Population')
plt.xlabel('t')
plt.savefig(sys.argv[2] + '.png')

#!/usr/bin/ipython
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

files = sys.argv[2:]

pop=[]
P = [65, 70, 75, 80, 85]
c = 0
hs=[]

for filea in files:
    with open(filea, 'r') as file:
        while True:
            #print line
            line = file.readline()

            if line == "":
                break

            nAtoms = int(line.strip())
            nCells = nAtoms/192
            pop.append(nCells)
            line = file.readline()
            for n in xrange(nAtoms): # skip coords + Step line
                line = file.readline()


    x=np.array(range(len(pop)))
    y=np.array(pop);
    h, = plt.plot(x,y, '.', label="P_max = %d" % P[c])
    hs.append(h)
    c+=1
    pop = []


plt.ylabel('Population')
plt.xlabel('t')
plt.legend(loc=2, prop={'size':10})


plt.savefig(sys.argv[1] + '.png')

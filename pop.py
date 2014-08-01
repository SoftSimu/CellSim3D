#!/usr/bin/ipython
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pop=[]

with open('pop.dat', 'w') as popFile:

    with open('traj.xyz', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not "C" in line and not ":" in line:
                pop.append(int(line.strip())/192)

x=np.array(range(len(pop)))
y=np.array(pop); 

plt.plot(x, y, 'k.')
plt.ylabel('Population')
plt.xlabel('t')
plt.savefig('pop.png')

                
            

#!/usr/bin/env python
import sys, os, argparse

import matplotlib.pyplot as plt
import numpy as np

desc="""
Plots the kinetic energy of the system
"""
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("trajPath", help="Path to the trajectory file.")

args = parser.parse_args()
trajPath = os.path.abspath(args.trajPath)
kinEnergy = []
energyStep = 0
with open(trajPath, "r") as trajFile:
    line = trajFile.readline()
    while(line != ""):
        nAtoms=int(line)
        step = int(trajFile.readline().strip()[6:])
        print step
        nCells = nAtoms/192
        energyStep = 0

        for c in xrange(nCells):
            for a in xrange(192):
                line = trajFile.readline()
                line = line.strip().split(',  ')
                Vx = float(line[0])
                Vy = float(line[1])
                Vz = float(line[2])
                energyStep += Vx**2 + Vy**2 + Vz**2

        kinEnergy.append([energyStep])
        line = trajFile.readline()

plt.plot(kinEnergy)
plt.savefig('Kinetic_Energy.png')
plt.close()

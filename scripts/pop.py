#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, argparse


desc= """
Plots the population of a simulation versus time.
"""

parser = argparse.ArgumentParser(description=desc)

parser.add_argument("trajPaths", nargs='+', help="Path to trajectory file(s)")

files = sys.argv[2:]

args = parser.parse_args()

trajPaths = [os.path.abspath(p) for p in args.trajPaths]

GetStorePath = lambda path: os.path.splitext(path)[0]

def PlotPop(trajPath):
    pop=[]
    c = 0
    q, name = os.path.split(trajPath)
    print "Processing %s" % name
    path = GetStorePath(trajPath)

    try:
        os.makedirs(path)
    except:
        pass

    path += "/pop.png"

    with open(trajPath, 'r') as trajFile:
        while True:
            #print line
            line = trajFile.readline()

            if line == "":
                break

            nAtoms = int(line.strip())
            nCells = nAtoms/192
            pop.append(nCells)
            line = trajFile.readline().strip()
            print "on %s" % line

            for n in xrange(nAtoms): # skip coords + Step line
                line = trajFile.readline()


    x=np.array(range(len(pop)))
    y=np.array(pop);

    plt.plot(x,y, '.')
    plt.ylabel('Population')
    plt.xlabel('t')
    plt.savefig(path)

for p in trajPaths:
    PlotPop(p)

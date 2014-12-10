#!/usr/bin/ipython
import matplotlib
matplotlib.use('Agg')


import sys

import matplotlib.pyplot as plt
import numpy as np

trajFileName = sys.argv[1]
cellNo = int(sys.argv[2]) - 1
nAtoms = 0
lineNo = 0

CoMx = 0.0
CoMy = 0.0
CoMz = 0.0
step  = 0

with open(trajFileName, "r") as trajFile:
    line = trajFile.readline()

    while (line != ""):
        line = line.strip
        nAtoms = int(line)
        line = trajFile.readline().strip()
        step = int(line[6:])
        print "Processing step no.: %d" % step

        if nAtoms < cellNo * 192:
            for i in xrange(nAtoms):
                next(trajFile)
        else:
            for i in xrange(cellNo*192):
                next(trajFile)

            for i in xrange(192):

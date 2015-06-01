#!/usr/bin/env python

#This script measures 3D packing. It is unclear how useful this measurement will
#be. But I think it is worth seeing what we get.


import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.qhull import QhullError as voroError
import os, sys, argparse

desc="""
This script will measure the polygonal packing of a 3D system of cells with
Voronoi Tesselation.
"""

parser = argparse.ArgumentParser(description=desc)

parser.add_argument("--traj", nargs="+",
                    help="Absolute or relative paths")
parser.add_argument("--endat", type=int,
                    help="time step to stop processing at")
args = parser.parse_args()

trajPaths = [os.path.abspath(p) for p in args.traj]
storePaths = [os.path.split(p)[0] for p in trajPaths]

if args.endat is None:
    endat = 1e100
else:
    endat = args.endat

def analyze(filePath, storePath, name=None):
    fileName = os.path.splitext(os.path.split(filePath)[1])[0]
    print "Processing %s" % filePath
    if name == None:
        storePath  += '/' + fileName + '/' + '3DPacking/'
    else:
        storePath +=  '/' + fileName + '/' + name + '/'

    try:
        os.makedirs(storePath)
    except:
        pass

    polyCountEvo = {}
    print "Generating cell CoMs..."
    # Now read trajectory file and get the centre of masses of
    # all cells at every time step
    with open(filePath, 'r') as trajFile:
        while True:
            line = trajFile.readline().strip()

            if line == '':
                break

            nCells = int(line)/192
            line = trajFile.readline().strip()
            print "Now on %s" % line

            step = int(line[6:])
            CoMs = []
            for i in xrange(nCells):
                CoM = np.zeros(3)
                for j in xrange(192):
                    line = trajFile.readline().strip().split(', ')
                    CoM += np.array([float(line[0]),
                                     float(line[1]),
                                     float(line[2])])
                    print CoM
                    print "------------"
                CoMs.append(CoM/192)

            CoMs = np.array(CoMs)

            # Normalize the system to have origin at centre of mass
            CoMs = CoMs - np.mean(CoMs, axis=0)

            # Now calculate 3D Voronoi Tesselation
            # First, ignore cells on the outside
            dists = np.linalg.norm(CoMs, axis=1)

            nearCoMs = CoMs[(dists/dists.max() <= 0.8)] # Make the 0.8 variable

            try:
                voro = Voronoi(nearCoMs)
            except voroError:
                print "Probably too few cells for meaningful Voroni, skipping"
                continue

            cellList = []
            for region in voro.regions:
                if -1 not in region and len(region) != 0: #regions with -1 have a neighbour at infinity
                    #reg = [n for n in region if n!= -1]
                    cellList.append(region)
                #else:
                    #reg = region
                #cellList.append(reg)


            #Count the number of neighbours for each cell

            #print cellList

            numNeigh = [len(r) for r in cellList]
            #print numNeigh

            # Now lets calculate percentage of cells with different number of
            # neighbours

            numCells = len(cellList)
            inc = 1.0/numCells * 100
            polyDict = {}
            colorDict = {}
            for n in numNeigh:
                if n not in polyDict:
                    polyDict.update({n:inc})
                    colorDict.update({n:np.random.uniform(0,1, 1)})
                else:
                    polyDict[n] += inc

            # Now plot some bar charts yo
            print "Plotting packing bar charts..."
            pos = np.array(polyDict.keys())
            percents = [polyDict[k] for k in polyDict]
            #plt.bar(pos, percents)
            plt.plot(pos, percents, '.')
            plt.xlabel('Number of neigbours')
            plt.ylabel('Percent')
            plt.savefig(storePath + '%d_barcharts.png' % step)
            plt.close()

            c0NeighInds = cellList[0]
            print c0NeighInds
            c0NeighCoords = [nearCoMs[i] for i in c0NeighInds]
            print c0NeighCoords


for i in xrange(len(trajPaths)):
    analyze(trajPaths[i], storePaths[i])

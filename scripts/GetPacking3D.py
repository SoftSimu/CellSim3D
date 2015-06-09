#!/usr/bin/env python

#This script measures 3D packing. It is unclear how useful this measurement will
#be. But I think it is worth seeing what we get.


import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, delaunay_plot_2d
from scipy.spatial.qhull import QhullError
import os, sys, argparse
from mpl_toolkits.mplot3d import Axes3D

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
                    line = trajFile.readline().strip().split(',  ')
                    CoM += np.array([float(p) for p in line])
                CoMs.append(CoM/192)

            CoMs = np.array(CoMs)

            # Normalize the system to have origin at centre of mass
            CoMs = CoMs - np.mean(CoMs, axis=0)

            # Now calculate 3D Voronoi Tesselation
            # First, ignore cells on the outside
            dists = np.linalg.norm(CoMs, axis=1)

            nearCoMs = CoMs[(dists/dists.max() <= 0.8)] # Make the 0.8 variable

            try:
                delaunay = Delaunay(CoMs)
            except QhullError:
                print "Probably too few cells for meaningful triangulation, skipping"
                continue

            cellList = []

            # If you are wondering why below is "weirdTuple":
            # The way that scipy organizes nearest neighbours in
            # Delaunay.vertex_neighbor_vertices is fucked up.
            # See http://goo.gl/ZeBdgj to understand what
            # we need to do to get nearest neighbour indices
            weirdTuple =  delaunay.vertex_neighbor_vertices

            neighborRange, neighborInds = weirdTuple


            #Count the number of neighbours for each cell
            numNeigh = []
            for cellInd in xrange(nCells):
                #numNeigh.append(len(neighborInds[neighborRange[cellInd]:
                                                 #neighborRange[cellInd+1]]))
                neighList = neighborInds[neighborRange[cellInd]:neighborRange[cellInd+1]]
                neighCoords = np.array([CoMs[i] for i in neighList])
                neighVecs = neighCoords - CoMs[cellInd]
                neighDists = np.linalg.norm(neighVecs, axis=1)
                closeNeighMask = neighDists<1*np.ceil(neighDists.min())
                neighCoords = neighCoords[closeNeighMask]

                numNeigh.append(closeNeighMask.sum())
            # Yeah, I know!

            # Now lets calculate percentage of cells with different number of
            # neighbours

            inc = 1.0/nCells * 100
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
            fig = plt.figure()
            SampleNeigh = neighborInds[neighborRange[0]:neighborRange[1]]
            neighCoords = np.array([CoMs[i] for i in SampleNeigh])
            neighDists = np.linalg.norm(neighCoords - CoMs[0], axis=1)
            #print neighDists.max(), np.mean(neighDists), neighDists.min()
            closeNeighMask = neighDists < 2*np.ceil(neighDists.min())
            neighCoords = neighCoords[closeNeighMask]
            ax = fig.add_subplot(1,1,1, projection='3d')
            ax.scatter3D(neighCoords[:, 0], neighCoords[:, 1], neighCoords[:,2], c=u'red')
            ax.scatter3D(CoMs[0][0], CoMs[0][1], CoMs[0][2], c=u'blue')
            plt.savefig(storePath + '%d_del.png' % step)
            plt.close()


for i in xrange(len(trajPaths)):
    analyze(trajPaths[i], storePaths[i])

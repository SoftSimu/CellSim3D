#!/usr/bin/env python
# This script uses a lot of python magic
# Please ask if there are is any confusion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import os, sys, argparse

desc="""
A script to compare the hexagon percentages in a number of trajectories.
"""

parser = argparse.ArgumentParser(description=desc)

parser.add_argument("--traj", nargs="+",
                    help="Absolute or relative paths")
parser.add_argument("--endat", type=int,
                    help="time step to stop processing at")

args = parser.parse_args()

if args.endat is None:
    endat = 1e100
else:
    endat = args.endat

trajList = [os.path.abspath(p) for p in args.traj]

try:
    os.makedirs("./graphs")
except:
    0

barFig = []
n = len(args.traj)
width = 0.25

gammas = ['0', '5', '10', '15', '20', '25']

def analyze(filePath, cm, barAx, c, width, ind, polyTrendAx, gamma):
    voro = []
    polyCountEvo = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    colorDict = {3:'k', 4:'w', 5:'g', 6:'r', 7:'c', 8:'m', 9:'y'}

    ncLocDict = {0.1:[], 0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    rList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    path, fileName = os.path.split(filePath)
    fileName = path.split('/')[-1]
    with open(filePath, 'r') as comFile:
        timeStep = 0
        U_prev = []
        while True:
            xLine = comFile.readline().strip()
            if xLine == "" or timeStep >= endat:
                break
            timeStep +=1
            #print "Processing time step %d of %s" % (timeStep, fileName)
            yLine = comFile.readline().strip()
            nCoM = 0
            X = np.array([float(f) for f in xLine.split(' ')])
            Y = np.array([float(f) for f in yLine.split(' ')])

            if (X.size != Y.size):
                raise('X.size != Y.size')

            U = np.array([X, Y]).T
            CoM = U.mean(axis=0)

            # Normalize data to CoM at origin
            U -= CoM

            # Calculate distance of each cell from CoM (origin)
            # ie get magnitude of each pos vector
            dists = np.linalg.norm(U, axis=1) # this needs latest numpy version

            # get max distance from CoM
            maxDist = dists.max()
            if (0 < maxDist < 0.001):
                raise("maxDist too low to continue")

            rel2MaxDists = dists/maxDist

            # Only use cells that are within some fraction of the maximum system size
            # This uses boollean indexing, please ask if confused

            U_near = U[rel2MaxDists < 0.8]
            nCells = U_near.shape[0]
            voro = Voronoi(U_near)
            cellList = []
            for region in voro.regions:
                if -1 not in region and len(region) != 0: # region containing index -1 means infinite region
                    cellList.append(region)


            # Count polygon types
            polyCount = {}
            polyTypes = [len(cell) for cell in cellList]
            nCellsCounted = len(cellList)
            increment = 1.0/nCellsCounted
            for poly in polyTypes:
                if not poly in polyCount:
                    polyCount.update({poly:increment})
                else:
                    polyCount[poly]+=increment

            for pt in polyCountEvo:
                if pt not in polyCount:
                    polyCountEvo[pt].append(0)
                else:
                    polyCountEvo[pt].append(polyCount[pt])

            # Get all new cells and process where they are
            if U_prev == [] or U_prev.size == U.size:
                U_prev = np.copy(U)
                for r in rList:
                    ncLocDict[r].append(0)
            else:
                # get all cells not in previous
                newCells = U[-1*(U.shape[0] - U_prev.shape[0]) :]
                inc = 1.0/U.shape[0]
                newCellDistFracs = np.linalg.norm(newCells, axis=1)/maxDist
                for r in rList:
                    m1 = newCellDistFracs >= r - 0.05
                    m2 = newCellDistFracs <= r + 0.05
                    frac = (m1*m2).sum()*inc
                    ncLocDict[r].append(frac)



    #print "Plotting %s polygon count..." % fileName
    #
    #timeRange=np.arange(len(polyCountEvo[6]))*0.2
    #
    #for pt in polyCountEvo:
    #    if sum(polyCountEvo[pt]) > 0:
    #        plt.plot(timeRange, polyCountEvo[pt], label='%s sided' % pt)
    #
    #plt.legend(loc='upper left')
    #plt.xlabel('num time steps')
    #plt.ylabel('Cell Fraction')
    #plt.savefig('graphs/%s_polyevo.pdf' % fileName)
    #plt.close()


    print "Charting %s bars..." % fileName
    # Generate bar charts
    rects = []
    meanFracs = [np.mean(polyCountEvo[key][:]) for key in ind]
    err = [np.std(polyCountEvo[key][100:]) for key in ind]
    cl = np.array([c*1.0/n])
    pos = ind*(n+2)*width + c*width - ((n+2)/2.0 * width)

    barAx.bar(pos, meanFracs, width, yerr=err,
              label='$\gamma_{ext} = %s$' % gamma, ecolor='k',
              color=cm(cl)[0])

    timeRange = xrange(len(polyCountEvo[6]))
    polyTrendAx.plot(timeRange, polyCountEvo[6],
                     color=cm(cl)[0])

barFig, barAx = plt.subplots()
polyTrendAx = plt.axes([0.2, 0.65, 0.4, 0.2])
cm = plt.get_cmap('rainbow')


bargap = 0.05
c = 0
ind = np.arange(4, 9)
for i in xrange(len(trajList)):
    analyze(trajList[i], cm , barAx, c, width, ind, polyTrendAx, gammas[i])
    c +=1


# plot experimental data
print "Plotting experimental data..."
pos = ind*(n+2)*width + c*width - ((n+2)/2.0 * width)

# Reference for experimental results: 10.1038/nature05014
expFracs = [0.028, 0.272, 0.458, 0.203, 0.015]
expErrs = [0.016, 0.018, 0.024, 0.025, 0]
#plot experimental data
barAx.bar(pos, expFracs, width, yerr=expErrs, label='Drosophila',
          ecolor = 'k', color = 'r')

pos = ind*(n+2)*width + (c+1)*width - ((n+2)/2.0 * width)

#barAx.bar(pos, [0.5 for j in xrange(len(ind))], width, color='k')

barAx.set_ylabel('Percent')
barAx.set_xlabel('Number of neighbours')
barAx.set_ylim((0,1))


barAx.set_xticks(ind*(n+2)*width - (width/2))
barAx.set_xticklabels(ind)
barAx.legend()

polyTrendAx.set_xlabel('Time')
polyTrendAx.set_ylabel('Hexagons')

plt.savefig('manybars.eps')

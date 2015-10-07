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
Anna's main analysis script. Produces many graphs
"""

parser = argparse.ArgumentParser(description=desc)

parser.add_argument("--traj", nargs="+",
                    help="Absolute or relative paths")
parser.add_argument("--endat", type=int,
                    help="time step to stop processing at")

args = parser.parse_args()

trajPaths = [os.path.abspath(p) for p in args.traj]
storePaths = [ os.path.split(p)[0] for p in trajPaths]
#a = [os.path.splitext(p) for p in trajPaths]
#b = [list(t) for t in zip(*a)]

#storePaths = b[0]

print storePaths
#sys.exit(12984)

for storePath in storePaths:
    try:
        os.makedirs(storePath + '/' + 'flat')
    except:
        pass

cm = plt.get_cmap('gist_rainbow')
c = 0

if args.endat is None:
    endat = 1e100
else:
    endat = args.endat


def analyze(filePath, storePath, name = None):
    fileName = os.path.split(storePath)[1]
    print filePath
    if name == None:
        storePath  += '/' + 'flat/'
    else:
        storePath +=  '/' + name + '/'

    voro = []
    #polyCountEvo = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[]}
    polyCountEvo = {4:[], 5:[], 6:[], 7:[], 8:[]}
    colorDict = {3:'k', 4:'w', 5:'g', 6:'r', 7:'c', 8:'m', 9:'y', 10:'k', 11:'k', 12:'k', 13:'k', 14:'k', 15:'k', 16:'k', 17:'k'}

    #ncLocDict = {0.1:[], 0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    #rList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ncLocDict = {0.1:[], 0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[]}
    rList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    with open(filePath, 'r') as comFile:
        timeStep = 0
        U_prev = []

        while True:
            xLine = comFile.readline()
            xLine = xLine.strip()
            if xLine == "" or timeStep >= endat:
                break
            timeStep +=1
            print "Processing time step %d of %s" % (timeStep, fileName)
            yLine = comFile.readline().strip()
            X = np.array([float(f) for f in xLine.split(' ')])
            Y = np.array([float(f) for f in yLine.split(' ')])

            if (X.size != Y.size):
                raise('X.size != Y.size')

            U = np.array([X, Y]).T
            CoM = U.mean(axis=0)

            # Normalize data to CoM at origin
            U -= CoM

            # Calculate distance of each cell from CoM (origin)
            # ie get magnitude of each position vector
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
                    #print region


            # Count polygon types
            polyCount = {}
            polyTypes = [len(cell) for cell in cellList]
            nCellsCounted = len(cellList)
            increment = 1.0/nCellsCounted
            for poly in polyTypes:
                if not poly in polyCount:
                    polyCount.update({poly:increment}) # test comment
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
                U = []
                for r in rList:
                    ncLocDict[r].append(0)
            else:
                # get all cells not in previous
                newCells = U[-1*(U.shape[0] - U_prev.shape[0]) :]
                print U_prev.shape, newCells.shape
                inc = 1.0/newCells.shape[0]
                newCellDistFracs = np.linalg.norm(newCells, axis=1)/maxDist
                for r in rList:
                    m1 = newCellDistFracs >= r - 0.05
                    m2 = newCellDistFracs <= r + 0.05
                    frac = (m1*m2).sum()*inc
                    ncLocDict[r].append(frac)
                U_prev=np.copy(U)
                U = []



    print "Plotting %s polygon count..." % fileName

    timeRange=np.arange(len(polyCountEvo[6]))

    for pt in xrange(4,9):
        if sum(polyCountEvo[pt]) > 0:
            plt.plot(timeRange, [100*f for f in polyCountEvo[pt]], lw = 2, label='%s sided' % pt, alpha=0.7)

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Polygon Fraction (%)')
    plt.savefig(storePath + '%d_polyevo.png' % timeStep)
    plt.close()


    print "Charting %s bars..." % fileName
    # Generate bar charts
    plt.bar(np.array(polyCountEvo.keys()) - 0.5,
            [np.mean(polyCountEvo[key][-20:]) for key in polyCountEvo],
            yerr = [np.std(polyCountEvo[key][-20:]) for key in polyCountEvo],
            ecolor='k', color='grey')
    yerr=[np.std(polyCountEvo[key][-100:]) for key in polyCountEvo]
    plt.xlabel('Polygon type')
    plt.ylabel('fraction')
    plt.ylim((0,1))
    plt.savefig(storePath + '%d_barcharts.png' % timeStep)
    plt.close()


    print "Voronoi-ing %s..." % fileName
    # Make voronoi of last time step
    for region in voro.regions:
        if not -1 in region and len(region)!=0:
            polygon=[voro.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=colorDict[len(region)])

    # plot the ridges so that we can see individual cells
    for simplex in voro.ridge_vertices:
        simplex=np.asarray(simplex)
        if np.all(simplex >= 0):
            plt.plot(voro.vertices[simplex,0], voro.vertices[simplex, 1], 'k-')


    # Readjust the graph to center on our cells
    ptp_bound = voro.points.ptp(axis=0)

    plt.xlim(voro.points[:, 0].min() - 0.1*ptp_bound[0],
             voro.points[:, 0].max() + 0.1*ptp_bound[0])
    plt.ylim(voro.points[:, 1].min() - 0.1*ptp_bound[1],
             voro.points[:, 1].max() + 0.1*ptp_bound[1])

    plt.savefig(storePath + '%d_voronoi.png' % (timeStep))
    plt.close()

    print "Mapping %s mitotic cells..." % fileName
    #Plot eh mitotic cells over time and r fraction
    # see http://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
    # and http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    fig=plt.figure()
    #cm = plt.get_cmap('gist_rainbow')
    ax =fig.add_subplot(111)
    #ax.set_color_cycle([cm(1.*i/10) for i in range(10)])
    c = 0
    for r in rList:
        c += 1
        ran = np.random.uniform(0, 1, 1)
        line,=ax.plot(timeRange[-50:], ncLocDict[r][-50:], marker=(4, 0, 0),
                      label="%.1f Rmax" % r)

    plt.ylabel("Fraction")
    plt.xlabel("Time")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol = 3,
               fancybox=True, shadow=True)
    plt.savefig(storePath + '%d_MitoticFractVSR.png' % timeStep)
    plt.close()

for i in xrange(len(trajPaths)):
    analyze(trajPaths[i], storePaths[i])

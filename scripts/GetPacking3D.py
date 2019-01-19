#!/usr/bin/env python3

#This script measures 3D packing. It is unclear how useful this measurement will
#be. But I think it is worth seeing what we get.


import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
import os, sys, argparse
import csv
from scipy.optimize import curve_fit
from scipy.stats import lognorm

sys.path.append("/home/pranav/dev/celldiv/scripts")
import celldiv

from scipy.spatial.qhull import QhullError

desc="""
This script will measure the polygonal packing of a 3D system of cells with
Voronoi Tesselation.
"""

parser = argparse.ArgumentParser(description=desc)

parser.add_argument("--traj", nargs="+",
                    help="One or more absolute or relative paths to trajectories")
parser.add_argument("--endat", type=int,
                    help="time step to stop processing at")
parser.add_argument("-k", "--skip", type=int, required=False,
                    help="Frame skip rate")

parser.add_argument("-f", "--flat", type=bool,
                    help="Flag that enables analysis of epithelia",
                    default=False, required=False)
args = parser.parse_args()

trajPaths = [os.path.abspath(p) for p in args.traj]
storePaths = [os.path.split(p)[0] for p in trajPaths]

if args.endat is None:
    endat = 1e100
else:
    endat = args.endat

aRate = args.skip

if args.skip is None:
    aRate = 1

def measurePacking(filePath, storePath, name=None):
    fileName = os.path.splitext(os.path.split(filePath)[1])[0]
    print("Processing %s" % filePath)
    if name == None:
        storePath  += '/' + fileName + '/' + '3DPacking/'
    else:
        storePath +=  '/' + fileName + '/' + name + '/'

    try:
        os.makedirs(storePath)
    except:
        pass

    dataPath = storePath + "data_best.dat"
    # Now read trajectory file and get the centre of masses of
    # all cells at every time step
    if os.path.isfile(dataPath):
        if os.path.getmtime(dataPath) > os.path.getmtime(filePath):
            print("{0} already processed.".format(filePath))
            print("Skipping")
            return dataPath

    print("Generating cell CoMs...")

    with celldiv.TrajHandle(filePath) as s,\
         open(storePath + "data.dat", 'w') as f:

        while True:
            try:
                frame = s.ReadFrame(inc = aRate)
            except:
                break
            step = s.step
            if s.fatalError:
                break

            nCells = len(frame)

            frame = [cell[:180] for cell in frame]

            if nCells > 3:
                CoMs = np.array([np.mean(cell, axis=0) for cell in frame])
                sysCoM = np.mean(CoMs, axis=0)
                CoMs -= sysCoM

                if args.flat == True:
                    CoMs = CoMs[:, 0:2]

                print("Doing Del of", CoMs.shape[0], "cells at step", step)
                try:
                    d = Delaunay(CoMs)
                    indices, indptr = d.vertex_neighbor_vertices
                    count = []
                    for p in range(nCells):
                        n = len(indptr[indices[p]:indices[p+1]])
                        if n > len(count):
                            for i in range(len(count), n):
                                count.append(0)
                        count[n-1] += 1/nCells

                    f.write(", ".join([str(c) for c in count]))
                    f.write("\n")

                except QhullError:
                    print("delaunay failed, skipping")
                    pass

    return dataPath

for i in range(len(trajPaths)):
    measurePacking(trajPaths[i], storePaths[i])

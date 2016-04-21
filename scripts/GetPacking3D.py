#!/usr/bin/env python3

#This script measures 3D packing. It is unclear how useful this measurement will
#be. But I think it is worth seeing what we get.


import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, delaunay_plot_2d, Voronoi
from scipy.spatial.qhull import QhullError
import os, sys, argparse
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binom
from scipy.misc import factorial

sys.path.append("/home/pranav/dev/celldiv/scripts")
import celldiv

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
    print("Processing %s" % filePath)
    if name == None:
        storePath  += '/' + fileName + '/' + '3DPacking/'
    else:
        storePath +=  '/' + fileName + '/' + name + '/'

    try:
        os.makedirs(storePath)
    except:
        pass

    print("Generating cell CoMs...")
    # Now read trajectory file and get the centre of masses of
    # all cells at every time step
    with open(filePath, 'r') as trajFile:
        step = 0
        while True:
            line = trajFile.readline().strip()

            if line == '':
                break
            CoMsx = np.array([float(c) for c in line.split(' ')])
            CoMsy = np.array([float(c) for c in trajFile.readline().strip().split(' ')])
            CoMsz = np.array([float(c) for c in trajFile.readline().strip().split(' ')])

            CoMs = np.array([CoMsx, CoMsy, CoMsz]).T

            # Normalize the system to have origin at centre of mass
            CoMs = CoMs - np.mean(CoMs, axis=0)
            nCells = CoMs.shape[0]

            if nCells > 3:
                step += 1
                # print("Doing Vor of", CoMs.shape[0], "cells at step", step)
                # t = Voronoi(CoMs)
                # count = [0]
                # for reg in t.regions:
                #     if len(reg)>0:
                #         r = np.array(reg)
                #         r = r[r>-1]
                #         n = r.shape[0]
                #         if n > len(count):
                #             for i in range(len(count), n):
                #                 count.append(0)

                #         count[n-1] += 1/nCells

                # plt.plot(count)
                # plt.minorticks_on()
                # plt.grid(b = True, which='minor', linestyle='--')
                # plt.grid(b = True, which='major', linestyle='-')
                # plt.savefig(storePath + "voro_step_%d.png" % step)
                # plt.close()

                print("Doing Del of", CoMs.shape[0], "cells at step", step)
                d = Delaunay(CoMs)
                indices, indptr = d.vertex_neighbor_vertices
                count = []


                for p in range(nCells):
                    n = len(indptr[indices[p]:indices[p+1]])
                    if n > len(count):
                        for i in range(len(count), n):
                            count.append(0)
                    count[n-1] += 1/nCells

                plt.plot(count, "-.")
                mu = count.index(max(count))
                N = len(count)
                p = mu/N

                P = np.zeros(N)
                for n in range(N):
                    P[n] = (factorial(N)/(factorial(n)*factorial(N-n))) * (p**n) * ((1-p)**(N-n))


                plt.plot(P, "--")
                plt.minorticks_on()
                # plt.grid(which='minor', linestyle='--')
                # plt.grid(which='major', linestyle='-')
                plt.savefig(storePath + "del_step_%d.png" % step)
                plt.close()
                aa = [str(c) for c in count]
                with open("data.dat", 'w') as f:
                    f.write(" ".join(aa))







for i in range(len(trajPaths)):
    analyze(trajPaths[i], storePaths[i])

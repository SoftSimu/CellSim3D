#!/usr/bin/env python
import sys, os, argparse
import numpy as np

desc = """
Centres a cell system to its center of mass.
"""

parser = argparse.ArgumentParser(description=desc)

parser.add_argument("trajPaths", nargs='+',
                    help="Path (absolute or relative) of any number of trajectory files")
args = parser.parse_args()
trajPaths = [os.path.abspath(p) for p in args.trajPaths]
a = [os.path.splitext(p) for p in trajPaths]
b = [list(t) for t in zip(*a)]

storePaths = b[0]

def Center(trajPath, storePath, name=None):
    if name == None:
        storePath += "_centered.xyz"
    else:
        storePath += name + ".xyz"
    print storePath

    X = []
    Y = []
    Z = []

    CMx = 0
    CMy = 0
    CMz = 0
    with open(trajPath, "r") as inFileHandle:
        with open(storePath, "w") as outFileHandle:
            line = inFileHandle.readline()
            while (line!=""):
                outFileHandle.writelines(line)
                line = line.strip()
                nAtoms = int(line)
                step = int(inFileHandle.readline().strip()[6:])
                outFileHandle.write("Step: %d\n" % step)
                print "Processing %d..." % step

                for i in xrange(nAtoms):
                    line = inFileHandle.readline().strip().split(',  ')
                    X.append(float(line[0]))
                    Y.append(float(line[1]))
                    Z.append(float(line[2]))

                CMx = np.mean(np.array(X))
                CMy = np.mean(np.array(Y))
                CMz = np.mean(np.array(Z))

                X = [t - CMx for t in X]
                Y = [t - CMy for t in Y]
                Z = [t - CMz for t in Z]

                for i in xrange(len(X)):
                    outFileHandle.writelines("%f,  %f,  %f\n" % (X[i], Y[i], Z[i]))

                CMx, CMy, CMz = 0, 0, 0

                X, Y, Z = [], [], []

                line = inFileHandle.readline()


for i in xrange(len(trajPaths)):
    Center(trajPaths[i], storePaths[i] )

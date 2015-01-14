#!/usr/bin/env python

import sys, os, argparse
import numpy as np

desc = """
Flattens the system to be analyzed by scripts that do stuff like Anna's.
In the future, this will be replaced by something more powerfull and better.
For now this will do.
"""
parser = argparse.ArgumentParser(description = desc)

parser.add_argument("trajPaths", nargs="+",
                    help = "Path (absolute or relative) of any number of trajectory files")

args = parser.parse_args()

trajPaths = [os.path.abspath(p) for p in args.trajPaths]
a = [os.path.splitext(p) for p in trajPaths]
b = [list(t) for t in zip(*a)]

storePaths = b[0]

for p in storePaths:
    try:
        os.makedirs(p)
    except:
        pass

def flatten(trajPath, storePath, name=None):
    if name == None:
        storePath += "/flattened.xvg"
    else:
        storePath += "/" + name + ".xvg"

    print "writing to %s" % storePath

    CMxList = []
    CMyList = []
    #CMzList = []

    CMx = 0
    CMy = 0
    #CMz = 0

    with open(trajPath, "r") as inFileHandle:
        with open(storePath, "w") as outFileHandle:
            line = inFileHandle.readline()
            while (line!=""):
                nAtoms = int(line.strip())
                nCells = nAtoms/192
                line = inFileHandle.readline()
                print "Processing %s" % line.strip()
                for i in xrange(nCells):
                    for j in xrange(192):
                        line = inFileHandle.readline().strip().split(',  ')
                        CMx += float(line[0])
                        CMy += float(line[1])

                    CMxList.append(CMx/192)
                    CMyList.append(CMy/192)
                    CMx = 0
                    CMy = 0

                if len(CMxList) != len(CMyList):
                    raise('uh oh hotdog. len(CMxList) != len(CMyList)')

                for n in xrange(len(CMxList)):
                    outFileHandle.write("%f " % CMxList[n])

                outFileHandle.write("\n")

                for n in xrange(len(CMyList)):
                    outFileHandle.write("%f " % CMyList[n])

                outFileHandle.write("\n")

                CMxList = []
                CMyList = []

                line = inFileHandle.readline()

for i in xrange(len(trajPaths)):
    flatten(trajPaths[i], storePaths[i])

#!/usr/bin/env python3

import sys, os, argparse
import numpy as np
import celldiv

desc = """
Flattens the system to be analyzed by scripts that do stuff like Anna's.
In the future, this will be replaced by something more powerfull and better.
For now this will do.
"""
parser = argparse.ArgumentParser(description = desc)

parser.add_argument("trajPaths", nargs="+",
                    help = "Path (absolute or relative) of any number of trajectory files")
parser.add_argument("-skip",
help="If specified, then only every nth frame will be processed. Where n is the\
 value specifed")

parser.add_argument("-f", type=bool, required=False,
                    help="flag for 3D (not flattening)")
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



def flatten(trajPath, storePath, inc, name=None, f=False):
    if name == None:
        storePath += "/flattened.xvg"
    else:
        storePath += "/" + name + ".xvg"

    print("writing to %s" % storePath)

    if os.path.isfile(storePath):

        print("overwriting {0}".format(storePath))


    CMx = 0
    CMy = 0
    CMz = 0

    with open(storePath, "w") as outFileHandle:
        t = celldiv.TrajHandle(trajPath)
        try:
            c = 0
            while c < t.maxFrames:
                frame = t.ReadFrame(inc)
                c += 1
                nCells = len(frame)
                t.nCellsLastFrame
                print("frame ", t.currFrameNum, nCells, " cells")
                CM = np.vstack([np.mean(c, axis=1) for c in frame])

                outFileHandle.write("%s\n" % " ".join([str(a) for a in CM[:, 0]]))
                outFileHandle.write("%s\n" % " ".join([str(a) for a in CM[:, 1]]))
                if f:
                    outFileHandle.write("%s\n" % " ".join([str(a) for a in CM[:, 2]]))




        except celldiv.IncompleteTrajectoryError as e:
            print("broken", e)





for i in range(len(trajPaths)):
    flatten(trajPaths[i], storePaths[i], inc = args.skip, f = args.f)

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
        print("Skipping %s" % storePath)
        return

    CMxList = []
    CMyList = []
    CMzList = []


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
                nCells = int(frame.shape[0]/192)
                t.nCellsLastFrame
                print("frame ", t.currFrameNum, nCells, " cells")
                for n in range(nCells):
                    CMxList.append(str(frame[n*192:(n+1)*192, 0].sum()/192))
                    CMyList.append(str(frame[n*192:(n+1)*192, 1].sum()/192))
                    CMzList.append(str(frame[n*192:(n+1)*192, 2].sum()/192))

                if len(CMxList) != len(CMyList):
                    raise('uh oh hotdog. len(CMxList) != len(CMyList)')

                outFileHandle.write("%s\n" % " ".join(CMxList))
                outFileHandle.write("%s\n" % " ".join(CMyList))
                if f:
                    outFileHandle.write("%s\n" % " ".join(CMzList))

                CMxList = []
                CMyList = []
                CMzList = []



        except celldiv.IncompleteTrajectoryError as e:
            print("broken", e)





for i in range(len(trajPaths)):
    flatten(trajPaths[i], storePaths[i], inc = args.skip, f = args.f)

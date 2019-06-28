import sys
import os
import numpy as np
import celldiv as cd
from tqdm import tqdm
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("--frame-num", "-fn", help="Single frame to analyze",
                type=int, default = 0)
ap.add_argument("--flat", "-f", help="whether trajectory is flat",
                nargs=1, type=bool, default=False)
ap.add_argument("--dom-len", "-dl", help="Domain size for neighbour list generation",
                type=float, default=2.9)
ap.add_argument("traj", help="trajectory file name",
                type=str)

args = ap.parse_args()
trajFile = os.path.abspath(args.traj)
fileName, ext = os.path.splitext(trajFile)
try:
    os.makedirs(fileName)
except:
    pass
outputFile = fileName + "/" + fileName.split("/")[-1] + "_nn.dat"



def GetNeighs(frame):
    frame = [c[:180] for c in frame]
    nCells = len(frame)
    def GenCellNeighs():
        boxMins = np.vstack(frame).min(axis=0)
        boxDims = np.vstack(frame).max(axis=0) - boxMins

        nXDoms, nYDoms, nZDoms = [int(a)+1 for a in boxDims/2.9]


        doms = [[] for _ in range(nXDoms*nYDoms*nZDoms)]

        for cellIdx, cell in enumerate(frame):
            mins = cell.min(axis=0) - boxMins
            maxs = cell.max(axis=0) - boxMins

            xl, yl, zl = [int(k) for k in (mins/args.dom_len)]
            xu, yu, zu = [int(k) for k in (maxs/args.dom_len)]

            for x in range(xl, xu+1):
                for y in range(yl, yu+1):
                    for z in range(zl, zu+1):
                        doms[x + y*nXDoms + z*nXDoms*nYDoms].extend([cellIdx])
                        if x != 0:
                            doms[x-1 + y*nXDoms + z*nXDoms*nYDoms].extend([cellIdx])
                        if x != nXDoms-1:
                            doms[x+1 + y*nXDoms + z*nXDoms*nYDoms].extend([cellIdx])

                        if y != 0:
                            doms[x + (y-1)*nXDoms + z*nXDoms*nYDoms].extend([cellIdx])
                        if y != nYDoms-1:
                            doms[x + (y+1)*nXDoms + z*nXDoms*nYDoms].extend([cellIdx])

                        if z != 0:
                            doms[x + y*nXDoms + (z-1)*nXDoms*nYDoms].extend([cellIdx])
                        if z != nZDoms-1:
                            doms[x + y*nXDoms + (z+1)*nXDoms*nYDoms].extend([cellIdx])

        cellNeighs = [[] for _ in range(nCells)]
        for cellIdx in range(nCells):
            for i, d in enumerate(doms):
                if cellIdx in d:
                    if len(d) > 2:
                        dd = d.copy()
                        dd.remove(cellIdx)
                        cellNeighs[cellIdx].extend(dd)

        cellNeighs = [list(set(n)) for n in cellNeighs]
        return cellNeighs


    neighCells = GenCellNeighs()
    neighs = [0 for _ in range(nCells)]
    #print("Doing neighbour search of {} particles in {} cells\n".format(nCells*180, nCells))
    for i in range(nCells):
        iCell = frame[i]
        for j in neighCells[i]:
            jCell = frame[j]
            dr = iCell - jCell[:, np.newaxis]
            r = np.linalg.norm(dr, axis=2)
            if ((r < 0.3)).sum() > 0:
                neighs[i] += 1

    return neighs

def WriteNeighs(f, step, neighs):
    f.write("{}\n".format(step))
    f.write(",".join([str(n) for n in neighs]) + "\n")

with cd.TrajHandle(trajFile) as tj, open(outputFile, "w") as f:
    if args.frame_num > 0:
        WriteNeighs(f, tj.currFrameNum, GetNeighs(tj.ReadFrame(args.frame_num)))
    else:
        for i in tqdm(range(tj.maxFrames)):
            WriteNeighs(f, tj.currFrameNum, GetNeighs(tj.ReadFrame()))

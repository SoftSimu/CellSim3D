#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import celldiv as cd

parser = argparse.ArgumentParser()

parser.add_argument("traj", type=str)
parser.add_argument("-o", "--out", type=str, default="out.xyz")
parser.add_argument("-k", "--skip", type=int, default=1)
parser.add_argument("-m", "--max", type=int, default=float("inf"))
parser.add_argument("-f", "--frame", type=int, default=-1)
args = parser.parse_args()

trajPath = os.path.abspath(args.traj)


def GetMaxPart():
    print("Scanning trajectory for parameters...")
    with cd.TrajHandle(trajPath) as th:
        n = len(th.ReadFrame(th.maxFrames))
    print("Done!")
    print("System contains {0} particles".format(n*180))
    return n*180

def WriteFrame(frame, n):
    nP = len(frame)*180
    outHandle.write("%d\n" % n)
    outHandle.write("%d, t = %d\n" % (nP, th.currFrameNum))
    CoM = np.mean(np.vstack(frame), axis=0)
    for c in frame:
        c = c[:-12] - CoM
        for p in c.tolist():
            outHandle.write("C  ")
            outHandle.write("  ".join([str(a) for a in p]))
            outHandle.write("\n")

        if nP < n:
            for _ in range(n - nP):
                outHandle.write(filler)

n = GetMaxPart()
filler = "C  0.0  0.0  0.0\n"
with cd.TrajHandle(trajPath) as th, open(args.out, "w") as outHandle:
    if args.frame != -1:
        frame = th.ReadFrame(frameNum=args.frame)
        WriteFrame(frame, len(frame)*180)
    else:
        for i in tqdm(range(min(args.max, int(th.maxFrames/args.skip)))):
            frame = th.ReadFrame(inc=args.skip)
            WriteFrame(frame, n)

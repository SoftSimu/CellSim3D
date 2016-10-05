#!/usr/bin/env python3

import argparse
import os
import numpy as np
from tqdm import tqdm

import celldiv as cd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

desc = """"""

parser = argparse.ArgumentParser()

parser.add_argument("traj",
                    help="Trajectory file name",
                    type=str)

parser.add_argument("-k", "--skip",
                    help="Trajectory frame skip rate",
                    type = int,
                    default = 1)
parser.add_argument("-t", "--threshold",
                    help="Threshold distance from plane within which plotted",
                    type = float,
                    default = 0.1)
parser.add_argument("-l", "--set-limits",
                    help="Set limits for the video. Must read trajectory twice. \
                    Must be a complete trajectory file.",
                    type = bool,
                    default = True)

parser.add_argument("-o", "--out",
                    help="Output file name (with format)",
                    type = str,
                    default = "cs.png")

parser.add_argument("-c", "--clear",
                    help="Toggle clearing of storage directory",
                    type = bool,
                    default=True)


args=parser.parse_args()
trajPath = os.path.abspath(args.traj)

storePath, trajFileName = os.path.split(trajPath)
trajFileName = os.path.splitext(trajFileName)[0]

storePath += "/" + trajFileName + "/cs/"

print ("Saving to", storePath)


if not os.path.exists(storePath):
    os.makedirs(storePath)

if args.clear:
    for f in os.listdir(storePath):
        os.remove(storePath + f)

outName = os.path.splitext(args.out)

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)


ax.set_xlabel("$X$")
ax.set_ylabel("$Y$")

frames = []


maxes = []
mins = []

if args.set_limits:
    with cd.TrajHandle(trajPath) as th:
        print("Getting movie limits...")
        frame = np.vstack(th.ReadFrame(th.maxFrames))
        frame = frame - np.mean(frame, axis=0)
        zs = np.abs(frame[:, 2])
        m = zs <= args.threshold
        frame = frame[m]
        maxes = np.ceil(np.max(frame, axis=0))
        mins = np.floor(np.min(frame, axis=0))
        print("Done")


ax.set_xlim([mins[0], maxes[0]])
ax.set_ylim([mins[1], maxes[1]])

print("Creating frames...")

with cd.TrajHandle(trajPath) as th:
    for i in tqdm(range(int(th.maxFrames/args.skip))):
        ax.set_title("$t = %d$" % i)
        try:
            frame = th.ReadFrame(inc=args.skip)
            cellTypes = th.currTypes
            Xs0 = []
            Ys0 = []

            Xs1 = []
            Ys1 = []

            sysCM = np.mean(np.array([np.mean(c, axis=0) for c in frame]), axis=0)
            frame = [c - sysCM for c in frame]

            for cell, cellType in zip(frame, cellTypes):
                Zs = np.abs(cell[:, 2])
                m = Zs <= args.threshold
                data = cell[m]
                if cellType == 0:
                    Xs0.extend(data[:, 0].tolist())
                    Ys0.extend(data[:, 1].tolist())
                else:
                    Xs1.extend(data[:, 0].tolist())
                    Ys1.extend(data[:, 1].tolist())


            ax.lines=[]

            ax.plot(Xs0, Ys0, "b.", label="hard")
            ax.plot(Xs1, Ys1, "g.", label="soft")
            fig.savefig(storePath + outName[0] + "%s" % i + outName[1])

        except cd.IncompleteTrajectoryError as e:
            print(e.value)
            print("Stopping")
            break
print("Done")

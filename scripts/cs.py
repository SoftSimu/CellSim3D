#!/usr/bin/env python3

import argparse
import os
import io
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd

import celldiv as cd

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

desc = """"""

parser = argparse.ArgumentParser()

parser.add_argument("traj",
                    help="Trajectory file name",
                    type=str)

parser.add_argument("-k", "--skip",
                    help="Trajectory frame skip rate",
                    type=int,
                    default=1)
parser.add_argument("-t", "--threshold",
                    help="Threshold distance from plane within which plotted",
                    type=float,
                    default=0.1)
parser.add_argument("-l", "--set-limits",
                    help="Set limits for the video. Must read trajectory twice. \
                    Must be a complete trajectory file.",
                    type=bool,
                    default=True)

parser.add_argument("-o", "--out",
                    help="Output file name (with format)",
                    type=str,
                    default="cs.png")

parser.add_argument("-p", "--pixel",
                    help="Plot pixels instead of points",
                    action="store_true")

parser.add_argument("-c", "--clear",
                    help="Toggle clearing of storage directory",
                    type=bool,
                    default=True)

parser.add_argument("--forces", type=str, default="",
                    help="contact force csv file name", required=False)

parser.add_argument("--colormap", type=str, default="viridis",
                    required=False,
                    help="Color map to use for the force heatmap.\
                    see: http://matplotlib.org/examples/color/colormaps_reference.html")

parser.add_argument("--ref-force", type=str, default="med", required=False,
                    help="The reference force to plot relative stress too. 'med'\
                    uses median, 'max' uses the maximum force")

parser.add_argument("--last-frame", "-lf", type=int, default=-1, required=False,
                    help="Only analyze upto this frame. This is useful for\
                    analyzing really large simulations that can also be still in\
                    progress")

parser.add_argument("-f", "--frame", type=int, default=-1, required=False,
                    help="Only analyze this one frame.")


args = parser.parse_args()
trajPath = os.path.abspath(args.traj)

forcePath = []
header = []
forceFileHandle = []
header = ""
if args.forces is not "":
    forcePath = os.path.abspath(args.forces)
    forceFileHandle = open(forcePath, "r")
    header = forceFileHandle.readline()

storePath, trajFileName = os.path.split(trajPath)
trajFileName = os.path.splitext(trajFileName)[0]

storePath += "/" + trajFileName + "/cs/"

print ("Saving to", storePath)

cmap = plt.get_cmap(args.colormap)

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

        if args.last_frame > 0 and args.last_frame > th.maxFrames:
            print("Trajectory only has {0} frames".format(th.maxFrames))
            args.last_frame = -1

        if args.frame > 0 and args.frame > th.maxFrames:
            print("Trajectory only has {0} frames".format(th.maxFrames))
            args.frame = th.maxFrames

        if args.frame > 0:
            frame = np.vstack(th.ReadFrame(args.frame))
        elif args.last_frame > 0:
            frame = np.vstack(th.ReadFrame(args.last_frame))
        else:
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

cax = fig.add_axes([0.925, 0.2, 0.025, 0.6])
#norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                 orientation='vertical')

cbar.set_ticks([0.0, 0.5, 1.0])

cax.text(0.0125, 1.05, "$\\frac{F_i}{F_{%s}}$" % args.ref_force, fontsize=20)

print("Creating frames...")


def GetStepForces(step):
    done = False
    stepForces = io.StringIO()
    stepForces.write(header) # write header
    n = 0
    while True:
        x = forceFileHandle.tell()
        line = forceFileHandle.readline().strip().split(",")
        if int(line[0]) < step:
            for _ in range(int(line[1]) * 180):
                #next(forceFileHandle)
                forceFileHandle.readline()

        if int(line[0]) > step:
            forceFileHandle.seek(x)
            break

        if int(line[0]) == step:
            n = int(line[1])*180
            forceFileHandle.seek(x)
            for _ in range(n):
                stepForces.write(forceFileHandle.readline())

    stepForces.seek(0)
    return stepForces


with cd.TrajHandle(trajPath) as th:
    n = args.last_frame if args.last_frame > 0 else th.maxFrames
    for i in tqdm(range(int(n/args.skip))):
        ax.set_title("$t = {0}$".format(i+1))
        try:
            frame = th.ReadFrame(inc=args.skip)
            if args.frame > 0 and th.currFrameNum < args.frame:
                continue

            if args.frame > 0 and th.currFrameNum > args.frame:
                break

            cellTypes = th.currTypes
            Xs0 = []
            Ys0 = []

            Xs1 = []
            Ys1 = []

            Forces = []
            avgCellForces = []
            datLens = []

            sysCM = np.mean(np.vstack(frame), axis=0)
            frame = [c[:180] - sysCM for c in frame]

            if args.forces is not "" and th.step>0:
                sF = GetStepForces(th.step)
                Forces = pd.read_csv(sF)[["cell_ind", "FX", "FY", "FZ"]]
                sF.close()

            c = 0
            for cell, cellType in zip(frame, cellTypes):
                Zs = np.abs(cell[:, 2])
                m = Zs <= args.threshold
                data = cell[m]
                datLens.append(data.shape[0])
                if cellType == 0:
                    Xs0.append(data[:, 0])
                    Ys0.append(data[:, 1])
                else:
                    Xs1.append(data[:, 0])
                    Ys1.append(data[:, 1])

                if args.forces is not "" and th.step > 0:
                    cellForces = Forces[Forces["cell_ind"] == c][["FX", "FY", "FZ"]]
                    avgCellForces.append(np.linalg.norm(cellForces.as_matrix(),
                                                        axis=1).sum())
                c += 1


            ax.lines=[]

            if args.pixel:
                ax.plot(np.hstack(Xs0), np.hstack(Ys0), "k,", label="hard")
                if sum(cellTypes) > 0:
                    ax.plot(np.hstack(Xs1), np.hstack(Ys1), "k,", label="soft")
            else:
                ax.plot(np.hstack(Xs0), np.hstack(Ys0), "k.", label="hard")
                if sum(cellTypes) > 0:
                    ax.plot(np.hstack(Xs1), np.hstack(Ys1), "k.", label="soft")

            if args.forces is not "" and th.step>0:
                ax.patches=[]
                ref_force = np.median(avgCellForces)
                if args.ref_force == "max":
                    ref_force = max(avgCellForces)

                for x,y,f in zip(Xs0, Ys0, avgCellForces):
                    p = np.array([x,y]).T
                    hull = ConvexHull(p)
                    ax.fill(p[hull.vertices, 0], p[hull.vertices, 1],
                            color=cmap(f/ref_force))

            fig.savefig(storePath + outName[0] + "{0}".format(i+1) + outName[1])

        except cd.IncompleteTrajectoryError as e:
            print(e.value)
            print("Stopping")
            break
print("Done")

if args.forces is not "":
    forceFileHandle.close()

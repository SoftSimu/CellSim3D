#!/usr/bin/env python3

import matplotlib

matplotlib.use('Agg')

import sys, os, argparse

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from subprocess import call
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
import matplotlib.animation

import celldiv as cd

from tqdm import tqdm

desc="""
Creates snapshots of the movement  of the center of mass of the system of cells.
"""
parser = argparse.ArgumentParser(description=desc,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("trajPath", nargs=1, help="Path to the trajectory file.")

parser.add_argument("-fps",
                    help="Optionally set the fps of the output video to this value",
                    default=20,
                    type=int)

parser.add_argument("-s",
                    help="Optionally set the scaling factor of the axes in the output video",
                    default=1,
                    type=float)

parser.add_argument("-d",
                    help="Directory relative to the root data directory to store the video.",
                    default="",
                    type=str)
parser.add_argument("--out",
                    help="Output file name and format",
                    default="3d_CoM.mp4",
                    type=str)
parser.add_argument("-k",
                    help="Skip interval",
                    type = int,
                    default =1);

parser.add_argument("--res",
                    help="16:9 resolution. E.g 720p for 1280x720, \
                    1080p for 1920:1080",
                    default="1080p",
                    type=str)
parser.add_argument("-m", "--movie",
                    help="Make a movie in 3D",
                    default=False)


args = parser.parse_args()

CoM = []
cellCoMs = []
trajPath = os.path.abspath(args.trajPath[0])

storPath, trajFileName = os.path.split(trajPath)
trajFileName = os.path.splitext(trajFileName)[0]

storPath += "/" + trajFileName + "/" + args.d
outName = args.out

print ("Saving to %s" % storPath)

try:
    os.makedirs(storPath)
except:
    pass

print("Trajectory read progress:")
with cd.TrajHandle(trajPath) as th:
    for i in tqdm(range(int(th.maxFrames/args.k))):
        try:
            frame = th.ReadFrame(inc=args.k)
            cc = np.array([np.mean(c, axis=0) for c in frame])
            CoM.append(np.mean(cc, axis=0))
            cellCoMs.append(cc)
        except cd.IncompleteTrajectoryError as e:
            print(e.value)
            print("Stopping")
            break


CoM = np.array(CoM)
print("Making plots")
plt.subplot(2, 2, 1)
plt.plot(CoM[:, 0], CoM[:, 1], '.')
plt.xlabel("X")
plt.ylabel("Y")

plt.subplot(2, 2, 2)
plt.plot(CoM[:, 0], CoM[:, 2], '.')
plt.xlabel("X")
plt.ylabel("Z")

plt.subplot(2, 2, 3)
plt.plot(CoM[:, 1], CoM[:, 2], '.')
plt.xlabel("Y")
plt.ylabel("Z")

plt.tight_layout()
plt.savefig(storPath + "COM.png")
plt.clf()

fig = plt.figure()
fps = args.fps
nFrames = CoM.shape[0]
tPerFrame = 1.0/fps
dur = (nFrames-1)*tPerFrame
s = args.s

ax = fig.add_subplot(1, 1, 1, projection='3d')

maxes = np.max(np.vstack([np.max(np.vstack(f), axis=0) for f in cellCoMs]), axis=0)
mins = np.min(np.vstack([np.min(np.vstack(f), axis=0) for f in cellCoMs]), axis=0)
xMax = maxes[0]
xMin = mins[0]

yMax = maxes[1]
yMin = mins[1]

zMax = maxes[2]
zMin = mins[2]

ax.set_xlim3d([xMax, xMin])
ax.set_ylim3d([yMax, yMin])
ax.set_zlim3d([zMax, zMin])

def make3DFrame(i):
    i = np.floor(i/tPerFrame)

    ax.scatter3D(CoM[i, 0], CoM[i, 1], CoM[i, 2], '.')
    return mplfig_to_npimage(fig)

def make3DFrameWithCells(i):
    i = int(np.floor(i/tPerFrame))
    ax.cla()
    ax.scatter3D(CoM[i, 0], CoM[i, 1], CoM[i, 2], '.')
    ax.scatter3D(cellCoMs[i][:, 0], cellCoMs[i][:, 1], cellCoMs[i][:, 2],
                 c='red', alpha=0.5, marker='.')

    ax.set_xlim3d([xMax, xMin])
    ax.set_ylim3d([yMax, yMin])
    ax.set_zlim3d([zMax, zMin])
    return mplfig_to_npimage(fig)


# This condition is false if there is more than one cell in the system
if CoM.shape[0] > 1:
    frameFunc = make3DFrameWithCells
else:
    frameFunc = make3DFrame

if args.movie:
    print ("Making movie...")
    animation = mpy.VideoClip(frameFunc, duration=dur)
    animation.write_videofile(storPath + outName, fps=fps)

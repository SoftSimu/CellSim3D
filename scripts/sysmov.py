#!/usr/bin/env python

import matplotlib

matplotlib.use('Agg')


import sys, os, argparse

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from subprocess import call
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

desc="""
Creates snapshots of the movement  of the center of mass of the system of cells.
"""
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("trajPath", nargs=1, help="Path to the trajectory file.")

parser.add_argument("--fps",
                    help="Optionally set the fps of the output video to this value",
                    default=20,
                    type=int)

parser.add_argument("--s",
                    help="Optionally set the scaling factor of the axes in the output video",
                    default=1,
                    type=float)

parser.add_argument("--d",
                    help="Directory relative to the root data directory to store the video.",
                    default="",
                    type=str)
parser.add_argument("--out",
                    help="Output file name and format",
                    default="3d_CoM.mp4",
                    type=str)
parser.add_argument("--res",
                    help="16:9 resolution. E.g 720p for 1280x720, 1080p for 1920:1080",
                    default="1080p",
                    type=str)

args = parser.parse_args()

nAtoms = 0
msg = 0
CoMx = 0.0
CoMy = 0.0
CoMz = 0.0

CoM = []
line = 0

CoMxt = []
CoMyt = []
CoMzt = []
CoMxt_cell = []
CoMyt_cell = []
CoMzt_cell = []
c = 0

trajPath = os.path.abspath(args.trajPath[0])

storPath, trajFileName = os.path.split(trajPath)
trajFileName = os.path.splitext(trajFileName)[0]

storPath += "/" + trajFileName + "/" + args.d
outName = args.out

print "Saving to %s" % storPath

try:
    os.makedirs(storPath)
except:
    pass

with open(trajPath, "r") as trajFile:
    line = trajFile.readline()
    while(line != ""):
        line = line.strip()
        nAtoms = int(line)
        step = int(trajFile.readline().strip()[6:])
        nCells = nAtoms/192
        print "Processing %d ..." % step
        CoMxt_cell.append([])
        CoMyt_cell.append([])
        CoMzt_cell.append([])
        for cell in xrange(nCells):
            for atom in xrange(192):
                line = trajFile.readline()
                line = line.strip()
                line = line.split(',  ');
                CoMx += float(line[0])
                CoMy += float(line[1])
                CoMz += float(line[2])

            CoMx = CoMx / 192
            CoMy = CoMy / 192
            CoMz = CoMz / 192
            CoMxt_cell[c].append(CoMx)
            CoMyt_cell[c].append(CoMy)
            CoMzt_cell[c].append(CoMz)

            CoMx = 0.0
            CoMy = 0.0
            CoMz = 0.0
        c += 1
        line = trajFile.readline()

CoMxt = [sum(a)/len(a) for a in CoMxt_cell]
CoMyt = [sum(a)/len(a) for a in CoMyt_cell]
CoMzt = [sum(a)/len(a) for a in CoMzt_cell]

CoMxt = np.array(CoMxt)
CoMyt = np.array(CoMyt)
CoMzt = np.array(CoMzt)


plt.subplot(2, 2, 1)
plt.plot(CoMxt, CoMyt, '.')
plt.xlabel("X")
plt.ylabel("Y")

plt.subplot(2, 2, 2)
plt.plot(CoMxt, CoMzt, '.')
plt.xlabel("X")
plt.ylabel("Z")

plt.subplot(2, 2, 3)
plt.plot(CoMyt, CoMzt, '.')
plt.xlabel("Y")
plt.ylabel("Z")

plt.tight_layout()
plt.savefig(storPath + "COM.png")
plt.clf()


fig = plt.figure()
fps = args.fps
nFrames = CoMxt.size
tPerFrame = 1.0/fps
dur = (nFrames-1)*tPerFrame
s = args.s

def make3DFrame(i):
    i = np.floor(i/tPerFrame)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(CoMxt[i], CoMyt[i], CoMzt[i], '.')

    xMax = s*max(np.ceil(max(max(CoMxt_cell))),
                 abs(np.floor(min(min(CoMxt_cell)))))
    xMin = -1 * xMax

    yMax = s*max(np.ceil(max(max(CoMyt_cell))),
                 abs(np.floor(min(min(CoMyt_cell)))))
    yMin = -1 * yMax

    zMax = s*max(np.ceil(max(max(CoMzt_cell))),
                 abs(np.floor(min(min(CoMzt_cell)))))
    zMin = -1 * zMax

    ax.set_xlim3d([xMax, xMin])
    ax.set_ylim3d([yMax, yMin])
    ax.set_zlim3d([zMax, zMin])


    return mplfig_to_npimage(fig)

def make3DFrameWithCells(i):
    i = int(np.floor(i/tPerFrame))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.scatter(CoMxt[i], CoMyt[i], CoMzt[i], '.')
    ax.scatter(CoMxt_cell[i], CoMyt_cell[i], CoMzt_cell[i], c='red', alpha=0.5,
               marker='.')

    xMax = s*max(np.ceil(max(max(CoMxt_cell))),
                 abs(np.floor(min(min(CoMxt_cell)))))
    xMin = -1 * xMax

    yMax = s*max(np.ceil(max(max(CoMyt_cell))),
                 abs(np.floor(min(min(CoMyt_cell)))))
    yMin = -1 * yMax

    zMax = s*max(np.ceil(max(max(CoMzt_cell))),
                 abs(np.floor(min(min(CoMzt_cell)))))
    zMin = -1 * zMax

    ax.set_xlim3d([xMax, xMin])
    ax.set_ylim3d([yMax, yMin])
    ax.set_zlim3d([zMax, zMin])

    return mplfig_to_npimage(fig)


# This condition is false if there is more than one cell in the system
if len(CoMxt_cell[-1]) > 1:
    frameFunc = make3DFrameWithCells
else:
    frameFunc = make3DFrame

print "Generating Gifs..."
print "3D..."

animation = mpy.VideoClip(frameFunc, duration=dur)
animation.write_videofile(storPath + outName, fps=fps)

import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from cellsim import CellSim3DTools as cs
from matplotlib.widgets import Slider, RadioButtons, Button
import os

parser = argparse.ArgumentParser()

parser.add_argument("traj", type=str)
parser.add_argument("-k", "--skip", type=int, default=1)

args = parser.parse_args()

trajFilePath = os.path.abspath(args.traj)

c = plt.get_cmap("viridis")
magma = plt.get_cmap("magma")
norm = mpl.cm.ScalarMappable(cmap=magma)

with cs.Trajectory(trajFilePath) as th:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    fig.subplots_adjust(bottom=0.25, left=0.25)
    axS = fig.add_axes([0.1, 0.1, 0.8, 0.03])
    axR = fig.add_axes([0.1, 0.5, 0.2, 0.2])
    axLB = fig.add_axes([0.1, 0.05, 0.1, 0.04])
    axRB = fig.add_axes([0.8, 0.05, 0.1, 0.04])

    startFrame = 0
    endFrame = th.numFrames

    nCells = th.MaxNumCells()
    nNodes = nCells*180
    cIncr = 1.0/nCells

    d={"No Forces": False, "Forces": True}
    lasIdx = 0
    plot_forces = False

    def Plot(frameIdx):
        frameIdx = int(frameIdx)
        global lastIdx
        lastIdx = frameIdx
        ax.cla()
        #ax.set_aspect=('equal')
        frame = th.ReadFrame(frameIdx)
        CoM = frame.CoM()
        ax.set_title("Frame {}".format(frameIdx))
        for i, cell in enumerate(frame):
            pos = cell.pos[:-12] - CoM
            if not plot_forces:
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], lw=1, color=c(i*cIncr), s=4)
            else:
                forces = cell.forces[:-12]
                colors = []
                fMags = np.linalg.norm(forces, axis=1)

                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                           color=norm.to_rgba(fMags), s=4)

    s = Slider(axS, "Time", 0, endFrame, valinit=0, valfmt="%d")
    s.on_changed(Plot)

    def RadioSelector(label):
        global plot_forces
        plot_forces = d[label]
        s.set_val(lastIdx)

    def PlotOneAfterLastFrame(a):
        if lastIdx != endFrame:
            s.set_val(lastIdx+1)

    def PlotOneBeforeLastFrame(a):
        if lastIdx != 0:
            s.set_val(lastIdx-1)

    r = RadioButtons(axR, d.keys())
    r.on_clicked(RadioSelector)


    lButton = Button(axLB, "<--")
    rButton = Button(axRB, "-->")

    lButton.on_clicked(PlotOneBeforeLastFrame)
    rButton.on_clicked(PlotOneAfterLastFrame)



    Plot(0)

    plt.show()

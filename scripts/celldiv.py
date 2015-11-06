#!/usr/bin/env ipython3

import sys
import os
import numpy as np

class TrajHandle(object):
    """
    Class that handles everything about getting trajectories. I will
    "Intelligently" read the trajectory and store it into an easily
    accessible form, and make that form available to the analysis scripts
    """
    def __init__(self, filePath, numPart=192, ramFrac=0.5,
                 writeName="celldiv-traj.npz", forceReRead=False):

        trajPath = os.path.abspath(filePath)

        self.fileExists = os.path.isfile(trajPath)

        try:
            os.makedirs(storePath)
        except:
            pass

        if not self.fileExists:
            print("Can't find the trajectory.")
            print("%s" % trajPath)
        else:
            print("Found trajectory, reading now...")

            # Check if the trajectory is newer than what was stored
            storePath, fileName = os.path.split(trajPath)
            fileName = os.path.splitext(fileName)[0]
            storePath = os.path.join(storePath, fileName, "celldiv-traj",)
            outFileName = storePath + os.sep + writeName
            self.npTrajPath = outFileName

            if (forceReRead) or (not os.path.isfile(outFileName)) or \
               (os.path.getmtime(trajPath) > os.path.getmtime(outFileName)):

                print("Trajectory is either newer, didn't exist before, " + \
                      "or reread was forced")

                print("Reading...")
                self.trajFileSize = os.path.getsize(trajPath)
                print("File size: %.3f MB" % (self.trajFileSize/(1024.0*1024)))
                self.ReadTraj(trajPath, numPart, ramFrac, outFileName)
            else:
                print("This trajectory was already read, skipping read")

            print("Trajectory retrieved!")
            print("It's at %s" % outFileName)

    def ReadTraj(self,  trajPath, numPart, ramFrac, outFileName, compress=True):
        # Start reading the trajectory
        # Assume binary trajectory for now.

        self.bytesRead = 0
        self.readDone = False

        with open(trajPath, "rb") as tfh:

            def GetArray(d, c):
                a = np.fromfile(tfh, dtype=d, count=c, sep="")
                self.bytesRead += c*4
                if a.size is not c:
                    self.readDone = True

                return a
                #trajVersion = np.fromfile(tfh, dtype=np.int32, count=1, sep="")

            maxNCells = GetArray(np.int32, 1)[0]
            variableStiffness = GetArray(np.int32, 1)[0]
            maxFrames = GetArray(np.int32, 1)[0]
            print(maxNCells)
            print(bool(variableStiffness))
            print(maxFrames)


            #self.traj = np.zeros((maxFrames, maxNCells, numPart, 3))
            traj = []


            self.bytesRead += 1*4
            c = -1
            while self.bytesRead < self.trajFileSize:
                step = GetArray(np.int32, 1)[0]
                frameCount = GetArray(np.int32, 1)[0]
                print("Reading frame %d" % frameCount)
                nCells = GetArray(np.int32, 1)[0]
                frame = np.zeros((nCells, numPart, 3))

                for i in range(nCells):
                    cellInd = GetArray(np.int32, 1)[0]
                    x = GetArray(np.float32, numPart)
                    y = GetArray(np.float32, numPart)
                    z = GetArray(np.float32, numPart)

                    if self.readDone:
                        print("Data missing")
                        print("Read until step %d" % step)
                        break
                    else:
                        frame[cellInd, :, 0] = x
                        frame[cellInd, :, 1] = y
                        frame[cellInd, :, 2] = z

                    c+= 1
                traj.append(frame)

            if compress:
                np.savez_compressed(outFileName, *traj)
            else:
                np.savez(outFileName, *traj)


    def GetTraj(self):
        return self.npTrajPath




def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("trajPath", help='Trajectory file to test.')

    args = parser.parse_args()

    th = TrajHandle(args.trajPath)

    with np.load(th.GetTraj(), 'r') as t:
        print(t["arr_0"].shape)


if __name__ == "__main__":
    main()

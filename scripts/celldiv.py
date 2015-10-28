#!/usr/bin/env python3

import sys
import os
import numpy as np

class TrajHandle(object):
    """
    Class that handles everything about getting trajectories. I will
    "Intelligently" read the trajectory and store it into an easily
    accessible form, and make that form available to the analysis scripts
    """

    def __init__(self, filePath, numPart=192):
        trajPath = os.path.abspath(filePath)
        self.fileExists = os.path.isfile(trajPath)

        if not self.fileExists:
            print("Can't find the trajectory.")
            print("%s" % trajPath)
        else:
            print("Found trajectory, reading now...")

            # Start reading the trajectory
            # Assume binary trajectory for now.
            trajFileSize = os.path.getsize(trajPath)
            print(trajFileSize)
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

                maxNCells = GetArray(np.int32, 1)
                variableStiffness = GetArray(np.int32, 1)
                maxFrames = GetArray(np.int32, 1)
                print(maxFrames)


                self.traj = np.zeros((maxFrames, maxNCells, numPart, 3))


                self.bytesRead += 1*4
                c = -1
                while self.bytesRead < trajFileSize:
                    step = GetArray(np.int32, 1)
                    frameCount = GetArray(np.int32, 1)
                    nCells = GetArray(np.int32, 1)
                    frame = np.zeros((maxNCells, numPart, 3))
                    #print(step, frameCount, nCells)

                    for i in range(nCells):
                        cellInd = GetArray(np.int32, 1)
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

                    if self.readDone:
                        break
                    c+= 1
                    self.traj[c] = frame



    def GetTraj(self):
        return self.traj




def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("trajPath", help='Trajectory file to test.')

    args = parser.parse_args()
    print(args.trajPath)

    th = TrajHandle(args.trajPath)

    print(np.mean(th.GetTraj()[0, 0], axis=0))


if __name__ == "__main__":
    main()

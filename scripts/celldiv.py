#!/usr/bin/env python3

import sys
import os
import numpy as np

class IncompleteTrajectoryError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class TrajHandle(object):
    """
    Class that handles everything about getting trajectories. I will
    "Intelligently" read the trajectory and store it into an easily
    accessible form, and make that form available to the analysis scripts
    """
    def __init__(self, filePath, numPart=192):
        import os
        import sys
        import numpy as np

        trajPath = os.path.abspath(filePath)

        self.fileExists = os.path.isfile(trajPath)
        self.trajHandle = None
        self.numPart = numPart
        self.frameReadFailed = False
        self.fatalError = False
        self.lastFrameNum = -1
        self.currFrameNum = 0
        self.nCellsLastFrame = 0
        self.frame = None
        self.fileSize = 0

        self.intSize = np.array([0], dtype=np.int32).itemsize
        self.floatSize = np.array([0.1], dtype=np.float32).itemsize

        if not self.fileExists:
            raise IOError("Couldn't find the trajectory file.")
        else:
            try:
                self.Initialize(trajPath)
            except:
                print("Something went wrong in:", sys.exc_info()[0])
                raise

    def _GetArray(self, d, c):
                a = np.fromfile(self.trajHandle, dtype=d, count=c, sep="")
                if a.size is not c:
                    self.readFailed = True
                    raise IncompleteTrajectoryError("Data missing from trajectory.")
                return a


    def Initialize(self, trajPath):
        self.fileSize = os.path.getsize(trajPath)
        self.trajHandle = open(trajPath, 'rb')
        try:
            self.maxNCells = self._GetArray(np.int32, 1)[0]
            self.variableStiffness = self._GetArray(np.int32, 1)[0]
            self.maxFrames = self._GetArray(np.int32,1)[0]
            self.trajStart = self.trajHandle.tell()
        except IncompleteTrajectoryError as e:
            print("This trajectory file is a stub.")
            self.close()
            self.FatalError = True
            raise


    def GoToFrame(self, frameNum, force=False):
        offset = self.currFrameNum
        if force:
            offset = 0

        if self.currFrameNum < frameNum or force:
            # move forwards
            for i in range(frameNum - offset - 1):
                self.trajHandle.seek(2*self.intSize, os.SEEK_CUR)
                nC = self._GetArray(np.int32, 1)[0]
                self.trajHandle.seek(nC*(self.intSize + 3*self.numPart*self.floatSize),
                                     os.SEEK_CUR)
            return False
        elif self.currFrameNum > frameNum:
            # move backwards
            self.trajHandle.seek(self.trajStart, os.SEEK_SET)
            self.GoToFrame(frameNum, force=True)
            return False
        else:
            return True

    def ReadFrame(self, frameNum=None, inc=1):
        # Read the frame!
        if self.fatalError:
            self.close()
            raise IncompleteTrajectoryError("Reading this trajectory failed.")


        if self.frameReadFailed or self.lastFrameNum == self.maxFrames:
            print("Can't read past frame %d" % self.currFrameNum)
            return self.frame
        else:
            try:
                if frameNum is not None:
                    if(self.GoToFrame(frameNum)):
                        return self.frame

                step = self._GetArray(np.int32, 1)[0]
                frameNum = self._GetArray(np.int32, 1)[0]
                nCells = self._GetArray(np.int32, 1)[0]
                frame  = np.empty((self.numPart*nCells, 3))
                for i in range(nCells):
                    cellInd = self._GetArray(np.int32, 1)[0]
                    x = self._GetArray(np.float32, self.numPart)
                    y = self._GetArray(np.float32, self.numPart)
                    z = self._GetArray(np.float32, self.numPart)

                    frame[i*self.numPart:(i+1)*self.numPart, 0] = x
                    frame[i*self.numPart:(i+1)*self.numPart, 1] = y
                    frame[i*self.numPart:(i+1)*self.numPart, 2] = z

                self.frame = frame
                self.nCellsLastFrame = nCells

                self.lastFrameNum = self.currFrameNum
                self.currFrameNum = frameNum

                return self.frame
            except IncompleteTrajectoryError as e:
                print ("Trajectory incomplete, maxFrames set to %d" % self.maxFrames)
                raise

    def close(self):
        self.trajHandle.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("trajPath", help='Trajectory file to test.')
    args = parser.parse_args()

    th = TrajHandle(args.trajPath)
    print(th.frame)

    th.close()

if __name__ == "__main__":
    main()

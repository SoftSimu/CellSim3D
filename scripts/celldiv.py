#!/usr/bin/env python3

import sys
import os
import numpy as np

class IncompleteTrajectoryError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

EndOfTrajectoryError = IncompleteTrajectoryError

class TrajHandleBase(object):
    def __init__(self, filePath, numNodes=192):
        if type(self) is TrajHandleBase:
            print("This class is not meant to be used directly. Use the \
            TrajHandle class.")
            raise NotImplementedError

        import os
        import sys
        import numpy as np

        trajPath = os.path.abspath(filePath)

        self.fileExists = os.path.isfile(trajPath)

        if not self.fileExists:
            raise IOError("Couldn't find the trajectory file at %s" % trajPath)
        else:
            try:
                self.Initialize(trajPath)
            except:
                   print("Something went wrong in:", sys.exec_info()[0])
                   raise

        self.fatalError = False

    def __getFileHandle(self, trajPath):
        raise NotImplementedError

    def __setInitVars():
        raise NotImplementedError

    def initialize (self, trajPath):
        self.fileSize = os.path.getsize(trajPath)
        self.trajHandle = __getFileHandle(trajPath)
        try:
            self.params = __setInitVars()
        except IncompleteTrajectoryError:
            print("This trajectory file is a stub.")
            self.closeHandle()
            self.fatalError = True
            raise


    def closeHandle(self):
        self.trajHandle.close()


class TrajHandleBinary(object):
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
        self.floatSize  =np.array([0.1], dtype=np.float32).itemsize

        if not self.fileExists:
            raise IOError("Couldn't find the trajectory file. at %s" % trajPath)
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

    def ResetRead(self, beforeHeader = False):
        """
        Resets the reader to just after the header. The next read will return
        the first frame. Pass beforeHeader = True to reset to the start of the
        trajectory file, then Initialize must be run again or frame will
        contain bogus data.
        """
        if beforeHeader:
            self.trajHandle.seek(0, os.SEET_SET)
        else:
            self.trajHandle.seek(self.trajStart, os.SEEK_SET)

    def GoToFrame(self, frameNum, force=False):
        offset = self.currFrameNum
        if force:
            offset = 0

        if self.currFrameNum < frameNum or force:
            # move forwards
            for i in range(frameNum - offset - 1):
                self.trajHandle.seek(2*self.intSize, os.SEEK_CUR)
                nC = self._GetArray(np.int32, 1)[0]
                i = 1
                if self.variableStiffness:
                    i+=1
                self.trajHandle.seek(nC*(i*self.intSize + 3*self.numPart*self.floatSize),
                                     os.SEEK_CUR)
            return False
        elif self.currFrameNum > frameNum:
            # move backwards
            self.ResetRead()
            self.GoToFrame(frameNum, force=True)
            return False
        else:
            return True

    def ReadFrame(self, frameNum=None, inc=1):
        # Read the frame!
        if self.fatalError:
            self.close()
            raise IncompleteTrajectoryError("Reading this trajectory failed.")

        if self.lastFrameNum == self.maxFrames:
            raise EndOfTrajectoryError("At end of trajectory.")

        if self.frameReadFailed:
            self.lastFrameNum = self.currFrameNum
            raise IncompleteTrajectoryError("Can't read past frame %d" %
                                            self.currFrameNum)
        else:
            try:
                if frameNum is not None:
                    if(self.GoToFrame(frameNum)):
                        return self.frame

                if self.currFrameNum > 0:
                    self.GoToFrame(self.currFrameNum + inc)

                step = self._GetArray(np.int32, 1)[0]
                frameNum = self._GetArray(np.int32, 1)[0]
                nCells = self._GetArray(np.int32, 1)[0]
                frame = [ np.zeros((192, 3)) for i in range(nCells)]
                types = [0 for i in range(nCells)]

                for i in range(nCells):
                    cellInd = self._GetArray(np.int32, 1)[0]
                    x = self._GetArray(np.float32, self.numPart)
                    y = self._GetArray(np.float32, self.numPart)
                    z = self._GetArray(np.float32, self.numPart)

                    if self.variableStiffness:
                        types[i] = self._GetArray(np.int32, 1)[0]

                    frame[i][:, 0] = x
                    frame[i][:, 1] = y
                    frame[i][:, 2] = z

                self.frame = frame
                self.nCellsLastFrame = nCells

                self.lastFrameNum = self.currFrameNum
                self.currFrameNum = frameNum
                self.step = step
                self.currTypes = types

                return self.frame
            except IncompleteTrajectoryError as e:
                print ("Trajectory incomplete, maxFrames set to %d" % self.lastFrameNum)
                self.fatalError = True
                raise

    def close(self):
        self.trajHandle.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_val, trace):
        self.close()


class TrajHandleAscii(object):
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_val, trace):
        self.close()

    def close(self):
        self.trajHandle.close()

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
        self.frame = None
        self.fileSize = 0
        self.lineCounter = 0

        if not self.fileExists:
            raise IOError("Couldn't find the trajectory file at %s." % trajPath)
        else:
            try:
                self.Initialize(trajPath)
            except:
                print("Something went wrong in:", sys.exc_info()[0])
                raise

    def _GetLine(self):
        self.lineCounter += 1
        return self.trajHandle.readline().strip()

    def Initialize(self, trajPath):
        self.fileSize = os.path.getsize(trajPath)
        self.trajHandle = open(trajPath, 'r')
        self.lineCounter = 0
        try:
            self._GetLine() # Header start
            self._GetLine() # Max number of cells
            self.maxNCells = int(self._GetLine())

            self._GetLine() # using variable stiffness
            s = self._GetLine().lower()
            self.variableStiffness = None
            if (s == "true"):
                self.variableStiffness = True
            elif(s == "false"):
                self.variableStiffness = False
            else:
                raise ValueError("Could not discern stiffness type")

            self._GetLine() # Max number of frames
            self.maxFrames = int(self._GetLine())
            self._GetLine() # Header End
            self.headerLength = self.lineCounter
        except ValueError:
            self.fatalError = True
            self.close()
            raise

    def ResetRead(self, beforeHeader = False):
        """
        Resets the reader to just after the header. The next read will return
        the first frame. Pass beforeHeader = True to reset to the start of the
        trajectory file, then Initialize must be run again or frame will
        contain bogus data.
        """
        self.trajHandle.seek(0, os.SEEK_SET)
        self.lineCounter = 0
        if not beforeHeader:
            for i in range(self.headerLength):
                _GetLine()
            self.lineCounter = 8

    def GoToFrame(self, frameNum, force=False):
        offset = self.currFrameNum
        if force:
            offset = 0

        if self.currFrameNum < frameNum or force:
            # move forwards
            for i in range(frameNum - offset - 1):
                nAtoms = int(self._GetLine())
                self._GetLine()
                for j in range(int(nAtoms/192)):
                    self._GetLine()
                    for k in range(192):
                        self._GetLine()
            return False
        elif self.currFrameNum > frameNum:
            self.ResetRead()
            self.GoToFrame(frameNum, force=True)
            return False
        else:
            return True



    def ReadFrame(self, frameNum = None, inc=1):
        if self.fatalError:
            self.close()
            raise IncompleteTrajectoryError("Reading this trajectory failed.")

        if self.frameReadFailed or self.lastFrameNum == self.maxFrames:
            print("This trajectory only has %d frames" % self.maxFrames)
            return self.frame
        else:
            try:
                if frameNum is not None:
                    if frameNum > self.maxFrames:
                        print("This trajectory only has %d frames" % self.maxFrames)
                        return self.frame
                    if (self.GoToFrame(frameNum)):
                        return self.frame

                if self.currFrameNum > 0:
                    self.GoToFrame(self.currFrameNum + inc)

                nAtoms = int(self._GetLine())
                nCells = int(nAtoms/192)
                l = self._GetLine().split(' ')
                step = int(l[1])
                frameNum = int(l[3])
                #print(step, frameNum)
                frame = []
                for i in range(nCells):
                    cellInd = int(self._GetLine().split(' ')[1])
                    for j in range(192):
                        l = self._GetLine().split(',  ')

                        frame.append([float(l[0]), float(l[1]), float(l[2])])


                self.frame = np.array(frame)
                self.nCellsLastFrame = nCells

                self.lastFrameNum = self.currFrameNum
                self.currFrameNum = frameNum + 1

                return self.frame
            except Exception as e:
                print("Unexpected error in: ", sys.exc_info()[0])
                raise









def TrajHandle(fPath):
    fileIsBinary = False
    with open(fPath) as f:
        try:
            f.readline()
        except ValueError:
            fileIsBinary = True

    if fileIsBinary:
        return TrajHandleBinary(fPath)
    else:
        return TrajHandleAscii(fPath)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("trajPath", help='Trajectory file to test.')
    args = parser.parse_args()

    with TrajHandle(args.trajPath) as th:
        for i in range(3):
            th.ReadFrame(inc=2)[0]

if __name__ == "__main__":
    main()

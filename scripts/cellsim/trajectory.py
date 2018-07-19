"""
This file contains tools to read CellSim3D files.
"""

import h5py
import os
import sys

import numpy as np

from frame import Frame
from cell import Cell

class Trajectory(object):
    """
    Class that contains tools to read CellSim3D trajectory files.
    """
    def __init__(self, filePath):

        self.h5Handle = None
        self.trajPath = os.path.abspath(filePath)
        (self.trajDir, self.trajName) = os.path.split(self.trajPath)

        if not os.path.isfile(self.trajPath):
            raise IOError("Couldn't find any trajectory files at {}".format(self.trajPath))
        else:
            self.h5Handle = h5py.File(self.trajPath)
            self._ReadSimParams()
            self._ReadMetaData()

    def _ReadSimParams(self):
        self.simParams = {}
        for (attr, val) in self.h5Handle.attrs.items():
            parent = self.simParams
            tree = attr.split(".")

            for node in tree[:-1]:
                if node not in parent:
                    parent[node] = {}
                parent = parent[node]

            attrName = tree[-1]

            if "name" in attrName or "version" in attrName:
                parent[attrName] = val.decode("utf8")
            else:
                if len(val.shape) == 1 and val.shape[0] == 1:
                    val = val[0]
                parent[attrName] = val

    def GetSimParams(self):
        return self.simParams

    def _ReadMetaData(self):
        """
        This function will set all meta data such as number of max cells, number of
        frames in the file, etc.
        """

        self.numFrames = len(self.h5Handle.items())
        self.fileSize = os.path.getsize(self.trajPath)
        self.lastModified = os.path.getmtime(self.trajPath)

        self.corrupted = False

        cp = self.simParams["core_params"]

        expectedNumFrames = cp["div_time_steps"] + cp["non_div_time_steps"]
        self.writeInt = cp["traj_write_int"]


        if self.numFrames is not expectedNumFrames:
            self.corrupted = True

        self.metaData = {
            "numFrames" : self.numFrames,
            "fileLocation": self.trajPath,
            "fileSize": self.fileSize,
            "lastModifed": self.lastModified,
            "detailLevel" : "Everything",
            "corrupted" : self.corrupted
        }

    def _ReadSimList(self, hdf5Group, dataSetName):
        dataSet = hdf5Group.get(dataSetName)
        tempArr = np.empty(dataSet.shape)
        dataSet.read_direct(tempArr)
        return tempArr

    def _ReadSimList3D(self, hdf5Group, dataSetName):
        tempX = self._ReadSimList(hdf5Group, dataSetName + ".x")
        tempY = self._ReadSimList(hdf5Group, dataSetName + ".y")
        tempZ = self._ReadSimList(hdf5Group, dataSetName + ".z")

        return np.array([tempX, tempY, tempZ]).T


    def GetFrame(self, index):
        index *= self.writeInt
        fGroup = self.h5Handle.get("frame{}".format(index))
        nCells = len(fGroup.items()) - 1
        frame = Frame(index)
        volumes = self._ReadSimList(fGroup, "volumes")

        for cellInd in range(nCells):
            cGroup = fGroup.get("cell{}".format(cellInd))

            pos = self._ReadSimList3D(cGroup, "pos")
            vels = self._ReadSimList3D(cGroup, "vels")
            forces = self._ReadSimList3D(cGroup, "forces")

            frame.AddCell(pos, vels, forces, volumes[cellInd])

        return frame

    def ReadFrame(self, index):
        return self.GetFrame(index)

    def Frames(self, increment=1):
        for fInd in range(0, self.numFrames, increment):
            yield self.GetFrame(fInd)

    def dump(self):
        for f in self.Frames():
            print(f)
            for cell in f:
                print(cell)


    def _Close(self):
        if self.h5Handle is not None:
            self.h5Handle.close()
            self.h5Handle = None

    def __del__(self):
        self._Close()

    def __str__(self):
        return "Trajectory {}\nContaining {} frames." .format(self.trajPath,
                                                              self.numFrames)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._Close()



def main():
    """
    Function for testing Trajectory reader.
    """
    import argparse
    import pprint
    parser = argparse.ArgumentParser()
    parser.add_argument("filePath", help="CellSim3D trajectory file to test.")
    args = parser.parse_args()

    with Trajectory(args.filePath) as tr:
        tr.dump()


if __name__ == "__main__":
    main()

"""
This file contains tools to read CellSim3D files.
"""

import h5py
import os
import sys

class trajectory():
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
            self._ReadParams()

    def _ReadParams(self):
        for attr, val in self.h5Handle.attrs.items():
            print(attr, val)

    def __del__(self):
        if self.h5Handle is not None:
            print("Closing {}".format(self.trajName))
            self.h5Handle.close()
            self.h5Handle = None



def main():
    """
    Function for testing cell
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filePath", help="CellSim3D trajectory file to test.")
    args = parser.parse_args()

    print(args.filePath)
    trajectory(args.filePath)

if __name__ == "__main__":
    main()

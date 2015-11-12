#!/usr/bin/env python3
import bpy
import csv
import numpy as np
import sys

import celldiv

class TrajHandle(object):
    """ Class that opens and displays frames in a trajectory"""
    prevFrame = -1
    def __init__(self, fPath):
        self.filePath = fPath
        self.th = celldiv.TrajHandle(fPath)


    def DisplayFrame(self, frame: int = None, inc: int = None) -> None:
        if self.th is None:
            print("No trajectory was found at %s" % self.filePath)
        if frame is None:
            # go to next frame
            pass

    def close(self):
        bpy.ops.object.select_pattern(pattern='cellObj%d' % self.ind)
        bpy.ops.object.delete()

#!/usr/bin/env python3
import bpy
import csv
import numpy as np
import sys

import celldiv

class Visualizer(object):
    """ Class that opens and displays frames in a trajectory"""
    prevFrame = -1
    ind = 0
    def __init__(self, fPath):
        self.filePath = fPath
        self.th = celldiv.TrajHandle(fPath)
        Visualizer.ind += 1
        self.ind = Visualizer.ind
        self.pF = None

        bpy.data.worlds["World"].horizon_color=[0.051, 0.051, 0.051]
        bpy.data.scenes["Scene"].render.alpha_mode='SKY'

        self.firstfaces = []

        with open("/home/pranav/dev/celldiv/C180_pentahexa.csv", newline='') as g:
            readerfaces = csv.reader(g, delimiter=",")
            for row in readerfaces:
                self.firstfaces.append([int(v) for v in row])


    def DisplayFrame(self, frameNum: int = None, inc: int = None) -> None:
        if self.pF != frameNum:
            bpy.ops.object.select_pattern(pattern='cellObj')
            bpy.ops.object.delete()
        fr = self.th.ReadFrame(frameNum)
        fr = np.vstack(fr)
        fr -= np.mean(fr, axis=0)

        f = []
        for c in range(int(len(fr)/192)):
            for r in self.firstfaces:
                f.append([(v + c*192) for v in r])


        mesh = bpy.data.meshes.new('cellMesh')
        ob = bpy.data.objects.new('cellObject', mesh)

        bpy.context.scene.objects.link(ob)
        mesh.from_pydata(fr, [], f)
        mesh.update()

        bpy.ops.object.select_by_type(type='MESH')
        bpy.context.scene.objects.active = bpy.data.objects['cellObject']
        #bpy.context.scene.objects.active = bpy.data.objects['Cube']
        bpy.ops.object.make_links_data(type='MATERIAL')
        bpy.ops.object.select_all(action='TOGGLE')

        self.pF = self.th.currFrameNum

    def close(self):
        bpy.ops.object.select_pattern(pattern='cellObj%d' % self.ind)
        bpy.ops.object.delete()
        Visualizer.ind -=1
        self.th.close()

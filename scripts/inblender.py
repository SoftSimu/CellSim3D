#!/usr/bin/env python3
import bpy
import csv

class TrajHandle(object):
    """ Class that opens and displays frames in a trajectory"""
    ind = 0
    def __init__(self, filePath):
        TrajHandle.ind += 1
        self.ind = TrajHandle.ind
        self.f = open(filePath)
        self.firstfaces = []
        self.frameCnt = 0
        self.prevFrame = 0

        bpy.data.worlds["World"].horizon_color=[0.051, 0.051, 0.051]
        bpy.data.scenes["Scene"].render.alpha_mode='SKY'

        with open('/home/pranav/dev/celldiv/C180_pentahexa.csv', "r", newline='') as g:
            readerfaces = csv.reader(g, delimiter=',')
            for row in readerfaces:
                self.firstfaces.append([int(v) for v in row])

    def DisplayFrame(self, frame: int = None, inc: int = 0) -> None:
        fH = self.f
        pF = self.prevFrame
        skipFrames = 0

        if frame is not None:
            frame += 1
            if pF > frame:
                fH.seek(0)
                skipFrames = frame - 1
            elif pF < frame:
                skipFrames = frame - pF - 1
            self.prevFrame = frame
        else:
             self.prevFrame += 1

        for fr in range(skipFrames):
            nAtoms = int(fH.readline().strip())
            step = fH.readline().strip()
            print("Skipping %s" % step)
            for i in range(nAtoms):
                next(fH)

        if pF != frame:
            bpy.ops.object.select_pattern(pattern="cellObj%d" % self.ind)
            bpy.ops.object.delete()
            verts = []
            nAtoms = int(fH.readline().strip())
            self.step = fH.readline()
            for i in range(nAtoms):
                line = fH.readline().strip().split(",  ")
                verts.append([float(line[n]) for n in range(3)])

            print(nAtoms)
            nCells = int(nAtoms/192)
            faces = []

            for c in range(nCells):
                for row in self.firstfaces:
                    faces.append([(v+c*192) for v in row])

            mesh = bpy.data.meshes.new("cellMesh%d" % self.ind)
            ob = bpy.data.objects.new("cellObj%d" % self.ind, mesh)

            bpy.context.scene.objects.link(ob)
            mesh.from_pydata(verts, [], faces)
            mesh.update()

        print("now on %s" % self.step)





    def close(self):
        bpy.ops.object.select_pattern(pattern='cellObj%d' % self.ind)
        bpy.ops.object.delete()
        self.f.close()

import bpy
import csv
import sys
import os
import argparse
import numpy as np

sys.path.append("/home/pranav/dev/celldiv/scripts")
import celldiv


argv = sys.argv

if "--" not in argv:
    print("ERROR: No arguments provided to script")
    sys.exit(80)
else:
    a = argv.index("--")
    argv = argv[a + 1:]

helpString = """
Run as:
blender --background --python %s --
[options]
""" % __file__

parser = argparse.ArgumentParser(description=helpString)

parser.add_argument("trajPath", type=str,
                    help="Trajectory path. Absolute or relative.")

parser.add_argument("-s", "--smooth", action='store_true',
                    help="Do smoothing (really expensive and doesn't look as good)")

parser.add_argument("-k", "--skip", type=int, required=False,
                    help="Trajectory frame skip rate. E.g. SKIP=10 will only \
                    render every 10th frame.",
                    default=1)

parser.add_argument("-nc", "--noclear", type=bool, required=False,
                    help="specifying this will not clear the destination directory\
                    and restart rendering.",
                    default=False)

parser.add_argument("--min-cells", type=int, required=False,
                    help='Start rendering when system has at least this many cells',
                    default=1)

parser.add_argument("--inds", type=int, required=False, nargs='+',
                    help="Only render cells with these indices",
                    default=[])

parser.add_argument("-nf", "--num-frames", type=int, required=False,
                    help="Only render this many frames.",
                    default=sys.maxsize)

parser.add_argument("-r", "--res", type=int, default=1, required=False,
                    help='Renders images with resolution RES*1080p. RES>=1. \
                    Use 2 for 4k. A high number will devour your RAM.')

parser.add_argument("-cc", "--cell-color", type=int, nargs=3, required=False,
                    default=[72, 38, 153],
                    help="RGB values of cell color. From 0 to 255")

parser.add_argument("-bc", "--background-color", type=int, nargs=3,
                    required=False, default=[255,255,255],
                    help="RGB values of cell color. From 0 to 255")

parser.add_argument("-si", "--specular-intensity", type=float, required=False,
                    default = 0.0,
                    help="Set specular-intensity (shininess). From 0.0 to 1.0")

args = parser.parse_args(argv)

imageindex = 0
firstfaces = []
bpy.data.worlds["World"].horizon_color=[ (1.0/255.0)*c for c in args.background_color]

bpy.data.scenes["Scene"].render.alpha_mode='SKY'

doSmooth = args.smooth
if doSmooth:
    print("Doing smoothing. Consider avoiding this feature...")

if (args.res < 1):
    print("ERROR: invalid resolution factor")
    sys.exit()

bpy.data.scenes["Scene"].render.resolution_x*=args.res
bpy.data.scenes["Scene"].render.resolution_y*=args.res

with open('C180_pentahexa.csv', newline='') as g:
    readerfaces = csv.reader(g, delimiter=',')
    for row in readerfaces:
        firstfaces.append([int(v) for v in row])


filename = os.path.realpath(args.trajPath)
basename = os.path.splitext(filename)[0] + "/images/CellDiv_"

nSkip = args.skip

if nSkip > 1:
    print("Rendering every %dth" % nSkip, "frame...")


noClear = args.noclear

sPath = os.path.splitext(filename)[0] + "/images/"

if not noClear and os.path.exists(sPath):
    for f in os.listdir(sPath):
        os.remove(sPath+f)

cellInds = []
minInd = args.min_cells - 1
if len(args.inds) > 0:
    minInd = max(minInd, min(args.inds))

stopAt = args.num_frames

# Set material color
bpy.data.materials['Material'].diffuse_color = [ (1/255.0) * c for c in args.cell_color]
bpy.data.materials['Material'].specular_intensity = args.specular_intensity

# reset camera
# bpy.data.objects["Camera"].location=(0.0, 0.0, 0.0)
# bpy.data.objects["Camera"].rotation_euler=(0, 0, 0)
# camConsty=np.tan(0.5*bpy.data.cameras["Camera"].angle_y)
# camConstx=np.tan(0.5*bpy.data.cameras["Camera"].angle_x)
# cps=[]
with celldiv.TrajHandle(filename) as th:
    frameCount = 0
    try:
        for i in range(int(th.maxFrames/nSkip)):

            frameCount += 1
            if frameCount > args.num_frames:
                break

            f = th.ReadFrame(inc=nSkip)

            if len(f) < minInd+1:
                print("Only ", len(f), "cells in frame ", th.currFrameNum,
                      " skipping...")
                continue

            if len(args.inds) > 0:
                f = [f[a] for a in args.inds]

            f = np.vstack(f)
            # adjust to CoM
            f -= np.mean(f, axis=0)

            # set camera
            # camPos = max(np.abs(f[:, 0]).max()/camConstx, np.abs(f[:, 1]).max()/camConsty)*1.1
            # cps.append(camPos)
            # bpy.data.objects["Camera"].location=(0, 0, camPos)

            faces = []
            for mi in range(int(len(f)/192)):
                for row in firstfaces:
                    faces.append([(v+mi*192) for v in row])


            mesh = bpy.data.meshes.new('cellMesh')
            ob = bpy.data.objects.new('cellObject', mesh)

            bpy.context.scene.objects.link(ob)
            mesh.from_pydata(f, [], faces)
            mesh.update()

            if doSmooth:
                bpy.ops.object.select_by_type(type='MESH')
                bpy.context.scene.objects.active = bpy.data.objects['cellObject']
                bpy.ops.object.editmode_toggle()
                bpy.ops.mesh.normals_make_consistent(inside=False)
                bpy.ops.object.editmode_toggle()
                bpy.ops.object.shade_smooth()
                bpy.context.scene.objects.active = bpy.data.objects['Cube']
                bpy.ops.object.make_links_data(type='MATERIAL')             # copy material from Cube
                bpy.context.scene.objects.active = bpy.data.objects['cellObject']
                bpy.ops.object.select_all(action='TOGGLE')
                bpy.ops.object.modifier_add(type='SUBSURF')

            bpy.ops.object.select_by_type(type='MESH')
            bpy.context.scene.objects.active = bpy.data.objects['cellObject']
            bpy.context.scene.objects.active = bpy.data.objects['Cube']
            bpy.ops.object.make_links_data(type='MATERIAL')
            bpy.ops.object.select_all(action='TOGGLE')

            imagename = basename + "%d.png" % frameCount
            bpy.context.scene.render.filepath = imagename

            bpy.ops.render.render(write_still=True)  # render to file

            bpy.ops.object.select_pattern(pattern='cellObject')
            bpy.ops.object.delete()                                     # delete mesh...


    except celldiv.IncompleteTrajectoryError:
        print ("Stopping...")

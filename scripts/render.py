import bpy
import csv
import sys
import os
import argparse
import numpy as np

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

parser.add_argument("--only-frame", type=int, required=False, default = -1,
                    help="Only render this frame.")

args = parser.parse_args(argv)

imageindex = 0
firstfaces = []
#bpy.data.worlds["World"].horizon_color=[ c/256.0 for c in args.background_color]
#bpy.data.worlds["World"].horizon_color="{}{}{}".format(args.background_color[0], args.background_color[1], args.background_color[2])

#bpy.data.scenes["Scene"].render.alpha_mode='TRANSPARENT'
bpy.data.scenes["Scene"].render.image_settings.color_mode="RGBA"
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
#bpy.data.materials['Material'].diffuse_color = [ (1/255.0) * c for c in args.cell_color]
bpy.data.materials['Material'].specular_intensity = args.specular_intensity

with celldiv.TrajHandle(filename) as th:
    frameCount = 1
    try:
        for i in range(int(th.maxFrames/nSkip)):

            if frameCount > args.num_frames:
                break

            if i <= args.only_frame:
                print("Not there yet")
                continue

            f = th.ReadFrame(inc=nSkip)

            if len(f) < minInd+1:
                print("Only ", len(f), "cells in frame ", th.currFrameNum,
                      " skipping...")
                continue

            if len(args.inds) > 0:
                f = [f[a] for a in args.inds]

            f = np.vstack([c[:180] for c in f])

            # adjust to CoM
            f -= np.mean(f, axis=0)

            faces = []
            for mi in range(int(len(f)/180)):
                for row in firstfaces:
                    faces.append([(v+mi*180) for v in row])


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
                bpy.ops.object.make_links_data(type='Material')             # copy material from Cube
                bpy.context.scene.objects.active = bpy.data.objects['cellObject']
                bpy.ops.object.select_all(action='TOGGLE')
                bpy.ops.object.modifier_add(type='SUBSURF')

            bpy.ops.object.select_by_type(type='MESH')
            bpy.context.scene.objects.active = bpy.data.objects['cellObject']
            bpy.context.scene.objects.active = bpy.data.objects['Cube']
            bpy.ops.object.make_links_data(type='MATERIAL')
            bpy.ops.object.select_all(action='TOGGLE')

            # for p in f:
            #     bpy.ops.mesh.primitive_uv_sphere_add(location=list(p), size=0.01)
            #     bpy.context.scene.objects.active = bpy.data.objects['Cube']
            #     bpy.ops.object.make_links_data(type='MATERIAL')
            #     bpy.ops.object.select_all(action='TOGGLE')

            bpy.ops.object.constraint_add(type="TRACK_TO")
            bpy.context.object.constraints["Track To"].target = bpy.data.objects["cellObject"]

            bpy.context.object.constraints["Track To"].track_axis = "TRACK_Z"
            bpy.context.object.constraints["Track To"].up_axis = "UP_Z"






            imagename = basename + "%d.png" % frameCount
            bpy.context.scene.render.filepath = imagename

            bpy.ops.render.render(write_still=True)  # render to file

            #bpy.ops.constraint.delete()

            bpy.ops.object.select_pattern(pattern='cellObject')
            bpy.ops.object.delete()                                     # delete mesh...
            frameCount += 1

    except celldiv.IncompleteTrajectoryError:
        print ("Stopping...")

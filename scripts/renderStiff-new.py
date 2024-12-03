import bpy
import csv
import sys
import os
import argparse
import numpy as np

sys.path.append("/home/mahmood/Desktop/CellSim/scripts")
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
                    default=[82, 38, 123],
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

# Get the active world
world = bpy.context.scene.world

# Set the background color
world.use_nodes = True
bg_node = world.node_tree.nodes['Background']
bg_node.inputs['Color'].default_value = [(1.0/255.0)*c for c in args.background_color] + [1.0]

bpy.context.view_layer.use_sky = True



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
bpy.data.materials['Material'].diffuse_color = [ (1/255.0) * c for c in args.cell_color] + [1.0]
bpy.data.materials['Material'].specular_intensity = args.specular_intensity



with celldiv.TrajHandle(filename) as th:
    frameCount = 1
    try:
        for i in range(int(th.maxFrames/nSkip)):

            if frameCount > args.num_frames:
                break


            f = th.ReadFrame(inc=nSkip)
            m = np.vstack(th.cellInd)
            s1 = np.where(m < 0)[0]
            s2 = np.where(m >= 0 )[0]
            h = [ f[i] for i in s1 ]
            g = [ f[i] for i in s2 ]


            if len(f) < minInd+1:
                print("Only ", len(f), "cells in frame ", th.currFrameNum,
                      " skipping...")
                continue

            if len(args.inds) > 0:
                f = [f[a] for a in args.inds]

            f = np.vstack([c[:180] for c in f])
            if len(h) > 0:
                h = np.vstack([c[:180] for c in h])
            if len(g) > 0:
                g = np.vstack([c[:180] for c in g])
            

            faces1 = []
            for mi in range(int(len(h)/180)):
                for row in firstfaces:
                    faces1.append([(v+mi*180) for v in row])


            mesh = bpy.data.meshes.new('cellMesh1')
            ob1 = bpy.data.objects.new('cellObject1', mesh)
            mat1 = bpy.data.materials.new(name="MATERIAL1")
            mat1.diffuse_color = [82/255, 38/255, 123/255, 0.5]
            mat1.specular_intensity = 0.3
            ob1.data.materials.append(mat1)
            
            bpy.context.scene.collection.objects.link(ob1) 
            mesh.from_pydata(h, [], faces1)
            mesh.update()

            bpy.ops.object.select_by_type(type='MESH')
            bpy.context.view_layer.objects.active = ob1
            bpy.context.view_layer.objects.active = bpy.data.objects['Cube']  
            bpy.ops.object.select_all(action='TOGGLE')
            
            
            faces2 = []
            for mi in range(int(len(g)/180)):
                for row in firstfaces:
                    faces2.append([(v+mi*180) for v in row])
            
            
            
            mesh = bpy.data.meshes.new('cellMesh2')
            ob2 = bpy.data.objects.new('cellObject2', mesh)
            mat2 = bpy.data.materials.new(name="Material2")
            mat2.diffuse_color = [202/255, 108/255, 40/255, 0.5]
            mat2.specular_intensity = 0.3
            ob2.data.materials.append(mat2)
            

            bpy.context.scene.collection.objects.link(ob2)
            mesh.from_pydata(g, [], faces2)
            mesh.update()
                    
            bpy.ops.object.select_by_type(type='MESH')
            bpy.context.view_layer.objects.active = ob2
            bpy.context.view_layer.objects.active = bpy.data.objects['Cube']
            bpy.ops.object.select_all(action='TOGGLE')
            

            imagename = basename + "%d.png" % frameCount
            bpy.context.scene.render.filepath = imagename

            bpy.ops.render.render(write_still=True)  # render to file

            bpy.ops.object.select_pattern(pattern='cellObject1')
            bpy.ops.object.delete()                                     # delete mesh...
            bpy.ops.object.select_pattern(pattern='cellObject2')
            bpy.ops.object.delete()
            
            frameCount += 1
            


    except celldiv.IncompleteTrajectoryError:
        print ("Stopping...")

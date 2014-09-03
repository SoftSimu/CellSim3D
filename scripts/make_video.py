#!/usr/bin/python
import bpy 
import csv

frames = 0
line = []
numVerts = 0
numCells = 0
vertsPerCell = 192
verts = []
with open ('/home/pranav/Remotes/office/dev/CellDivision/dev/test.xyz', 'r') as trajFile:
    line = trajFile.readline()
    while (line != ""):
        if not "," in line and not ":" in line:
            frames += 1
            line = line.strip()
            numVerts = int(line)
            numCells = int(numVerts/vertsPerCell)
           
        line = trajFile.readline()
        for linNum in range(numVerts):
            line = trajFile.readline()
            line = line.strip()
            line = line.split(',')
            verts.append([float(v) for v in line])
        line = trajFile.readline()

firstfaces = []
faces = []



with open('/home/pranav/Remotes/office/dev/CellDivision/dev/C180_pentahexa.csv', 'r') as faceFile:
    faceVerts = csv.reader(faceFile, delimiter=',')
    for row in faceVerts:
        firstfaces.append([int(v) for v in row])

    for cellIndex in range(numCells):
        for row in firstfaces:
            faces.append([(v+(cellIndex*192)) for v in row])


mesh = bpy.data.meshes.new('cellMesh')
ob = bpy.data.objects.new('cellObject', mesh)
bpy.context.scene.objects.link(ob)
mesh.from_pydata(verts, [], faces)
mesh.update()
original_type = bpy.context.area.type
bpy.context.area.type = "VIEW_3D"                           # switch to view3d context
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
#    bpy.ops.render.render()                                    # ...if you want to render on screen
bpy.context.area.type = original_type 

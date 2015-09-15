import bpy
import csv
import sys

# This script presupposes a scene with camera, lights, and mesh object named Cube.
# Cube material is copied to fullerene cell meshes. Current settings are used for rendering.

imageindex = 0
f = open(sys.argv[6])
while True:
# read and append fullerene cell vertex coords for each time step
    imageindex += 1                 # current time step
    line = f.readline()             # first line of the time step
    if line == '':                  # when very last line of the file has been reached
        break                       # script terminates
    line = line.strip()             # remove new line (next line char)
    cell_length = int(line)         # number of lines to read for current time step
    verts = []
    next(f)                         # skip one irrelevant info line
    for c in range(int(cell_length/192)):
#        next(f)                     # skip cell index
        for i in range(192):    # append vertex xyz coords line by line
            line = f.readline()
            line = line.strip()
            line = line.split(',')
            verts.append([float(line[i]) for i in range(3)])

# read and append fullerene cell vertex indexes making up face polygons
    firstfaces = []
    faces = []
    kpl = int(len(verts)/192)       # number of fullerene cells in current time step
    with open('C180_pentahexa.csv', newline='') as g:
        readerfaces = csv.reader(g, delimiter=',')
        for row in readerfaces:
            firstfaces.append([int(v) for v in row])
        for moduloindex in range(kpl):
            for row in firstfaces:
                faces.append([(v+moduloindex*192) for v in row])

#    print(kpl)                      # just checking...
#    print(verts[0])                 # just checking...
#    print(imageindex)               # just checking...

# define and build up mesh for current time stepp
    mesh = bpy.data.meshes.new('cellMesh')
    ob = bpy.data.objects.new('cellObject', mesh)
    bpy.context.scene.objects.link(ob)
    mesh.from_pydata(verts,[],faces)
    mesh.update()
# fix mesh normals, smooth it, and assign material
#    original_type = bpy.context.area.type
#    bpy.context.area.type = "VIEW_3D"                           # switch to view3d context
    # bpy.ops.object.select_by_type(type='MESH')
    # bpy.context.scene.objects.active = bpy.data.objects['cellObject']
    # bpy.ops.object.editmode_toggle()
    # bpy.ops.mesh.normals_make_consistent(inside=False)
    # bpy.ops.object.editmode_toggle()
    # bpy.ops.object.shade_smooth()
    # bpy.context.scene.objects.active = bpy.data.objects['Cube']
    # bpy.ops.object.make_links_data(type='MATERIAL')             # copy material from Cube
    # bpy.context.scene.objects.active = bpy.data.objects['cellObject']
    # bpy.ops.object.select_all(action='TOGGLE')
    # bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.data.worlds["World"].horizon_color=[0.051, 0.051, 0.051]
    bpy.data.scenes["Scene"].render.alpha_mode='SKY'
#    bpy.ops.render.render()                                    # ...if you want to render on screen
#    bpy.context.area.type = original_type                       # switch back to text window

# save image indexed by current time step
    imagename = sys.argv[7]
    imagename +="/images/CellDiv_""%i"".png" % imageindex
    bpy.context.scene.render.filepath = imagename
    bpy.ops.render.render(write_still=True)  # render to file

    bpy.ops.object.select_pattern(pattern='cellObject')
    bpy.ops.object.delete()                                     # delete mesh...
#    if imageindex<11:                                          # ...or use this instead to leave
#        bpy.ops.object.delete()                                # mesh from last time step (11)

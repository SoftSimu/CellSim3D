import bpy
import csv

# This script presupposes a scene with camera, lights, and mesh object named Cube.
# Cube material is copied to fullerene cell meshes. Current settings are used for rendering.

imageindex = 0                  
f = open('/home/pranav/Remotes/office/dev/CellDivision/dev/traj.xyz')
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
    for i in range(cell_length):    # append vertex xyz coords line by line
        line = f.readline()
        line = line.strip()
        line = line.split(',  ')
        verts.append([float(v) for v in line])

# read and append fullerene cell vertex indexes making up face polygons
    firstfaces = []
    faces = []
    kpl = int(len(verts)/192)       # number of fullerene cells in current time step
    with open('/home/pranav/Remotes/office/dev/CellDivision/dev/C180_pentahexa.csv', newline='') as g:
        readerfaces = csv.reader(g, delimiter=',')
        for row in readerfaces:
            firstfaces.append([int(v) for v in row])
        for moduloindex in range(kpl):
            for row in firstfaces:
                faces.append([(v+moduloindex*192) for v in row])

#    print(kpl)                      # just checking...
#    print(verts[0])                 # just checking...
#    print(imageindex)               # just checking...

# define and build up mesh for current time step
    mesh = bpy.data.meshes.new('cellMesh')
    ob = bpy.data.objects.new('cellObject', mesh)
    bpy.context.scene.objects.link(ob)
    mesh.from_pydata(verts,[],faces)
    mesh.update()

# save image indexed by current time step    
#    imagename = "/home/pranav/Remotes/office/dev/CellDivision/dev/testcells_""%i"".png" % imageindex
#    bpy.context.scene.render.filepath = imagename
#    bpy.ops.render.render(write_still=True, use_viewport=True)  # render to file
        
    bpy.ops.object.select_pattern(pattern='cellObject') 
#    bpy.ops.object.delete()                                     # delete mesh...
#    if imageindex<11:                                          # ...or use this instead to leave
#        bpy.ops.object.delete()                                # mesh from last time step (11)



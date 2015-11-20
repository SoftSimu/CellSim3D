import bpy
import csv
import sys

sys.path.append("/home/pranav/dev/celldiv/scripts")
import celldiv

imageindex = 0
firstfaces = []
bpy.data.worlds["World"].horizon_color=[0.051, 0.051, 0.051]
bpy.data.scenes["Scene"].render.alpha_mode='SKY'

doSmooth = sys.argv[8].lower() == "smooth"

with open('C180_pentahexa.csv', newline='') as g:
    readerfaces = csv.reader(g, delimiter=',')
    for row in readerfaces:
        firstfaces.append([int(v) for v in row])

basename = sys.argv[7] + "/images/CellDiv_"


with celldiv.TrajHandle(sys.argv[6]) as th:
    for i in range(th.maxFrames):
        f = th.ReadFrame().tolist()

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

        imagename = basename + "%d.png" % th.currFrameNum
        bpy.context.scene.render.filepath = imagename

        bpy.ops.render.render(write_still=True)  # render to file

        bpy.ops.object.select_pattern(pattern='cellObject')
        bpy.ops.object.delete()                                     # delete mesh...

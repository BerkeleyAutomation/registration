import numpy as np
import trimesh
from visualization import Visualizer3D as vis
from collections import Counter
from meshrender import ViewsphereDiscretizer


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b

vsp_cfg = {
    'radius': {
        'min' : 0.5,
        'max' : 0.5,
        'n'   : 1
    },
    'azimuth': {
        'min' : 0.0,
        'max' : 360.0,
        'n'   : 10
    },
    'elev': {
        'min' : 0.0,
        'max' : 180.0,
        'n'   : 10
    },
    'roll': {
        'min' : 0.0,
        'max' : 0.0,
        'n'   : 1
    },
}

x = trimesh.load_mesh('./data/bar_clamp.obj')
x = trimesh.load_mesh('./data/73061.obj')
x.apply_translation(-x.center_mass)

edges = x.face_adjacency
verts = x.vertices[x.face_adjacency_edges] 
midpoints = (verts[:,0] + verts[:,1]) / 2.0
n1 = x.face_normals[edges][:,0]
n2 = x.face_normals[edges][:,1]

intersect_counter = np.zeros(midpoints.shape[0])

#views = [np.array([100,100,100]), np.array([100, 0 , 100]), np.array([0,100,100])]

view_tfs = ViewsphereDiscretizer.get_camera_poses(vsp_cfg, frame='obj')
views = [v.translation for v in view_tfs]

for view in views: 
    directions = view - midpoints
    d1 = np.einsum('ij,ij->i', n1, view-verts[:,0])
    d2 = np.einsum('ij,ij->i', n2, view-verts[:,1])
    m = d1*d2 < 0
    intersect_array = x.ray.intersects_any(midpoints + 1e-5*directions, directions)
    m = np.logical_and(m, np.logical_not(intersect_array))
    intersect_counter[m] += 1.0

maximum_count = np.amax(intersect_counter)
minimum_count = np.amin(intersect_counter)
print maximum_count
print minimum_count

vis.figure()
vis.mesh(x)
for i in range(len(verts)):
    rgb_color = rgb(minimum_count, maximum_count, intersect_counter[i])
    vis.plot3d(verts[i], color=rgb_color, tube_radius=0.001)
vis.show()





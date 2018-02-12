import numpy as np
import trimesh
from visualization import Visualizer3D as vis
from collections import Counter


def compute_midpoints(vertices):
	return (vertices[0,:] + vertices[1,:]) / 2



def rgb(minimum, maximum, value):
	minimum, maximum = float(minimum), float(maximum)
	ratio = 2 * (value-minimum) / (maximum - minimum)
	b = int(max(0, 255*(1 - ratio)))
	r = int(max(0, 255*(ratio - 1)))
	g = 255 - b - r
	return r, g, b

x = trimesh.load_mesh('./data/bar_clamp.obj')
x.apply_translation(-x.center_mass)

edges = x.face_adjacency
verts = x.vertices[x.face_adjacency_edges] 
midpoints = np.asarray(map(compute_midpoints, verts)) 

intersect_counter = np.zeros(midpoints.shape[0])

views = [np.array([100,100,100]), np.array([100, 0 , 100]), np.array([0,100,100])]

for view in views: 
	directions = view - midpoints
	intersect_array = x.ray.intersects_any(midpoints, directions)
	intersect_counter += intersect_array

maximum_count = np.amax(intersect_counter)
minimum_count = np.amin(intersect_counter)

vis.figure()
vis.mesh(x)
for i in range(len(verts)):
	rgb_color = rgb(maximum_count, minimum_count, intersect_counter[i])
	vis.plot3d(verts[i], color=rgb_color, tube_radius=0.001)
vis.show()





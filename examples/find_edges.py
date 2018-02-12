import numpy as np
import trimesh
from visualization import Visualizer3D as vis

x = trimesh.load_mesh('./data/bar_clamp.obj')
x = trimesh.load_mesh('./data/73061.obj')
x.apply_translation(-x.center_mass)

vec = np.random.multivariate_normal(np.zeros(3), np.eye(3))
vec = vec / np.linalg.norm(vec)

edges = x.face_adjacency
verts = x.vertices[x.face_adjacency_edges]
n1 = x.face_normals[edges][:,0]
n2 = x.face_normals[edges][:,1]
d1 = np.einsum('ij,ij->i', n1, vec-verts[:,0])
d2 = np.einsum('ij,ij->i', n2, vec-verts[:,1])
edge_inds = np.arange(len(edges))[d1*d2 < 0]

vis.figure()
vis.mesh(x)
vis.plot3d(np.vstack((vec * 0.10, vec * 0.09)), color=(0,0,1), tube_radius=0.003)
for edge_ind in edge_inds:
    verts = x.vertices[x.face_adjacency_edges[edge_ind]]
    vis.plot3d(verts, color=(0,1,0), tube_radius=0.001)
vis.show()

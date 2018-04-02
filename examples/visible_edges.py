import numpy as np
import trimesh
from visualization import Visualizer3D as vis
from collections import Counter
from meshrender import ViewsphereDiscretizer
import matplotlib.pyplot as plt

filepath = '/nfs/diskstation/projects/dex-net/objects/meshes/cleaned/thingiverse-pruned-meshes'

def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def sample_on_sphere(radius=0.5, num_points=100):
    points = []
    for i in range(num_points):
        sample = radius * np.random.normal(loc=0.0, scale=1.0, size=3)
	points.append(sample / np.linalg.norm(sample))
    return points


def plot_histogram(intersect_counter):
    n_bins = 10
    plt.hist(intersect_counter, bins=n_bins)
    plt.axvline(x=np.percentile(intersect_counter, 80))
    plt.show()


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

x = trimesh.load_mesh('./data/demon_helmet.obj')
# x = trimesh.load_mesh('./data/bar_clamp.off', process=False)
# x = trimesh.load_mesh('./data/73061.obj')
# x = trimesh.load_mesh('./data/2126220.obj')
# x = trimesh.load_mesh('./data/2677384.obj')
# x = trimesh.load_mesh('./data/4249857.obj')
# x = trimesh.load_mesh('./data/4470711.obj')
# x = trimesh.load_mesh('./data/90005.obj')

def compute_salient_edges(mesh, num_views=300):
    """Uses different views to find the salient images from a mes

    Parameters
    ----------
    mesh : trimesh.Trimesh
        the mesh to operate on
    num_views : int
        the number of camera positions that will be tested

    Returns
    -------
    a mask for ___ which identifies the important edges
    """

    x.apply_translation(-x.center_mass)

    edges = x.face_adjacency
    verts = x.vertices[x.face_adjacency_edges]
    midpoints = (verts[:,0] + verts[:,1]) / 2.0
    n1 = x.face_normals[edges][:,0]
    n2 = x.face_normals[edges][:,1]

    views = sample_on_sphere(num_points=num_views)

    visible_counter = np.zeros(midpoints.shape[0])
    salience_counter = np.zeros(midpoints.shape[0])

    for view in views: 
        directions = view - midpoints
        d1 = np.einsum('ij,ij->i', n1, view-verts[:,0])
        d2 = np.einsum('ij,ij->i', n2, view-verts[:,1])
        m = d1*d2 < 0
        
        intersect_array = x.ray.intersects_any(midpoints + 1e-5*directions, directions)
        visible_counter[np.logical_not(intersect_array)] += 1.0
        salience_counter[np.logical_and(m, np.logical_not(intersect_array))] += 1.0
    
        m = np.logical_and(m, np.logical_not(intersect_array))
        salience_counter[m] += 1.0
    

    visible_cutoff = np.percentile(visible_counter, 30)
    visible_counter[visible_counter < visible_cutoff] = 0
    edge_scores = np.divide(salience_counter, visible_counter, out=np.zeros_like(salience_counter), where=visible_counter!=0)
    edge_scores[edge_scores > 1.0] = 1.0

    return edge_scores >= 0.5


def visualize_salient_edges(mesh, edge_mask):
    """
    plots the identified important edges

    Parameters
    ----------
    mesh : trimesh.Trimesh
        the base mesh
    edge_mask : ndarray(num_edges,)
        a mask for ___ that identifies the important edges
    """

    # maximum_count = np.amax(edge_scores)
    # minimum_count = np.amin(edge_scores)
    verts = x.vertices[x.face_adjacency_edges]
    vis.figure()
    vis.mesh(x)
    for i in range(len(verts)):
        if edge_mask[i]:
            # rgb_color = rgb(minimum_count, maximum_count, edge_scores[i])
            vis.plot3d(verts[i], color=(1,0,0), tube_radius=0.001)
    vis.show()


if __name__ == "__main__":
    mask = compute_salient_edges(x)
    visualize_salient_edges(x, mask)



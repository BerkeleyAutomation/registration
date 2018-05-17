"""
Tools for computing, saving, and loading salient edge masks for objects.
Authors: Matt Matl and Amit Tajrela
"""
import numpy as np
import trimesh
import os
import shutil
from zipfile import ZipFile

from visualization import Visualizer3D as vis3d

class SalientEdgeSet(object):

    def __init__(self, mesh, config=None, edge_mask=None):
        """Create a SalientEdgeSet.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            A mesh corresponding to this edge set.
        config : autolab_core.YamlConfig
            A configuration for computing the salient edges.
            Required entries are given in the Other Parameters section below.
        edge_mask : (n,) bool
            A mask for the salient edges in mesh.edges_unique.

        Other Parameters
        ----------------
        num_views : int
            The number of viewpoints to use when computing edge saliency.
        saliency_threshold : float
            The threshold in [0,1] for n_times_vanishing / n_times_visible for
            an edge to be considered salient.
        """
        self._mesh = mesh
        self._config = config
        if self._config is None:
            self._config = {
                'num_views' : 500,
                'saliency_threshold' : 0.3
            }
        self._edge_mask = edge_mask
        if self._edge_mask is None:
            self._compute_salient_edges()

    @property
    def edge_mask(self):
        """(n,) bool: A mask for the salient edges of the mesh.

        Note
        ----
        Should be applied to either mesh.edges_unique or mesh.face_adjacency_edges.
        """
        return self._edge_mask

    @property
    def salient_edges(self):
        """(n,2) int: A list of edges (pairs of vertex indices) that are salient
        on the mesh.
        """
        return self._mesh.face_adjacency_edges[self.edge_mask]

    @property
    def mesh(self):
        """trimesh.Trimesh: The mesh associated with this edge set.
        """
        return self._mesh

    def visualize(self):
        """Visualize the salient edges of the mesh.
        """
        vis3d.figure()
        for edge in self.salient_edges:
            vis3d.plot3d(self.mesh.vertices[edge], color=(0.0, 1.0, 0.0), tube_radius=0.0005)
        vis3d.mesh(self.mesh)
        vis3d.show()

    def save(self, filename):
        """Save the salient edge mask out to a file.

        Parameters
        ----------
        filename : str
            The filename (.zip extension) to save the mesh and mask to.
        """
        direc, fn = os.path.split(filename)
        base, ext = os.path.splitext(fn)
        tmpdir = os.path.join(direc, base)
        os.makedirs(tmpdir)
        mesh_fn = os.path.join(tmpdir, 'mesh.obj')
        mask_fn = os.path.join(tmpdir, 'mask.npy')

        self.mesh.export(mesh_fn)
        np.save(open(mask_fn, 'w'), self.edge_mask)
        with ZipFile(filename, 'w') as zf:
            zf.write(mesh_fn, 'mesh.obj')
            zf.write(mask_fn, 'mask.npy')
        shutil.rmtree(tmpdir)

    def _compute_salient_edges(self):
        """Compute salient edges for a mesh.

        This works by:
            - Rendering the mesh from a variety of viewpoints.
            - Note which edges are visible, and of those, which are vanishing,
              which means that one face is facing the camera and one is facing away.
            - Compute the percentage of the time each edge is vanishing when it is visible.
            - Edges with this percentage over a certain threshold (defined in the config)
              are considered salient.
        """
        # Compute a set of viewpoints, uniformly sampled on a sphere about the object
        num_views = self._config['num_views']
        radius = 2.0 * np.max(self.mesh.extents)
        center = np.mean(self.mesh.bounds)
        camera_centers = SalientEdgeSet._sample_on_sphere(center, radius, num_views)

        # Find midpoints of edges and adjoining surface normals
        edges = self.mesh.face_adjacency_edges
        verts = self.mesh.vertices[edges]
        midpoints = (verts[:,0] + verts[:,1]) / 2.0
        normals = self.mesh.face_normals[self.mesh.face_adjacency]
        n1 = normals[:,0]
        n2 = normals[:,1]

        # Keep tabs on how often each edge is salient and how often each edge is visible
        salient_counts = np.zeros(len(edges), dtype=np.int)
        visible_counts = np.zeros(len(edges))

        for cc in camera_centers:
            view_vecs = midpoints - cc

            dots_1 = np.einsum('ij,ij->i', n1, view_vecs)
            dots_2 = np.einsum('ij,ij->i', n2, view_vecs)
            is_vanishing = (dots_1 * dots_2) < 0
            is_visible = np.logical_not(self.mesh.ray.intersects_any(midpoints - 1e-5 * view_vecs, -view_vecs))
            is_salient = np.logical_and(is_vanishing, is_visible)
            visible_counts[is_visible] += 1
            salient_counts[is_salient] += 1

        edge_scores = np.divide(salient_counts, visible_counts,
            out=np.zeros_like(visible_counts), where=(visible_counts != 0))

        self._edge_mask = edge_scores >= self._config['saliency_threshold']

    @staticmethod
    def load(filename):
        """Load a SalientEdgeSet from a .pkl file.

        Parameters
        ----------
        filename : str
            The .pkl file containing the edge set.
        """
        direc, fn = os.path.split(filename)
        base, ext = os.path.splitext(fn)
        tmpdir = os.path.join(direc, base)
        os.makedirs(tmpdir)
        mesh_fn = 'mesh.obj'
        mask_fn = 'mask.npy'

        with ZipFile(filename, 'r') as zf:
            zf.extract(mesh_fn, tmpdir)
            zf.extract(mask_fn, tmpdir)

        mesh = trimesh.load_mesh(os.path.join(tmpdir, mesh_fn))
        mask = np.load(open(os.path.join(tmpdir, mask_fn)))
        shutil.rmtree(tmpdir)

        return SalientEdgeSet(mesh, edge_mask=mask)

    @staticmethod
    def _sample_on_sphere(center, radius, n_samples=1):
        """Sample n_samples points on a sphere defined by center and radius.
        """
        us = np.random.uniform(-1, 1, size=n_samples)
        thetas = np.random.uniform(0, 2*np.pi, size=n_samples)
        inter = np.sqrt(1.0 - us*us)
        xs = inter * np.cos(thetas)
        ys = inter * np.sin(thetas)
        zs = us

        return np.c_[xs, ys, zs] + center

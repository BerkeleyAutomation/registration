import os
import subprocess

import numpy as np
import trimesh
from plyfile import PlyElement, PlyData

from autolab_core import RigidTransform, PointCloud
from visualization import Visualizer3D as vis

from .data_generator import PointCloudGenerator

class Super4PCSAligner(object):

    def __init__(self, config):
        """Initialize a Super4PCSAligner.

        Parameters
        ----------
        config : autolab_core.YamlConfig
            Config containing information for parameterizing Super4PCS

        Other Parameters
        ----------------
        overlap : float
            The expected overlap between future point clouds in [0, 1]
        accuracy : float
            The distance between two points for them to be considered aligned (in meters).
        samples : int
            The number of samples to take when using Super4PCS -- lower numbers are faster
            but potentially less accurate.
        timeout : int
            The number of seconds to try to align for at a maximum
        cache_dir : str
            A cache directory for the Super4PCSAligner.
        """
        self._overlap = config['overlap']
        self._accuracy = config['accuracy']
        self._samples = config['samples']
        self._timeout = config['timeout']
        self._cache_dir = config['cache_dir']
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)
        self._points1_fn = os.path.join(self._cache_dir, 'points1.ply')
        self._points2_fn = os.path.join(self._cache_dir, 'points2.ply')
        self._tf_fn = os.path.join(self._cache_dir, 'tf.txt')

    def align(self, points1, points2):
        """Compute an aligning transform between two point clouds.

        Parameters
        ----------
        points1 : autolab_core.PointCloud
            The first point cloud.
        points2 : autolab_core.PointCloud
            The second point cloud.

        Returns
        -------
        autolab_core.RigidTransform
            A Rigid Tranformation taking points in the second cloud
            to the same frame as the first cloud.
        """

        # Export each point cloud as a .ply file
        data = np.array([(x[0], x[1], x[2]) for x in points1.data.T], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(data, 'vertex')
        PlyData([el], text=True).write(self._points1_fn)

        data = np.array([(x[0], x[1], x[2]) for x in points2.data.T], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(data, 'vertex')
        PlyData([el], text=True).write(self._points2_fn)

        # Run Super4PCS on those
        subprocess.call(["Super4PCS", "-i", self._points1_fn, self._points2_fn,
                                      "-o", str(self._overlap),
                                      "-d", str(self._accuracy),
                                      "-n", str(self._samples),
                                      "-t", str(self._timeout),
                                      "-m", self._tf_fn
                        ])

        # Read back in the transform
        mat = []
        with open(self._tf_fn) as f:
            for line in f:
                dat = line.split()
                try:
                    a = [float(dat[0]), float(dat[1]), float(dat[2]), float(dat[3])]
                    mat.append(a)
                except:
                    pass
        mat = np.array(mat)

        return RigidTransform(mat[:3,:3], mat[:3,3], from_frame=points2.frame, to_frame=points1.frame)


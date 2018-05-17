"""
Class for generating artificial examples for testing registration algorithms.
Authors: Matt Matl and Amit Tajrela
"""
import numpy as np
import trimesh
from autolab_core import RigidTransform, PointCloud
from meshrender import UniformPlanarWorksurfaceImageRandomVariable, Scene, SceneObject
from perception import RenderMode, CameraIntrinsics, BinaryImage
from scipy.spatial import KDTree
import dill

from visualization import Visualizer3D as vis3d
from visualization import Visualizer2D as vis2d

from .salient_edges import SalientEdgeSet

class RegistrationExample(object):

    def __init__(self, salient_edge_set_filename, depth_im, camera_intrs, T_obj_camera):
        """Initialize a RegistrationExample.

        Parameters
        ----------
        salient_edge_set_filename : str
            The canonical salient edge set's filename.
        depth_im : perception.DepthImage
            A rendered depth image containing the target object.
        camera_intrs : perception.CameraIntrinsics
            Camera intrinsics used to render the depth image.
        T_obj_camera : autolab_core.RigidTransform
            True pose of object in camera frame.
        """
        self._salient_edge_set_filename = salient_edge_set_filename
        self._depth_im = depth_im
        self._camera_intrs = camera_intrs
        self._T_obj_camera = T_obj_camera
        self._salient_edge_set = None

    @property
    def depth_im(self):
        return self._depth_im

    @property
    def camera_intrs(self):
        return self._camera_intrs

    @property
    def T_obj_camera(self):
        return self._T_obj_camera

    @property
    def salient_edge_set(self):
        if self._salient_edge_set is None:
            print self._salient_edge_set_filename
            self._salient_edge_set = SalientEdgeSet.load(self._salient_edge_set_filename)
        return self._salient_edge_set

    def save(self, filename):
        ses = self._salient_edge_set
        self._salient_edge_set = None
        dill.dump(self, open(filename, 'w'))
        self._salient_edge_set = ses

    def alignment_error(self, T_obj_camera_est, n_samples=10000):
        mesh = self.salient_edge_set.mesh
        sampled_points = mesh.sample(n_samples)

        true_points = self.T_obj_camera.apply(PointCloud(sampled_points.T, frame='obj')).data.T
        est_points  = T_obj_camera_est.apply(PointCloud(sampled_points.T, frame='obj')).data.T

        lookup_tree = KDTree(true_points)
        _, indices = lookup_tree.query(est_points)
        squared_err = np.linalg.norm(est_points - true_points[indices], axis=1)**2
        err = np.sum(squared_err)
        return err

    def visualize_alignment(self, T_obj_camera_est):
        mesh = self.salient_edge_set.mesh
        m_true = mesh.copy().apply_transform(self.T_obj_camera.matrix)
        m_est = mesh.copy().apply_transform(T_obj_camera_est.matrix)

        vis3d.figure()
        vis3d.mesh(m_true, color=(0.0, 1.0, 0.0))
        vis3d.mesh(m_est, color=(0.0, 0.0, 1.0))
        vis3d.show()

    @staticmethod
    def load(filename):
        return dill.load(open(filename))

class RegistrationExampleGenerator(object):

    def __init__(self, config):
        """Initialize a RegistrationExampleGenerator with a configuration file.

        Parameters
        ----------
        config : autolab_core.YamlConfig
            A config file for the generator. Required parameters are listed
            in the Other Parameters section.

        Other Parameters
        ----------------
        TODO
        """
        self._config = config

    def generate_examples(self, salient_edge_set_filename, n_samples=1):
        """Generate RegistrationExamples for evaluating the algorithm.

        Parameters
        ----------
        salient_edge_set_filename : str
            A file containing the salient edge set to generate images of.
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        list of RegistrationExample
            A list of RegistrationExamples.
        """
        # Compute stable poses of mesh
        salient_edge_set = SalientEdgeSet.load(salient_edge_set_filename)
        mesh = salient_edge_set.mesh

        stp_pose_tfs, probs = mesh.compute_stable_poses()
        probs = probs / sum(probs)

        # Generate n renders
        examples = []
        scene = Scene()
        so = SceneObject(mesh, RigidTransform(from_frame='obj', to_frame='world'))
        scene.add_object('object', so)

        for i in range(n_samples):
            # Sample random stable pose.
            tf_id = np.random.choice(np.arange(len(probs)), p=probs)
            tf = stp_pose_tfs[tf_id]
            T_obj_world = RigidTransform(tf[:3,:3], tf[:3,3], from_frame='obj', to_frame='world')
            so.T_obj_world = T_obj_world

            # Create the random uniform workspace sampler
            ws_cfg = self._config['worksurface_rv_config']
            uvs = UniformPlanarWorksurfaceImageRandomVariable('object', scene, [RenderMode.DEPTH], frame='camera', config=ws_cfg)

            # Sample and extract the depth image, camera intrinsics, and T_obj_camera
            sample = uvs.sample()
            depth_im = sample.renders[RenderMode.DEPTH]
            cs = sample.camera
            ci = CameraIntrinsics(frame='camera', fx=cs.focal, fy=cs.focal, cx=cs.cx, cy=cs.cy,
                                  skew=0.0, height=ws_cfg['im_height'], width=ws_cfg['im_width'])
            T_obj_camera = cs.T_camera_world.inverse().dot(T_obj_world)
            examples.append(RegistrationExample(salient_edge_set_filename, depth_im, ci, T_obj_camera))

        return examples

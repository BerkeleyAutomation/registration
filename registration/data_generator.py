import numpy as np
import trimesh
from autolab_core import RigidTransform
from meshrender import UniformPlanarWorksurfaceImageRandomVariable, Scene, SceneObject
from perception import RenderMode, CameraIntrinsics

from visualization import Visualizer3D as vis3d
from visualization import Visualizer2D as vis2d

class PointCloudGenerator(object):

    def __init__(self, ws_config):
        """Initialize a PointCloudGenerator with a worksurface config.
        
        Parameters
        ----------
        ws_config : autolab_core.YamlConfig
            A config file for the renderer's random variable.
        """
        self._ws_cfg = ws_config

    def generate_worksurface_point_clouds(self, mesh, n_samples, vis=False):
        """Generate point clouds of a mesh in isolation on a table.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh to generate data for.
        n_samples : int
            The number of samples to generate.
        vis : bool
            If true, each sample is visualized.

        Returns
        -------
        point_clouds : list of autolab_core.PointCloud
            The generated point clouds in camera coordinates.
        depth_images : list of perception.DepthImage
            The corresponding depth images.
        obj_to_cam_poses : list of autolab_core.RigidTransform
            The object-to-camera transforms.
        """
        # Compute stable poses of mesh
        stp_pose_tfs, probs = mesh.compute_stable_poses()
        probs = probs / sum(probs)

        # Generate n datapoints
        point_clouds, depth_images, obj_to_cam_poses = [], [], []
        for i in range(n_samples):
            # Sample a pose tf
            tf_id = np.random.choice(np.arange(len(probs)), p=probs)
            tf = stp_pose_tfs[tf_id]
            T_obj_world = RigidTransform(tf[:3,:3], tf[:3,3], from_frame='obj', to_frame='world')

            # Create the scene object
            so = SceneObject(mesh, T_obj_world)

            # Create the scene
            scene = Scene()
            scene.add_object('object', so)

            # Create the image generation random variable
            uvs = UniformPlanarWorksurfaceImageRandomVariable('object', scene, [RenderMode.DEPTH], frame='camera', config=self._ws_cfg)

            # Sample and extract depth and camera data
            sample = uvs.sample()
            depth_image = sample.renders[RenderMode.DEPTH]
            cs = sample.camera
            T_obj_camera = cs.T_camera_world.inverse().dot(T_obj_world)
            ci = CameraIntrinsics(frame='camera', fx=cs.focal, fy=cs.focal, cx=cs.cx, cy=cs.cy,
                                  skew=0.0, height=self._ws_cfg['im_height'], width=self._ws_cfg['im_width'])
            point_cloud = ci.deproject(depth_image)

            point_clouds.append(point_cloud)
            depth_images.append(depth_image)
            obj_to_cam_poses.append(T_obj_camera)

            if vis:
                vis2d.figure()
                vis2d.imshow(depth_image)
                vis2d.show()

                vis3d.figure()
                vis3d.mesh(mesh, T_obj_world)
                vis3d.points(cs.T_camera_world * point_cloud, scale=0.001, subsample=3, random=True)
                vis3d.pose(cs.T_camera_world)
                vis3d.show()

        return point_clouds, depth_images, obj_to_cam_poses

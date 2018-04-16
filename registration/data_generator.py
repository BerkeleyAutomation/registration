import numpy as np
import trimesh
from autolab_core import RigidTransform
from meshrender import UniformPlanarWorksurfaceImageRandomVariable, Scene, SceneObject
from perception import RenderMode, CameraIntrinsics, BinaryImage

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
    def generate_worksurface_point_clouds(self, mesh, edge_mask, n_samples, vis=False):
        """Generate point clouds of a mesh in isolation on a table.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh to generate data for.
        edge_mask : numpy.ndarray
            the edge mask generated from compute_salient_edges 
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
        cis : list of perception.CameraIntrinsics
            the intrinsic parameters for the camera in each view. 
        di_masks : list of perception.BinaryImage
            each element is a binary mask that identifies edges in the depth image
        """
        # Compute stable poses of mesh
        stp_pose_tfs, probs = mesh.compute_stable_poses()
        probs = probs / sum(probs)

        # Generate n datapoints
        point_clouds, depth_images, obj_to_cam_poses, cis, di_masks = [], [], [], [], []
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
            cis.append(ci)

            # Create a binary image rendering of the salient edges of the mesh
            di_mask = self._compute_edge_mask(mesh, edge_mask, ci, T_obj_camera, depth_image)
            di_mask = BinaryImage(di_mask * 255)
            di_masks.append(di_mask)
            #vis2d.figure()
            #vis2d.subplot(121)
            #vis2d.imshow(di_mask)
            #vis2d.subplot(122)
            #vis2d.imshow(depth_image)
            #vis2d.show()

            if vis:
                vis2d.figure()
                vis2d.imshow(depth_image)
                vis2d.show()

                vis3d.figure()
                vis3d.mesh(mesh, T_obj_world)
                vis3d.points(cs.T_camera_world * point_cloud, scale=0.001, subsample=3, random=True)
                vis3d.pose(cs.T_camera_world)
                vis3d.show()

        return point_clouds, depth_images, obj_to_cam_poses, cis, di_masks
    def _compute_edge_mask(self, mesh, edge_mask, ci, T_obj_camera, depth_im):

        # Allocate array for storing mask
        di_mask = np.zeros(depth_im.data.shape, dtype=np.uint8)

        # ---------------
        # Compute edge endpoint coordinates in camera frame and up/down biases for each
        # ---------------
        m = mesh.copy().apply_transform(T_obj_camera.matrix)
        vertex_inds = m.face_adjacency_edges[edge_mask]
        endpoints_3d = m.vertices[vertex_inds]

        # For each edge, compute the face that we should be sampling most closely against
        vecs = (endpoints_3d[:,0] + endpoints_3d[:,1]) / 2.0
        face_inds = m.face_adjacency[edge_mask]
        normals = m.face_normals[face_inds]
        dots = np.einsum('ijk,ik->ij', normals, vecs)
        valid = np.where(np.logical_not((dots > 0).all(1)))[0]
        vertex_inds = vertex_inds[valid]

        # Find "other" vertex per edge of interest
        for i in range(len(vertex_inds)):
            # Extract other vertex end
            endpoint_inds = vertex_inds[i]

            # Project vertices down to 2D
            endpoints = m.vertices[endpoint_inds]

            points = endpoints.T
            points_proj = ci._K.dot(points)
            point_depths = np.tile(points_proj[2,:], [3, 1])
            points_proj = np.divide(points_proj, point_depths).T[:,:2]
            point_depths = point_depths[0][:2]
            endpoints = points_proj[:2]

            delta = endpoints[1] - endpoints[0]
            dy = np.abs(delta[1])
            dx = np.abs(delta[0])

            depth_offset=5e-3

            # Move along y
            if dy > dx:
                order = np.argsort(endpoints[:,1])
                endpoints = endpoints[order]
                depths = point_depths[order]

                # x = my + b
                delta = endpoints[1] - endpoints[0]
                slope = delta[0] / delta[1]
                intercept = endpoints[0][0] - slope * endpoints[0][1]

                # Move along y axis
                y = int(endpoints[0][1]) + 1
                y_end = int(endpoints[1][1])

                while y <= y_end:
                    if y < 0 or y >= di_mask.shape[0]:
                        y += 1
                        continue

                    exp_x = slope * y + intercept
                    x = int(exp_x)
                    xs_to_attempt = [x, x+1, x-1, x+2, x-2]
                    exp_depth = ((y - endpoints[0][1]) / delta[1]) * (depths[1] - depths[0]) + depths[0]

                    for x in xs_to_attempt:
                        if x < 0 or x >= di_mask.shape[1]:
                            continue
                        depth = depth_im.data[y,x]
                        if np.abs(depth - exp_depth) < depth_offset:
                            di_mask[y][x] = 1
                            break
                    y += 1
            else:
                order = np.argsort(endpoints[:,0])
                endpoints = endpoints[order]
                depths = point_depths[order]

                # y = mx + b
                delta = endpoints[1] - endpoints[0]
                slope = delta[1] / delta[0]
                intercept = endpoints[0][1] - slope * endpoints[0][0]

                # Move along y axis
                x = int(endpoints[0][0]) + 1
                x_end = int(endpoints[1][0])

                while x <= x_end:
                    if x < 0 or x >= di_mask.shape[1]:
                        x += 1
                        continue

                    exp_y = slope * x + intercept
                    y = int(exp_y)
                    ys_to_attempt = [y, y+1, y-1, y+2, y-2]
                    exp_depth = ((x - endpoints[0][0]) / delta[0]) * (depths[1] - depths[0]) + depths[0]

                    for y in ys_to_attempt:
                        if y < 0 or y >= di_mask.shape[0]:
                            continue
                        depth = depth_im.data[y,x]
                        if np.abs(depth - exp_depth) < depth_offset:
                            di_mask[y][x] = 1
                            break
                    x += 1

            ## Move along y
            #if dy > dx:
            #    endpoints = endpoints[np.argsort(endpoints[:,1])]

            #    # x = my + b
            #    delta = endpoints[1] - endpoints[0]
            #    slope = delta[0] / delta[1]
            #    intercept = endpoints[0][0] - slope * endpoints[0][1]

            #    # Use "other" point to determine rounding
            #    exp_x = (other[1]) * slope + intercept
            #    x_offset = 1 if (other[0] > exp_x) else 0

            #    # Move along y axis
            #    y = int(endpoints[0][1]) + 1
            #    y_end = int(endpoints[1][1])

            #    while y <= y_end:
            #        exp_x = slope * y + intercept
            #        x = int(exp_x) + x_offset
            #        depth = depth_im.data[y,x]
            #        if depth >= depth_lb and depth <= depth_ub:
            #            di_mask[y][x] = 1
            #        y += 1
            #else:
            #    endpoints = endpoints[np.argsort(endpoints[:,0])]

            #    # y = mx + b
            #    delta = endpoints[1] - endpoints[0]
            #    slope = delta[1] / delta[0]
            #    intercept = endpoints[0][1] - slope * endpoints[0][0]

            #    # Use "other" point to determine rounding
            #    exp_y = (other[0]) * slope + intercept
            #    y_offset = 1 if (other[1] > exp_y) else 0

            #    # Move along y axis
            #    x = int(endpoints[0][0]) + 1
            #    x_end = int(endpoints[1][0])

            #    while x <= x_end:
            #        exp_y = slope * x + intercept
            #        y = int(exp_y) + y_offset
            #        depth = depth_im.data[y,x]
            #        if depth >= depth_lb and depth <= depth_ub:
            #            di_mask[y][x] = 1
            #        x += 1

        return di_mask


        

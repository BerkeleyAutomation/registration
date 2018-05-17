"""
Tools for generating depth images and corresponding salient
edge masks for training an edge detection network.
"""
import numpy as np
from autolab_core import RigidTransform
from meshrender import UniformPlanarWorksurfaceImageRandomVariable, Scene, SceneObject
from perception import RenderMode, CameraIntrinsics, BinaryImage

from visualization import Visualizer3D as vis3d
from visualization import Visualizer2D as vis2d

class SalientEdgeImageGenerator(object):

    def __init__(self, config):
        """Initialize a SalientEdgeImageGenerator with a configuration file.

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

    def generate_images(self, salient_edge_set, n_samples=1):
        """Generate depth image, normal image, and binary edge mask tuples.

        Parameters
        ----------
        salient_edge_set : SalientEdgeSet
            A salient edge set to generate images of.
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        depth_ims : (n,) list of perception.DepthImage
            Randomly-rendered depth images of object.
        normal_ims : (n,) list of perception.PointCloudImage
            Normals for the given image
        edge_masks : (n,) list of perception.BinaryImage
            Masks for pixels on the salient edges of the object.
        """
        # Compute stable poses of mesh
        mesh = salient_edge_set.mesh

        stp_pose_tfs, probs = mesh.compute_stable_poses()
        probs = probs / sum(probs)

        # Generate n renders
        depth_ims, normal_ims, edge_masks = [], [], []
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
            edge_mask = self._compute_edge_mask(salient_edge_set, depth_im, ci, T_obj_camera)
            point_cloud_im = ci.deproject_to_image(depth_im)
            normal_im =  point_cloud_im.normal_cloud_im()


            depth_ims.append(depth_im)
            normal_ims.append(normal_im)
            edge_masks.append(edge_mask)

        return depth_ims, normal_ims, edge_masks

    def _compute_edge_mask(self, salient_edge_set, depth_im, ci, T_obj_camera):
        """Compute the edge mask for a given salient edge set, depth image, camera intrinsics,
        and object-to-camera transform.
        """
        mesh = salient_edge_set.mesh
        edge_mask = salient_edge_set.edge_mask

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
                            di_mask[y][x] = 0xFF
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
                            di_mask[y][x] = 0xFF
                            break
                    x += 1

        return BinaryImage(di_mask)

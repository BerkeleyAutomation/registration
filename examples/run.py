import cv2
import numpy as np
import trimesh
from matplotlib import pyplot as plt
from autolab_core import RigidTransform, PointCloud
from perception import BinaryImage
from visualization import Visualizer3D as vis3d
from visualization import Visualizer2D as vis2d
from registration import PointCloudGenerator, Super4PCSAligner
from meshrender import ViewsphereDiscretizer

from visible_edges import compute_salient_edges

def sample_from_edges(x, sigma=0.0):
    vec = np.random.multivariate_normal(np.zeros(3), np.eye(3))
    vec = vec / np.linalg.norm(vec)

    edges = x.face_adjacency
    angles = x.face_adjacency_angles
    verts = x.vertices[x.face_adjacency_edges]
    n1 = x.face_normals[edges][:,0]
    n2 = x.face_normals[edges][:,1]
    d1 = np.einsum('ij,ij->i', n1, vec-verts[:,0])
    d2 = np.einsum('ij,ij->i', n2, vec-verts[:,1])
    edge_inds = np.arange(len(edges))[d1*d2 < 0]

    sampled_points = []
    for edge_ind in edge_inds:
        verts = x.vertices[x.face_adjacency_edges[edge_ind]]
        angle = angles[edge_ind]
        if angle < np.pi / 6:
            continue
        n_points = max(int(np.linalg.norm(verts[1]-verts[0]) / 0.001), 1)
        points = np.random.uniform(size=(n_points,))
        sampled_points.extend([verts[0] + (verts[1] - verts[0]) * p for p in points])
    sampled_points = np.array(sampled_points)
    sampled_points += np.random.normal(scale=sigma, size=sampled_points.shape)
    return sampled_points

def generate_canonical_pc(mesh, edge_mask, n_points=1000):
    edges = mesh.vertices[mesh.face_adjacency_edges][edge_mask]
    lengths = np.linalg.norm(edges[:,0] - edges[:,1], axis=1)
    ppe = np.array(lengths / np.sum(lengths) * n_points, dtype=np.int32) + 1

    sampled_points = []
    for edge, n_samples in zip(edges, ppe):
        points = np.random.uniform(size=(n_samples,))
        sampled_points.extend([edge[0] + (edge[1] - edge[0]) * p for p in points])
    sampled_points = np.array(sampled_points)
    return sampled_points

def run_test_cases(mesh, pcs_cfg, pcg_cfg, n_test_cases=1, vis=False):

    # Extract "salient" edges from the mesh, sample points from those edges exactly
    #edge_mask = get_edge_mask(mesh)
    #edge_pc_obj = sample_from_edges(mesh, edge_mask)
    edges = mesh.vertices[mesh.face_adjacency_edges]
    edge_mask = compute_salient_edges(mesh, 400)
    edge_pc_obj = PointCloud(generate_canonical_pc(mesh, edge_mask).T, frame='mesh')
    print edge_pc_obj.data.shape

    vis3d.figure()
    vis3d.points(edge_pc_obj, scale=0.001)
    vis3d.mesh(mesh)
    vis3d.show()

    pcg = PointCloudGenerator(pcg_cfg)
    aligner = Super4PCSAligner(pcs_cfg)

    # Render test cases
    pcs, dis, poses, cis = pcg.generate_worksurface_point_clouds(mesh, edge_mask, n_test_cases)

    return

    # For each test case
    for i in range(n_test_cases):
        pc_cam, depth_im, T_obj_camera, ci = pcs[i], dis[i], poses[i], cis[i]

        # TODO
        # Sample points from the salient edges of the transformed mesh with some sigma
        # Discard "invisible" points (i.e. edges below the surface of the object)

        # Try canny edge detector
        img_data = np.array(depth_im.data * 256, dtype=np.uint8)
        image_edges = cv2.Canny(img_data, 2, 30)

        if vis:
            vis2d.figure()
            vis2d.subplot(121)
            vis2d.imshow(depth_im)
            vis2d.subplot(122)
            plt.imshow(image_edges, cmap='gray')
            vis2d.show()

        mask = BinaryImage(image_edges)
        depth_im_edges = depth_im.mask_binary(mask)
        edge_pc_cam = ci.deproject(depth_im_edges)
        edge_pc_cam.remove_zero_points()
        print edge_pc_cam.data.shape

        if vis:
            vis3d.figure()
            vis3d.points(edge_pc_cam, scale=0.001)
            vis3d.show()

        # Align the two point sets with Super4PCS
        T_obj_camera_est = aligner.align(edge_pc_cam, edge_pc_obj)

        # Visualize the result
        true_mesh = mesh.copy()
        true_mesh.apply_transform(T_obj_camera.matrix)

        est_mesh = mesh.copy()
        est_mesh.apply_transform(T_obj_camera_est.matrix)

        vis3d.figure()
        vis3d.points(T_obj_camera_est.as_frames('mesh','camera') * edge_pc_obj, color=(0,0,1), scale=0.001)
        vis3d.points(edge_pc_cam, scale=0.001)
        vis3d.mesh(true_mesh, style='surface')
        vis3d.mesh(est_mesh, style='surface', color=(0,0,1))
        vis3d.show()



def main():
    pcs_cfg = {
        'overlap' : 0.65,
        'accuracy': 0.0005,
        'samples' : 2000,
        'timeout' : 3000,
        'cache_dir' : './.cache'
    }
    vsp_cfg = {
        'radius': {
            'min' : 0.3,
            'max' : 0.3,
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
    pcg_cfg = {
        'focal_length': {
            'min' : 520,
            'max' : 530,
        },
        'delta_optical_center': {
            'min' : 0.0,
            'max' : 0.0,
        },
        'radius': {
            'min' : 0.2,
            'max' : 0.2,
        },
        'azimuth': {
            'min' : 0.0,
            'max' : 360.0,
        },
        'elevation': {
            'min' : 0.0,
            'max' : 180.0,
        },
        'roll': {
            'min' : 0.0,
            'max' : 0.0,
        },
        'x': {
            'min' : 0.0,
            'max' : 0.0,
        },
        'y': {
            'min' : 0.0,
            'max' : 0.0,
        },
        'im_width': 600,
        'im_height': 600
    }

    # Load a trimesh and sample points from it
    # m = trimesh.load_mesh('data/bar_clamp.obj')
    m = trimesh.load_mesh('data/demon_helmet.obj')
    #m = trimesh.load_mesh('data/73061.obj') # bad
    #m = trimesh.load_mesh('data/grip.obj')
    #m = trimesh.load_mesh('data/254883.obj')
    #m = trimesh.load_mesh('data/418149.obj') # slow
    #m = trimesh.load_mesh('data/2580763.obj') # bad
    #m = trimesh.load_mesh('data/294517.obj') # tough

    run_test_cases(m, pcs_cfg, pcg_cfg, 3)

    return
    #m = trimesh.load_mesh('data/73061.obj')
    #points, triinds = trimesh.sample.sample_surface_even(m, 100000)
    points = sample_from_edges(m)
    print len(points)
    pcmesh = PointCloud(points.T, frame="mesh")

    # Sample a point cloud from a depth image
    #pcg = PointCloudGenerator(pcg_cfg)
    #point_clouds, _, obj_to_cam_poses = pcg.generate_worksurface_point_clouds(m, 1)
    #pcrender = point_clouds[0]
    #T_obj_camera = obj_to_cam_poses[0]
    T_cam_to_objs = ViewsphereDiscretizer.get_camera_poses(vsp_cfg, 'obj')
    T_objs_to_camera = [T.inverse() for T in T_cam_to_objs]

    T_obj_camera = T_objs_to_camera[0]
    data = sample_from_edges(m, sigma=0.0002)
    data = data[data[:,1] > 0.015]
    pcrender = T_obj_camera * PointCloud(data.T, frame='obj')
    #pcrender = T_obj_camera.as_frames('mesh','camera') * pcmesh

    aligner = Super4PCSAligner(pcs_cfg)
    T_obj_camera_aligned = aligner.align(pcrender, pcmesh)

    mtf = m.copy()
    mtf.apply_transform(T_obj_camera_aligned.matrix)
    motf = m.copy()
    motf.apply_transform(T_obj_camera.matrix)

    vis3d.figure()
    vis3d.points(T_obj_camera_aligned.as_frames('mesh','camera') * pcmesh, color=(0.5,0.5,0.5), scale=0.001)
    vis3d.points(pcrender, scale=0.001, color=(0,0,1))
    vis3d.mesh(mtf, style='surface')
    vis3d.mesh(motf, style='surface', color=(0,0,1))
    vis3d.show()

if __name__ == '__main__':
    main()


import numpy as np
import trimesh
from autolab_core import RigidTransform, PointCloud
from visualization import Visualizer3D as vis
from registration import PointCloudGenerator, Super4PCSAligner

def sample_from_edges(x):
    vec = np.random.multivariate_normal(np.zeros(3), np.eye(3))
    vec = vec / np.linalg.norm(vec)

    edges = x.face_adjacency
    verts = x.vertices[x.face_adjacency_edges]
    n1 = x.face_normals[edges][:,0]
    n2 = x.face_normals[edges][:,1]
    d1 = np.einsum('ij,ij->i', n1, vec-verts[:,0])
    d2 = np.einsum('ij,ij->i', n2, vec-verts[:,1])
    edge_inds = np.arange(len(edges))[d1*d2 < 0]

    sampled_points = []
    for edge_ind in edge_inds:
        verts = x.vertices[x.face_adjacency_edges[edge_ind]]
        points = np.random.uniform(size=(1,))
        sampled_points.extend([verts[0] + (verts[1] - verts[0]) * p for p in points])
    return np.array(sampled_points)


def main():
    pcs_cfg = {
        'overlap' : 0.8,
        'accuracy': 0.0005,
        'samples' : 2000,
        'timeout' : 3000,
        'cache_dir' : './.cache'
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
            'min' : 0.5,
            'max' : 0.7,
        },
        'azimuth': {
            'min' : 0.0,
            'max' : 360.0,
        },
        'elevation': {
            'min' : 0.10,
            'max' : 10.0,
        },
        'roll': {
            'min' : -0.2,
            'max' : 0.2,
        },
        'x': {
            'min' : -0.01,
            'max' : 0.01,
        },
        'y': {
            'min' : -0.01,
            'max' : 0.01,
        },
        'im_width': 600,
        'im_height': 600
    }

    # Load a trimesh and sample points from it
    m = trimesh.load_mesh('data/bar_clamp.obj')
    m = trimesh.load_mesh('data/73061.obj')
    #points, triinds = trimesh.sample.sample_surface_even(m, 100000)
    points = sample_from_edges(m)
    pcmesh = PointCloud(points.T, frame="mesh")

    # Sample a point cloud from a depth image
    #pcg = PointCloudGenerator(pcg_cfg)
    #point_clouds, _, obj_to_cam_poses = pcg.generate_worksurface_point_clouds(m, 1)
    #pcrender = point_clouds[0]
    #T_obj_camera = obj_to_cam_poses[0]

    T_obj_camera = RigidTransform(translation=np.array([0.1, 0.05, 0.05]), from_frame='obj', to_frame='camera')
    pcrender = T_obj_camera * PointCloud(sample_from_edges(m).T, frame='obj')
    #pcrender = T_obj_camera.as_frames('mesh','camera') * pcmesh

    aligner = Super4PCSAligner(pcs_cfg)
    T_obj_camera_aligned = aligner.align(pcrender, pcmesh)

    mtf = m.copy()
    mtf.apply_transform(T_obj_camera_aligned.matrix)
    motf = m.copy()
    motf.apply_transform(T_obj_camera.matrix)

    vis.figure()
    vis.points(T_obj_camera_aligned.as_frames('mesh','camera') * pcmesh, color=(0.5,0.5,0.5), scale=0.001)
    vis.points(pcrender, scale=0.001, subsample=10, color=(0,0,1))
    vis.mesh(mtf, style='surface')
    vis.mesh(motf, style='surface', color=(0,0,1))
    vis.show()

if __name__ == '__main__':
    main()


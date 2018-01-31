import numpy as np
import trimesh
from autolab_core import RigidTransform, PointCloud
from visualization import Visualizer3D as vis
from registration import PointCloudGenerator, Super4PCSAligner

def main():
    pcs_cfg = {
        'overlap' : 0.7,
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
    points, triinds = trimesh.sample.sample_surface_even(m, 100000)
    pcmesh = PointCloud(points.T, frame="mesh")

    # Sample a point cloud from a depth image
    pcg = PointCloudGenerator(pcg_cfg)
    point_clouds, _, obj_to_cam_poses = pcg.generate_worksurface_point_clouds(m, 1)
    pcrender = point_clouds[0]
    T_obj_camera = obj_to_cam_poses[0]

    aligner = Super4PCSAligner(pcs_cfg)
    T_obj_camera_aligned = aligner.align(pcrender, pcmesh)

    mtf = m.copy().apply_transform(T_obj_camera_aligned.matrix)
    motf = m.copy().apply_transform(T_obj_camera.matrix)

    vis.figure()
    vis.mesh(mtf, style='surface')
    vis.mesh(motf, style='surface', color=(0,0,1))
    vis.show()

if __name__ == '__main__':
    main()


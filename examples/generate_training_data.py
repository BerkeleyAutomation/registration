import numpy as np
import trimesh
import argparse
import os
import pathlib2
from visible_edges import compute_salient_edges
from registration import PointCloudGenerator

def point_normal_cloud(depth_image, camera_intr):
    """Computes a point normal cloud of the depth image

    Parameters
    ----------
    camera_intr : CameraIntrinsics object
        The camera parameters from which this was taken

    Returns
    -------
    autolab_core.PointNormalCloud
        A point normal cloud from the depth image
    """
    point_cloud_im = camera_intr.deproject_to_image(depth_image)
    return point_cloud_im.normal_cloud_im()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="the input folder of mesh files")
    parser.add_argument("-od", "--output_depth", type=str, help="the output folder for depth images")
    parser.add_argument("-om", "--output_mask", type=str, help="the output folder for corresponding edge masks")
    parser.add_argument("-n", "--num_views", type=int, help="the number of views to render each mesh from", default=3)
    args = parser.parse_args()

    input_folder = args.input
    output_depth_folder = args.output_depth
    output_mask_folder = args.output_mask
    num_views = args.num_views

    if not os.path.isdir(input_folder):
        print("Must provide a valid input folder")
        exit()
    
    pathlib2.Path(output_depth_folder).mkdir(exist_ok=True)
    pathlib2.Path(output_mask_folder).mkdir(exist_ok=True)

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
            'min' : 0.4,
            'max' : 0.6,
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
        'im_width': 256,
        'im_height': 256
        }

    pcg = PointCloudGenerator(pcg_cfg)
    print "start"
    filename_counter = 0
    for mesh_filepath in os.listdir(input_folder):
        print mesh_filepath
        full_path = os.path.join(input_folder, mesh_filepath)
        m = trimesh.load_mesh(full_path)
        edge_mask = compute_salient_edges(m, 300)
        print "starting render"

        _, dis, _, cis, true_edge_masks = pcg.generate_worksurface_point_clouds(m, edge_mask, num_views)
        for i in range(num_views):
            print('view:', i)
            di, ci, true_edge_mask = dis[i], cis[i], true_edge_masks[i]
            normals = point_normal_cloud(di, ci)
            import pdb
            pdb.set_trace()
            
            np.save(os.path.join(output_depth_folder, str(filename_counter)), di.data)
            np.save(os.path.join(output_mask_folder, str(filename_counter)), true_edge_mask.data)
            print (mesh_filepath, filename_counter)
            filename_counter += 1

import numpy as np
from autolab_core import RigidTransform, PointCloud, YamlConfig
from perception import BinaryImage
from registration import Super4PCSAligner, RegistrationExample
from keras.models import load_model
import argparse
import os


def generate_canonical_pc(mesh, edge_mask, n_points=200):
    edges = mesh.vertices[mesh.face_adjacency_edges][edge_mask]
    lengths = np.linalg.norm(edges[:,0] - edges[:,1], axis=1)
    ppe = np.array(lengths / np.sum(lengths) * n_points, dtype=np.int32) + 1

    sampled_points = []
    for edge, n_samples in zip(edges, ppe):
        points = np.random.uniform(size=(n_samples,))
        sampled_points.extend([edge[0] + (edge[1] - edge[0]) * p for p in points])
    sampled_points = np.array(sampled_points)
    return sampled_points

def register_example(reg_example, aligner, model):
    ses = reg_example.salient_edge_set

    # Pre-process mesh
    mesh = ses.mesh
    edge_mask = ses.edge_mask
    edge_pc_obj = PointCloud(generate_canonical_pc(mesh, edge_mask).T, frame='obj')

    # Process Depth Image
    ci = reg_example.camera_intrs
    depth_im = reg_example.depth_im
    point_cloud_im = ci.deproject_to_image(depth_im)
    normal_cloud_im  = point_cloud_im.normal_cloud_im()
    joined = np.dstack((depth_im.data, normal_cloud_im.data))
    mask = model.predict(joined[np.newaxis, :, :, :])[0]
    mask *= 255.0
    mask = BinaryImage(mask.astype(np.uint8))

    depth_im_edges = depth_im.mask_binary(mask)
    edge_pc_cam = ci.deproject(depth_im_edges)
    edge_pc_cam.remove_zero_points()

    # Align the two point sets with Super4PCS
    T_obj_camera_est = aligner.align(edge_pc_cam, edge_pc_obj)

    return T_obj_camera_est

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    args = parser.parse_args()

    config_filename = args.config_filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                                       'cfg/tools/evaluate_registration_examples.yaml')
    cfg = YamlConfig(config_filename)
    input_dir = cfg['examples_dir']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['cuda_device'])

    aligner = Super4PCSAligner(cfg['super4pcs_config'])
    model = load_model(cfg['model_file'])

    for filepath in os.listdir(input_dir):
        full_path = os.path.join(input_dir, filepath)
        example = RegistrationExample.load(full_path)
        T_obj_camera_est = register_example(example, aligner, model)

        print('Error: {}'.format(example.alignment_error(T_obj_camera_est)))
        example.visualize_alignment(T_obj_camera_est)

if __name__ == '__main__':
    main()



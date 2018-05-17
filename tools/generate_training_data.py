import numpy as np
import argparse
import os

from autolab_core import YamlConfig

from registration import SalientEdgeSet, SalientEdgeImageGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    args = parser.parse_args()

    config_filename = args.config_filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                                       'cfg/tools/generate_training_data.yaml')
    cfg = YamlConfig(config_filename)
    input_dir = cfg['edge_set_dir']
    output_dir = cfg['output_dir']

    image_generator = SalientEdgeImageGenerator(cfg['generator_config'])

    filename_counter = 0
    for zip_filepath in os.listdir(input_dir):
        full_path = os.path.join(input_dir, zip_filepath)
        ses = SalientEdgeSet.load(full_path)
        depth_ims, normal_ims, edge_masks = image_generator.generate_images(ses, cfg['n_samples_per_mesh'])

        for di, ni, em in zip(depth_ims, normal_ims, edge_masks):
            fn = os.path.join(output_dir, '{}.npz'.format(filename_counter))
            filename_counter += 1
            np.savez(fn, depth=di.data.astype(np.float32),
                         normals=ni.data.astype(np.float32),
                         mask=em.data.astype(np.float32) / 255.0)

if __name__ == "__main__":
    main()

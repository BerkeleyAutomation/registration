import numpy as np
import trimesh
import argparse
import os

from autolab_core import YamlConfig

from registration import SalientEdgeSet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    args = parser.parse_args()

    config_filename = args.config_filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                                       'cfg/tools/generate_salient_edge_sets.yaml')

    cfg = YamlConfig(config_filename)
    mesh_dir = cfg['mesh_dir']
    output_dir = cfg['output_dir']

    filename_counter = len(os.listdir(output_dir))
    for mesh_filepath in os.listdir(mesh_dir):
        print(mesh_filepath)
        full_path = os.path.join(mesh_dir, mesh_filepath)
        m = trimesh.load_mesh(full_path)
        if len(m.faces) > cfg['max_n_faces']:
            continue
        ses = SalientEdgeSet(m, cfg['salient_edge_set_cfg'])

        ses.save(os.path.join(output_dir, '{}.zip'.format(filename_counter)))
        filename_counter += 1

if __name__ == "__main__":
    main()


import numpy as np
import argparse
import os

from autolab_core import YamlConfig

from registration import SalientEdgeSet, RegistrationExample, RegistrationExampleGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    args = parser.parse_args()

    config_filename = args.config_filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                                       'cfg/tools/generate_test_cases.yaml')
    cfg = YamlConfig(config_filename)
    input_dir = cfg['edge_set_dir']
    output_dir = cfg['output_dir']

    example_generator = RegistrationExampleGenerator(cfg['generator_config'])

    filename_counter = 0
    for ses_filepath in os.listdir(input_dir):
        full_path = os.path.join(input_dir, ses_filepath)
        examples = example_generator.generate_examples(full_path, cfg['n_samples_per_mesh'])

        for example in examples:
            fn = os.path.join(output_dir, '{}.pkl'.format(filename_counter))
            filename_counter += 1
            example.save(fn)

if __name__ == "__main__":
    main()


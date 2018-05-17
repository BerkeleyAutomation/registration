import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

import argparse
import numpy as np
import os

from autolab_core import YamlConfig

class DataGenerator(keras.utils.Sequence):

    def __init__(self, filenames, batch_size=2, dim=(256, 256), n_channels=4, shuffle=True):
        self.filenames = filenames
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indices = np.arange(len(filenames))
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.filenames))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        length = int(np.floor(len(self.filenames) / self.batch_size))
        return length

    def __getitem__(self, index):
        # Generate indices of batch
        batch_inds = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        filenames = [self.filenames[i] for i in batch_inds]
        X, y = self.__data_generation(filenames)
        return X, y

    def __data_generation(self, filenames):
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.dim[0], self.dim[1]))

        for i, fn in enumerate(filenames):
            npz = np.load(open(fn))
            depth = npz['depth']
            normals = npz['normals']
            mask = npz['mask']

            X[i,:,:,0] = depth
            X[i,:,:,1:4] = normals
            y[i,:,:] = mask

        y = y[:,:,:,np.newaxis]
        return X, y

def get_edge_detection_model(config):
    model = Sequential()
    n_filters = config['n_filters']
    kernel_size = (config['kernel_size'], config['kernel_size'])
    input_shape = (config['input_width'],
                    config['input_height'],
                    config['input_channels'])
    model.add(Conv2D(n_filters, kernel_size=kernel_size, input_shape=input_shape,
                        data_format='channels_last', padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(1, kernel_size=1, padding='same'))
    model.add(Activation('sigmoid'))
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    args = parser.parse_args()

    # Load Config
    config_filename = args.config_filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                                       'cfg/tools/train_edge_detection_net.yaml')
    cfg = YamlConfig(config_filename)
    data_dir = cfg['data_dir']
    output_fn = cfg['output_file']
    validation_pct = cfg['validation_pct']
    model_cfg = cfg['model_cfg']
    training_cfg = cfg['training_cfg']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(training_cfg['cuda_device'])

    # Create data generators
    filenames = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)]
    n_data_points = len(filenames)
    n_train = int(np.floor((1.0 - validation_pct) * n_data_points))
    training_files = filenames[:n_train]
    validation_files = filenames[n_train:]
    training_gen = DataGenerator(training_files)
    validation_gen = DataGenerator(validation_files)

    # Create model
    model = get_edge_detection_model(model_cfg)

    # Train model
    model.compile(loss=binary_crossentropy,
                  optimizer=Adam(lr=training_cfg['learning_rate']),
                  metrics=['accuracy'])

    model.fit_generator(generator=training_gen,
                        validation_data=validation_gen,
                        #use_multiprocessing=True,
                        #workers=2,
                        verbose=1,
                        epochs=training_cfg['num_epochs'])

    # Save model
    model.save(output_fn)

if __name__ == '__main__':
    main()

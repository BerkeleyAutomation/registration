from keras.models import Sequential
from keras.layers import Conv2D, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import mean_squared_error, binary_crossentropy
from keras.optimizers import Adam
import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir
import os
from keras.utils import plot_model
from perception import GrayscaleImage, BinaryImage
from visualization import Visualizer2D as vis2d

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def generate_training_data(filepath):
    sigma = 0.3
    max_size = 256
    
    num_samples = len(list(listdir(filepath)))
    x = np.zeros((num_samples, max_size, max_size, 3))
    y = np.zeros((num_samples, max_size, max_size))
    
    for i, imgpath in enumerate(listdir(filepath)):
        img = cv2.imread(filepath + imgpath)
        
        height, width = img.shape[:2]
        if max_size < height or max_size < width:
            img = cv2.resize(img, (max_size, max_size), interpolation=cv2.INTER_AREA)
        
        med = np.median(img)
        lower_threshold = int(max(0, (1.0 - sigma) * med))
        upper_threshold = int(min(255, (1.0 + sigma) * med))

        edges = cv2.Canny(img, 170, 220)#lower_threshold, upper_threshold)
        x[i,:,:,:] = img / 256.0
        y[i,:,:] = edges > 126

    y = y[:,:,:,np.newaxis]
    return x, y
        

train_data, train_labels = generate_training_data('./data/rgb_images/')

window_size = (7,7)
input_size = (256,256,3)
num_epochs = 5000

model = Sequential()

model.add(Conv2D(30, kernel_size=window_size, input_shape=input_size, 
                 data_format='channels_last', padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(1, kernel_size=1, padding='same'))
model.add(Activation('sigmoid'))

model.compile(loss=binary_crossentropy,
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])

model.fit(train_data, train_labels, verbose=1, epochs=num_epochs)

test_data, test_labels = generate_training_data('./data/test/')
predicted_edge_mask = model.predict(test_data)

predicted_edge_mask = np.array(predicted_edge_mask[0] * 255, dtype=np.uint8)
gi = BinaryImage(predicted_edge_mask)
actual_edge_mask = np.array(test_labels[0] * 255, dtype=np.uint8)
bi = BinaryImage(actual_edge_mask)
vis2d.figure()
vis2d.subplot(121)
vis2d.imshow(gi)
vis2d.subplot(122)
vis2d.imshow(bi)
vis2d.show()







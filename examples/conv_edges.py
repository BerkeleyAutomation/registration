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
from perception import GrayscaleImage, BinaryImage, DepthImage
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
        

def depth_image_train_data(depth_image_path, edge_mask_path,  n, dim=256):
    x = np.zeros((n, dim, dim))
    y = np.zeros((n, dim, dim))

    for i in range(n):
        di = np.load(depth_image_path + str(i) + '.npy')
        edge_mask = np.load(edge_mask_path + str(i) + '.npy')

        x[i, :, :] = di
        y[i, :, :] = edge_mask
        
    # rescale depth image to be between 0 and 1
    max_depth = np.max(x[x > 0])  
    
    x[x == 0] += max_depth + 0.1 
    min_depth = np.min(x[x > 0])
    max_depth = np.max(x[x > 0])
    x = (x - min_depth) / (max_depth - min_depth)

    x = x[:,:,:,np.newaxis]
    y = y[:,:,:,np.newaxis] / 255.0
    return x, y

def depth_image_normals_train_data(x_path, y_path, n, dim=256): 
    x = np.zeros((n, dim, dim, 4))
    y = np.zeros((n, dim, dim))

    for i in range(n):
        di = np.load(x_path + str(i) + '.npy')
        edge_mask = np.load(y_path + str(i) + '.npy')
        #rescale the depth image 
        min_depth = np.min(di[di > 0])
        max_depth = np.max(di[di > 0])
        di = (di - min_depth) / (max_depth - min_depth)
        x[i, :, :, :] = di
        y[i, :, :] = edge_mask

    y = y[:, :, :, np.newaxis] / 255.0
    return x,y

# train_data, train_labels = generate_training_data('./data/rgb_images/')
#data, labels = depth_image_train_data('./data/depth_images/', './data/edge_masks/', n=2000)
data, labels = depth_image_normals_train_data('./data/depth_normal/', './data/edge_normal/', n=2000)
percent_train = 0.1
split_idx = int(percent_train * data.shape[0])
train_data, train_labels = data[:split_idx, :, :, :], labels[:split_idx, :, :, :]
test_data, test_labels = data[split_idx:, : ,: , :], labels[split_idx:, :, :, :]

window_size = (7,7)
input_size = (256,256, 4)
# input_size = (256, 256, 3)
# input_size = (256, 256, 1)
num_epochs = 100

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

# test_data, test_labels = generate_training_data('./data/test/')

predicted_edge_mask = model.predict(test_data)

predicted_edge_mask = np.array(predicted_edge_mask[3] * 255, dtype=np.uint8)
gi = BinaryImage(predicted_edge_mask)
actual_edge_mask = np.array(test_labels[3] * 255, dtype=np.uint8)
bi = BinaryImage(actual_edge_mask)
vis2d.figure()
vis2d.subplot(121)
vis2d.imshow(gi)
vis2d.subplot(122)
vis2d.imshow(bi)
vis2d.show()







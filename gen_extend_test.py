# Standardize images across the dataset, mean=0, stdev=1
from keras.datasets import mnist
from keras.preprocessing.image_extend import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
import os

input_dir = os.getcwd()+'/input/'
# define data preparation
datagen = ImageDataGenerator(
            horizontal_flip=True,
            rescale = 1./255.)

input_dir_grip = input_dir + 'sword_grip/'
input_dir_train = input_dir + 'sword_train/'


flow_grip = datagen.flow_from_directory(input_dir_grip, 
        class_mode='image', 
        read_formats={'png'}, 
        batch_size=2,
        sub_directory_list=['grip'],
        sub_directory_y='train')
# fit parameters from data
datagen.fit_generator(flow_grip)

# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow_from_directory(X_train, y_train, batch_size=9):
    # create a grid of 3x3 images
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    # show the plot
    pyplot.show()
    break
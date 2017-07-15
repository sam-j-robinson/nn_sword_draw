import keras as k
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Conv2D, multiply
from keras.layers.advanced_activations import PReLU
<<<<<<< HEAD

#from image_extend import ImageDataGenerator
=======
>>>>>>> 316289f4d5b7967362046b42c7befe08822929df
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from scipy import misc
import os
<<<<<<< HEAD
#/Users/figaro/anaconda2/envs/py36/lib/python3.6/site-packages/tensorflow/contrib/keras/python/keras/preprocessing
#/Users/figaro/anaconda2/envs/py36/lib/python3.6/site-packages/keras/preprocessing/image.py

#print(os.path.abspath(k.preprocessing.image.__file__))
# print(os.path.abspath(image_extend.__file__))

# raise Exception("testing")


seed = 1
input_dir = os.getcwd()+'/input/'
x_list = ['magic', 'grip', 'blade']
y_sub_dir = 'train'

x_list = ['x', 'y', 'z']

datagen = ImageDataGenerator(horizontal_flip=True)

img_flow = datagen.flow_from_directory(directory='input',
        class_mode='image', 
        target_size = (512,128),
        batch_size=50,
        sub_directory_list=['magic', 'grip', 'magic', 'blade'],
        sub_directory_y='train',
        color_mode = 'rgba',
        input_format='png')



input_magic = Input(shape=(512,128,4))
input_magic2 = Input(shape=(512,128,4))
input_grip = Input(shape=(512,128,4))
input_blade = Input(shape=(512,128,4))

#magic_branch = Input(shape=(512,128,4))
magic_branch = Conv2D(filters=128, kernel_size=(3,3), padding='same')(input_magic)
magic_branch = PReLU()(magic_branch)
magic_branch = Conv2D(filters=64, kernel_size=(2,2), padding='same')(magic_branch)
magic_branch = PReLU()(magic_branch)

#grip_branch = Input(shape=(512,128,4))
grip_branch = Conv2D(filters=128, kernel_size=(3,3), padding='same')(input_grip)
grip_branch = PReLU()(grip_branch)
grip_branch = Conv2D(filters=64, kernel_size=(2,2), padding='same')(grip_branch)
grip_branch = PReLU()(grip_branch)

merge_grip - multiply([grip_branch, magic_branch])
merge_grip = Conv2D(filters=64, kernel_size=(2,2), padding='same')(merge_grip)
merge_grip = PReLU()(merge_grip)

#magic branch with blade
magic_branch2 = Conv2D(filters=128, kernel_size=(3,3), padding='same')(input_magic2)
magic_branch2 = PReLU()(magic_branch2)
magic_branch2 = Conv2D(filters=64, kernel_size=(2,2), padding='same')(magic_branch2)
magic_branch2 = PReLU()(magic_branch2)

#blade_branch = Input(shape=(512,128,4))
blade_branch = Conv2D(filters=128, kernel_size=(3,3), padding='same')(input_blade)
blade_branch = PReLU()(blade_branch)
blade_branch = Conv2D(filters=64, kernel_size=(2,2), padding='same')(input_blade)
blade_branch = PReLU()(blade_branch)

merge_blade - multiply([grip_branch, magic_branch])
merge_blade = Conv2D(filters=64, kernel_size=(2,2), padding='same')(merge_blade)
merge_blade = PReLU()(merge_blade)

merge_branch = multiply([merge_blade, merge_grip])
merge_branch = Conv2D(filters=4, kernel_size=(2,2), padding='same')(merge_branch)

final_model = Model(inputs = [input_magic,input_grip,input_blade], outputs = merge_branch)

sgd = SGD(lr=0.1, momentum=0.9, decay=0.000001)
final_model.compile(optimizer=sgd, loss='mean_squared_error')


final_model.fit_generator(
    img_flow,
    steps_per_epoch=50,
    epochs=1,
=======

from image_adjust.py import return_sword_img_names

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
seed = 1

input_dir = os.getcwd()+'/input/'


#Part list must explicitly align with the way the data is input into the model
def img_generator(epochs=1, sample_size=10, n_vars=5, part_list=None, hue_range, seed_start):
    random.seed(seed_start)

    values_sword = random.sample(range(1,n_vars), sample_size)
    values_hue = random.sample(hue_range, sample_size)

    output = []
    for part in part_list:
        part_file_names = []


magic_branch = Conv2D(filters=16, kernel_size=(3,3), padding='same')
grip_branch = Conv2D(filters=16, kernel_size=(3,3), padding='same')
#magic_branch = Conv2D(filters=16, kernel_size=(3,3), padding='same')
blade_branch = Conv2D(filters=16, kernel_size=(3,3), padding='same')

final_model = multiply([magic_branch, grip_branch, blade_branch])
final_mode.compile(optimzer=sgd,
                    loss='mean_squared_error')
sgd = SGD(lr=0.1, momentum=0.9, decay=0.000001)

final_model.fit_generator(
    train_generator,
    samples_per_epoch=5,
    nb_epoch=1,
    validation_data = valid_generator,
    nb_val_samples = 1,
>>>>>>> 316289f4d5b7967362046b42c7befe08822929df
    )

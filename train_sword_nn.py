import keras as k
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Conv2D, multiply, add
from keras.activations import softmax, tanh, sigmoid
from keras.layers.advanced_activations import PReLU

from image_extend import ImageDataGenerator
#from keras.models import Model
#from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import History, ModelCheckpoint, TensorBoard, ProgbarLogger, ReduceLROnPlateau

from time import time
from PIL import Image

import random
import os
#from PIL import img
#/Users/figaro/anaconda2/envs/py36/lib/python3.6/site-packages/tensorflow/contrib/keras/python/keras/preprocessing
#/Users/figaro/anaconda2/envs/py36/lib/python3.6/site-packages/keras/preprocessing/image.py

#print(os.path.abspath(k.preprocessing.image.__file__))
# print(os.path.abspath(image_extend.__file__))

# raise Exception("testing")


seed = random.seed()
input_dir = os.getcwd()+'/input/'
y_sub_dir = 'train'
sub_dir_list = ['magic', 'grip', 'magic', 'blade']

datagen = ImageDataGenerator(horizontal_flip=True, rescale=1./255.)

img_flow = datagen.flow_from_directory(directory='input',
        class_mode='image', 
        target_size = (64,16),
        batch_size=90,
        sub_directory_list=sub_dir_list,
        sub_directory_y='train',
        color_mode = 'rgba',
        input_format='png')

input_magic = Input(shape=(64,16,4))
input_magic2 = Input(shape=(64,16,4))
input_grip = Input(shape=(64,16,4))
input_blade = Input(shape=(64,16,4))

#Grip branch
magic_branch = Conv2D(filters=512, kernel_size=(2,2), padding='same')(input_magic)
magic_branch = PReLU()(magic_branch)
# magic_branch = Conv2D(filters=64, kernel_size=(2,2), activation='softmax', padding='same')(magic_branch)
# magic_branch = Conv2D(filters=128, kernel_size=(2,2), padding='same')(magic_branch)
# magic_branch = PReLU()(magic_branch)

grip_branch = Conv2D(filters=512, kernel_size=(2,2), padding='same')(input_grip)
grip_branch = PReLU()(grip_branch)
# grip_branch = Conv2D(filters=64, kernel_size=(2,2), activation='softmax', padding='same')(grip_branch)
# grip_branch = Conv2D(filters=128, kernel_size=(2,2), padding='same')(grip_branch)
# grip_branch = PReLU()(grip_branch)

merge_grip = add([grip_branch, magic_branch])
merge_grip = Conv2D(filters=512, kernel_size=(2,2), padding='same')(merge_grip)
merge_grip = PReLU()(merge_grip)
merge_grip = Conv2D(filters=256, kernel_size=(3,3), padding='same')(merge_grip)
merge_grip = PReLU()(merge_grip)

#blade branch
magic_branch2 = Conv2D(filters=512, kernel_size=(1,1), padding='same')(input_magic2)
magic_branch2 = PReLU()(magic_branch2)
# magic_branch2 = Conv2D(filters=64, kernel_size=(2,2), activation='softmax', padding='same')(input_magic2)
# magic_branch2 = Conv2D(filters=128, kernel_size=(2,2), padding='same')(input_magic2)
# magic_branch2 = PReLU()(magic_branch2)

blade_branch = Conv2D(filters=512, kernel_size=(1,1), padding='same')(input_blade)
blade_branch = PReLU()(blade_branch)
# blade_branch = Conv2D(filters=64, kernel_size=(2,2), activation='softmax', padding='same')(input_blade)
# blade_branch = Conv2D(filters=128, kernel_size=(2,2), padding='same')(input_blade)
# blade_branch = PReLU()(blade_branch)

#Final merge
merge_blade = add([blade_branch, magic_branch2])
merge_blade = Conv2D(filters=512, kernel_size=(2,2), padding='same')(merge_blade)
merge_blade = PReLU()(merge_blade)
merge_blade = Conv2D(filters=256, kernel_size=(3,3), padding='same')(merge_blade)
merge_blade = PReLU()(merge_blade)

#Deep checking
merge_branch = add([merge_blade, merge_grip])
merge_branch = Conv2D(filters=256, kernel_size=(2,2), padding='same')(merge_branch)
merge_branch = PReLU()(merge_branch)
merge_branch = Conv2D(filters=128, kernel_size=(2,2), padding='same')(merge_branch)
merge_branch = PReLU()(merge_branch)
merge_branch = Conv2D(filters=64, kernel_size=(2,2), padding='same')(merge_branch)
merge_branch = PReLU()(merge_branch)
merge_branch = Conv2D(filters=16, kernel_size=(3,3), padding='same')(merge_branch)
merge_branch = PReLU()(merge_branch)
merge_branch = Conv2D(filters=4, kernel_size=(4,4), padding='same')(merge_branch)
merge_branch = PReLU()(merge_branch)

final_model = Model(inputs = [input_magic,input_grip, input_magic2, input_blade], outputs = merge_branch)
# sgd = SGD(lr=0.9, decay=1e-6, momentum=1.9)
sgd = SGD(lr=.011, momentum=0.5)
final_model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

model_epoch = int(time())
if not os.path.exists('meta_model/%s'%model_epoch):
    os.makedirs('meta_model/%s'%model_epoch)

checkpointer = ModelCheckpoint(filepath='meta_model/%s/weights.hdf5' %model_epoch, 
                                verbose=0, save_best_only=True, monitor='loss')
reduce_learnRate = ReduceLROnPlateau(monitor='loss', factor=0.0005, patience=3, min_lr=0.001)
# tensorboard_viz = TensorBoard(log_dir='meta_model/%s/Graph' %model_epoch, histogram_freq=0, write_graph=True, write_images=True)
# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# print('To visualize tensor model copy and paste into a new terminal window at %s' %os.getcwd())
# print('tensorboard --logdir meta_model/%s/Graph' %(model_epoch))

history = final_model.fit_generator(
    img_flow,
    steps_per_epoch=30,
    epochs=180,
    callbacks = [checkpointer, reduce_learnRate],
    )

if not os.path.exists('meta_model/%s/img_output'%model_epoch):
    os.makedirs('meta_model/%s/img_output'%model_epoch)

prediction = final_model.predict_generator(img_flow, 1)

img_iter = 0
for img in prediction:
    img = Image.fromarray(img, 'RGBA')
    output_loc = 'meta_model/%s/img_output/predict_%s.png'%(model_epoch, img_iter)
    img_iter = img_iter + 1
    img.save(output_loc)
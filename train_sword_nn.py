import keras as k
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Conv2D, multiply
from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from scipy import misc
import os

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
    )

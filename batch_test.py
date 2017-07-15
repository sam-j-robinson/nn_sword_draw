import itertools
import random
from scipy import misc
#scipy.ndimage.imread
from scipy.ndimage import imread
from tqdm import tqdm
import os
from timeit import timeit
import numpy as np

random.seed()

def random_gen(sample_size, low, high):
    i = 0
    while i < sample_size:
        yield random.randint(low, high)
        i=i+1

# for value in range(1,6):
#     values_sword = random_gen(5, 1, 5)
#     values_hue = random_gen(5, 1, 20)
#     value_str = ''
#     for sword in values_sword:
#         for hue in values_hue:
#             value_str = '%ssword_%s_%s.png    ' %(value_str, sword, hue)

#     print('Batch: %s, %s' %(value, value_str))
input_dir = os.getcwd()+'/input/'

def open_sword_as_numpy(folder, num, part, hue=False):
    if hue:
        return imread('%ssword_%s_%s_%s.png' %(folder,num,part,hue))
    else:
        return imread('%ssword_%s_%s.png' %(folder,num,part))

#Part list must explicitly align with the way the data is input into the model
def img_generator(part_list, epochs=1, sample_size=10, n_vars=5, hue_range=(1,360,1), seed_start=None):
    if seed_start:
        random.seed(seed_start)
    else:
        random.seed()

    sword_vals =[random.randrange(1,n_vars) for n in range(sample_size)]
    hue_vals = [random.randrange(hue_range[0], hue_range[1], hue_range[2]) for n in range(sample_size)]

    output = []
    for pIdx in tqdm(range(len(part_list))):
        part = part_list[pIdx]
        part_files = []
        location = input_dir + 'sword_%s/' %part

        for idx in tqdm(range(sample_size)):
            if part.strip() == 'blade':
                part_files.append(open_sword_as_numpy(location, sword_vals[idx], part))
            else:
                part_files.append(open_sword_as_numpy(location, sword_vals[idx], part, hue_vals[idx])) 

        output.append(np.array(part_files)/255.)

    for part in output:
        print(part.shape)

#11.053219307999825

    #vectorize attempt:
    # def vectorized_app2_simplified(volume, roi, radius):
    # m,n,r = volume.shape    
    # x,y,z = np.ogrid[0:m,0:n,0:r]-roi
    # return (x**2+y**2+z**2) < radius**2
def img_generator_vec(part_list, epochs=1, sample_size=10, n_vars=5, hue_range=(1,360,1), seed_start=None):
    if seed_start:
        random.seed(seed_start)
    else:
        random.seed()

    sword_vals = np.array([random.randrange(1,n_vars) for n in range(sample_size)])
    hue_vals = np.array([random.randrange(hue_range[0], hue_range[1], hue_range[2]) for n in range(sample_size)])
    
    output = []
    for pIdx in tqdm(range(len(part_list))):
        part = part_list[pIdx]

        generate_file_list

        output.append(open_sword_as_numpy(location, **sword_vals, part, **hue_vals))

    for part in output:
        print(part.shape)

def test():
    img_generator(['grip', 'magic', 'blade'], sample_size = 200)
    img_generator_vec(['grip', 'magic', 'blade'], sample_size = 200)

if __name__ == '__main__':
    import timeit
    epochs = 5
    print(timeit.timeit("test()", setup="from __main__ import test", number=epochs))


from keras.preprocessing.image import ImageDataGenerator,standardize,random_transform
# input generator with standardization on
datagen_grip = ImageDataGenerator(
    horizontal_flip=True,
    seed=0,
    verbose=1)

datagen_blade = ImageDataGenerator(
    horizontal_flip=True,
    seed=0,
    verbose=1)

datagen_magic = ImageDataGenerator(
    horizontal_flip=True,
    seed=0,
    verbose=1)

# output generator with standardization off
datagen_train = ImageDataGenerator(
    horizontal_flip=True,
    seed=0)

# flow from directory is extended to support more format and also you can even use your own reader function
# here is an example of reading image data saved in csv file
# datagenX.flow_from_directory(csvFolder, image_reader=csvReaderGenerator, read_formats={'csv'}, reader_config={'target_size':(572,572),'resolution':20, 'crange':(0,100)}, class_mode=None, batch_size=1)

dgx_grip = datagen_grip.flow_from_directory(inputDir, class_mode=None, read_formats={'png'}, batch_size=2)
dgx_blade = datagen_blade.flow_from_directory(inputDir, class_mode=None, read_formats={'png'}, batch_size=2)
dgx_magic = datagen_magic.flow_from_directory(inputDir, class_mode=None, read_formats={'png'}, batch_size=2)
dgy_train = datagenX.flow_from_directory(outputDir,  class_mode=None, read_formats={'png'}, batch_size=2)

# you can now fit a generator as well
datagen_grip.fit_generator(dgx_grip, nb_iter=100)
datagen_blade.fit_generator(dgx_blade, nb_iter=100)
datagen_magic.fit_generator(dgx_magic, nb_iter=100)
datagen_train.fit_generator(dgy_train, nb_iter=100)

# here we sychronize two generator and combine it into one
train_generator = (dgx_grip+dgx_blade+dgx_magic)+dgdy

model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)
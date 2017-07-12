import itertools
import random
from scipy import misc

random.seed()

def random_gen(sample_size, low, high):
    i = 0
    while i < sample_size:
        yield random.randint(low, high)
        i=i+1


def img_generator(epochs=1, sample_size=10, n_vars=5, part_list=None, hue_range, seed_start):
    random.seed(seed_start)

    values_sword = random.sample(range(1,n_vars), sample_size)
    values_part = random.sample(hue_range, sample_size)

    output = []
    if part_list:
        for part in part_list:
            part_np = 
    else:
        raise ValueError('This generator requires a list of parts to find data')
        

for value in range(1,6):
    values_sword = random_gen(5, 1, 5)
    values_hue = random_gen(5, 1, 20)
    value_str = ''
    for sword in values_sword:
        for hue in values_hue:
            value_str = '%ssword_%s_%s.png    ' %(value_str, sword, hue)

    print('Batch: %s, %s' %(value, value_str))
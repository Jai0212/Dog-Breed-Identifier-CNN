
import tensorflow as tf
from scipy.io import loadmat

#  GPU assignemnt to prevent out of memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# name of all the dog breeds to be displayed
train_mat = loadmat('mat_files/train_list.mat')
train_mat_data = train_mat['file_list']

dog_breeds = []
breed_array = []

for dog_img_info in train_mat_data:

    index = dog_img_info[0][0].index('/')
    temp_breed = dog_img_info[0][0][10:index]

    breed_array = temp_breed.split('_')

    breed = ''
    for word in breed_array:
        breed += word[0].upper() + word[1:] + ' '

    breed = breed[0:-1]

    if breed not in dog_breeds:
        dog_breeds.append(breed)

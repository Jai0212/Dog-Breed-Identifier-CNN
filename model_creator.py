
import os
from scipy.io import loadmat
import cv2
import random
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPool2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization,
                                     InputLayer, GlobalAveragePooling2D)
from keras.optimizers.legacy import Adam

#  Loading in all the directories that will be used in the project
data_dir = 'data'

# image size, reduce if needed to run the model
image_length = 256
image_width = 256
colour = 1

train_mat = loadmat('mat_files/train_list.mat')
test_mat = loadmat('mat_files/test_list.mat')

train_mat_data = train_mat['file_list']
test_mat_data = test_mat['file_list']


#  Extarcts the 120 dog breed names, the name by which the training images have been saved
#  and the path of the directories in data
dog_breeds = []
train_img_name = []
directory_paths = []

for dog_img_info in train_mat_data:

    img_name = dog_img_info[0][0].split('/')[1]
    train_img_name.append(img_name)

    index = dog_img_info[0][0].index('/')
    breed = dog_img_info[0][0][10:index]

    if breed not in dog_breeds:
        dog_breeds.append(breed)

    if dog_img_info[0][0][:index] not in directory_paths:
        directory_paths.append(dog_img_info[0][0][:index])


# Training and Testing Data Generator
training_images = []
testing_images = []


for dir_path in directory_paths:

    path = os.path.join(data_dir, dir_path)

    for img in os.listdir(path):

        if img in train_img_name:  # if training data
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            # , cv2.IMREAD_GRAYSCALE
            new_array = cv2.resize(img_array, (image_length, image_width))

            breed = dir_path[10:]
            label = dog_breeds.index(breed)

            training_images.append([new_array, label])

            # plt.imshow(new_array)
            # plt.show()
            # print(new_array.shape)
            # break

        else:  # if testing data
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            # , cv2.IMREAD_GRAYSCALE
            new_array = cv2.resize(img_array, (image_length, image_width))

            breed = dir_path[10:]
            label = dog_breeds.index(breed)

            testing_images.append([new_array, label])


# shuffles the training and testing data for better accuracy
random.shuffle(training_images)
random.shuffle(testing_images)

# separates the image data and its label into separate arrays and converst the resulting arrays into numpy arrays
x_train = []
y_train = []

for features, label in training_images:
    x_train.append(features)
    y_train.append(label)


x_test = []
y_test = []

for features, label in testing_images:
    x_test.append(features)
    y_test.append(label)


# converts data to a numpy array and reduces matrix values
x_train = np.array(x_train).reshape(-1, image_length, image_width, colour)
x_test = np.array(x_test).reshape(-1, image_length, image_width, colour)

x_train = x_train / 255
x_test = x_test / 255


# The 5-Layerd Neural Network
model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(image_length, image_width, colour)))
model.add(MaxPooling2D())

model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Dropout(0.2))

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(120, activation='softmax'))  # first number of classifications


opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(np.asarray(x_train), np.asarray(y_train, dtype=np.int32),
                 epochs=10,
                 validation_data=(np.asarray(x_test), np.asarray(y_test, dtype=np.int32)))


# saves the model in directory models
model.save(os.path.join('models', 'dog_breed_classifier_test.h5'))

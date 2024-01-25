
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from main import dog_breeds
import tensorflow as tf

path_user_images = 'images_to_be_identified_user'

# loads the model
new_model = tf.keras.models.load_model(os.path.join('models', 'dog_breed_classifier_final.h5'))
# modify the above line to use a different model

user_img_data = []

img = 'No Image'

# predicts dog breed using the model
try:
    for img in os.listdir(path_user_images):

        user_img_data = cv2.imread(os.path.join(path_user_images, img), cv2.IMREAD_GRAYSCALE)
        # , cv2.IMREAD_GRAYSCALE
        resized_user_img_data = cv2.resize(user_img_data, (256, 256))
        prediction = new_model.predict(np.array([resized_user_img_data]))

        index = np.argmax(prediction)
        print(f'The dog in image {img} is a {dog_breeds[index]}')

        # code below is for showing the image with prediction on top
        img_data_for_showing = cv2.imread(os.path.join(path_user_images, img))
        resized_img_data_for_showing = cv2.resize(img_data_for_showing, (256, 256))
        plt.imshow(resized_img_data_for_showing)
        plt.imshow(cv2.cvtColor(resized_img_data_for_showing, cv2.COLOR_BGR2RGB))
        plt.title(f'Prediction: {dog_breeds[index]}')
        plt.axis('off')
        plt.show()
        # break

except (Exception, ):
    print(f'Error with input image - {img}. Check file type/size/dimesions')

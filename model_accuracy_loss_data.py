
import tensorflow as tf
from model_creator import *

model = tf.keras.models.load_model(os.path.join('models', 'dog_breed_classifier.h5'))

# for plotting the loss graph
fig = plt.figure()
plt.plot(hist.history['loss'], color='orange', label='loss')
# plt.plot(hist.history['val_loss'], color='teal', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# for plotting the accuracy graph
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
# plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# for plotting the loss and val_loss graph
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# for plotting the accuracy and val_accuracy graph
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()

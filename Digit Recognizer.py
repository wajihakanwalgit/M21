import keras
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Train Dataset(x):",x_train.shape)
print("Train Dataset(y):",y_train.shape)
print("Test Dataset(x):",x_test.shape)
print("Test Dataset(y):",y_test.shape)

for i in range(9):
    plt.subplot(330,1,i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.show()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

from tensorflow.keras.utils import to_categorical
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

opt=SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test))
model.save('mnist.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from numpy import  argmax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
img = image.load_img('C:/Users/PC/Desktop/image.jpg', target_size=(28, 28))

def load_image(filename):
    img=load_img(filename, target_size=(28, 28),grayscale=True)
    img=img_to_array(img)
    img=img.reshape(1,28,28,1)
    img=img.astype('float32')
    img/=255
    return img

def run_example():
    img=load_image('C:/Users/PC/Desktop/image.jpg')
    model=load_model('mnist.h5')
    prediction=model.predict(img)
    digit=argmax(prediction)
    print('Predicted digit:', digit)
    

run_example()


    

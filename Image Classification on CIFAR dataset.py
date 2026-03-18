import keras
import tenserflow.keras as tf
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from  keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from keras import regularizers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D  
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.layers import Activation, Dropout, Flatten, Dense


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
for i in range(9):
    plt.subplot(330,3,1+i)
    plt.imshow(x_train[i],cmap=plt.get_cmap('gray'))
    plt.show()
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
model=Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32,32,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu',kernel_constraint=max_norm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu',kernel_constraint=max_norm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())
opt = SGD(learning_rate=0.01, momentum=0.9, decay=0.0002, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
from numpy import argmax
from keras.processing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.model import load_model
import numpy as np 
import cv2

img = cv2.imread("C:/Users/PC/Desktop/image.jpg")

def load_image(filename, size=(32, 32)):
    img = load_img(filename, target_size=size)
    img = img_to_array(img)
    img= np.expand_dims(img, axis=0)
    return img

def run_example():
    img= load_image('C:/Users/PC/Desktop/image.jpg')
    result = model.predict(img)
    model=load_model("classifier.h5")
    result = model.predict(img)
    print(result)
    if result[0][0] == 1:
        prediction = 'airplane'
    elif result[0][1] == 1:
        prediction = 'automobile'
    elif result[0][2] == 1:
        prediction = 'bird'
    elif result[0][3] == 1:
        prediction = 'cat'
    elif result[0][4] == 1: 
        prediction = 'deer'
    elif result[0][5] == 1:
        prediction = 'dog'
    elif result[0][6] == 1:
        prediction = 'frog'
    elif result[0][7] == 1:
        prediction = 'horse'
    elif result[0][8] == 1:
        prediction = 'ship'
    elif result[0][9] == 1:
        prediction = 'truck'
    print('Predicted:', prediction)

run_example()



